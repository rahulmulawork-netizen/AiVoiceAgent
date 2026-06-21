"""
STT Module: Deepgram Nova-3 (streaming WebSocket — listen-only, NOT the agent endpoint)

To swap provider:
  1. Create stt/<new_provider>_stt.py
  2. Implement: async def run_stt(audio_queue, transcript_queue, barge_in_event)
  3. Update the import in main.py — nothing else changes.

Contract:
  - Reads raw mulaw bytes from `audio_queue`
  - Pushes final transcript strings into `transcript_queue`
  - Sets `barge_in_event` when VAD detects user starting to speak
  - Stops cleanly when `audio_queue` yields None (sentinel)

Finalization strategy (per Deepgram's official guidance):
  - Buffer text from every `is_final: true` Results event.
  - Flush to LLM on `speech_final: true`  → endpointer detected end of speech.
  - Flush to LLM on `UtteranceEnd` event   → word-timing-based silence backstop.
  Reference: https://developers.deepgram.com/docs/understanding-end-of-speech-detection
"""

import asyncio
import json
import time

import websockets

import config

# Deepgram STT-only WebSocket (separate from the agent endpoint).
# Both endpointing (acoustic) AND utterance_end_ms (word-timing) — whichever
# fires first finalizes the turn. Required reading:
#   https://developers.deepgram.com/docs/endpointing
#   https://developers.deepgram.com/docs/utterance-end
_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    f"?model={config.DEEPGRAM_MODEL}"
    f"&encoding={config.AUDIO_ENCODING}"
    f"&sample_rate={config.SAMPLE_RATE}"
    "&channels=1"
    "&punctuate=true"
    "&interim_results=true"   # required for utterance_end_ms
    "&vad_events=true"        # gives us SpeechStarted for barge-in detection
    "&endpointing=300"        # 300 ms silence → mark utterance as speech_final
    "&utterance_end_ms=1000"  # backstop: 1 s of word-timing silence → UtteranceEnd
    "&no_delay=true"          # commit words faster; reduces mid-utterance retraction
                              #   (e.g., "stomach upset" losing "upset" on phone audio)
)

# Send {"type":"KeepAlive"} as a text frame this often, when no audio has been
# forwarded recently. Deepgram closes the socket after 10 s of no data with
# NET-0001 — 3 s gives us comfortable margin.
#   https://developers.deepgram.com/docs/audio-keep-alive
_KEEPALIVE_INTERVAL_S = 3.0

# Ignore SpeechStarted events that arrive within this window of the previous one
# — Deepgram's VAD fires once per silence→speech transition, including breaths
# and lip closures inside a single utterance.
_BARGE_IN_DEBOUNCE_S = 0.5


async def run_stt(
    audio_queue: asyncio.Queue,
    transcript_queue: asyncio.Queue,
    barge_in_event: asyncio.Event,
) -> None:
    """
    Main STT coroutine. Connects to Deepgram, streams audio, returns transcripts.
    Runs for the lifetime of a single call.
    """
    try:
        async with websockets.connect(
            _STT_URL,
            subprotocols=["token", config.DEEPGRAM_API_KEY],
        ) as dg_ws:
            print("[STT] Connected to Deepgram STT (Nova-3)")

            # Shared mutable state between coroutines below.
            state = {
                "last_send": time.monotonic(),   # for keepalive watchdog
                "last_barge_in": 0.0,            # for barge-in debounce
                "utterance_buffer": [],          # is_final segments awaiting flush
            }

            async def _sender():
                """Pull audio from the queue and forward it to Deepgram."""
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:          # sentinel → end of call
                        try:
                            await dg_ws.send(json.dumps({"type": "CloseStream"}))
                        except Exception:
                            pass
                        return
                    try:
                        await dg_ws.send(chunk)
                        state["last_send"] = time.monotonic()
                    except Exception as e:
                        print(f"[STT] sender error: {e}")
                        return

            async def _keepalive():
                """
                Send a KeepAlive text frame whenever audio has been idle ≥3 s.
                Deepgram closes the socket after 10 s without any frame.
                """
                while True:
                    await asyncio.sleep(1.0)
                    idle = time.monotonic() - state["last_send"]
                    if idle >= _KEEPALIVE_INTERVAL_S:
                        try:
                            await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                            state["last_send"] = time.monotonic()
                        except Exception:
                            return

            async def _flush_utterance(reason: str) -> None:
                """Concatenate buffered is_final segments and push to the LLM."""
                if not state["utterance_buffer"]:
                    return
                text = " ".join(state["utterance_buffer"]).strip()
                state["utterance_buffer"] = []
                if text:
                    print(f"[STT] Final ({reason}): '{text}'")
                    await transcript_queue.put(text)

            async def _receiver():
                """Listen to Deepgram events and dispatch accordingly."""
                async for raw in dg_ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    # ── Barge-in: user started speaking ──────────────────────
                    if msg_type == "SpeechStarted":
                        now = time.monotonic()
                        if now - state["last_barge_in"] < _BARGE_IN_DEBOUNCE_S:
                            continue   # debounced — likely mid-utterance breath
                        state["last_barge_in"] = now
                        print("[STT] VAD: user started speaking → barge-in")
                        barge_in_event.set()

                    # ── Transcript result ────────────────────────────────────
                    elif msg_type == "Results":
                        channel      = data.get("channel", {})
                        alternatives = channel.get("alternatives", [{}])
                        transcript   = alternatives[0].get("transcript", "").strip()
                        is_final     = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)

                        if not transcript:
                            continue

                        if is_final:
                            # Commit this segment to the utterance buffer.
                            state["utterance_buffer"].append(transcript)
                            if speech_final:
                                # Endpointer detected end of speech — flush now.
                                await _flush_utterance("speech_final")
                            else:
                                print(f"[STT] Segment: '{transcript}'")
                        else:
                            print(f"[STT] Interim: '{transcript}'")

                    # ── UtteranceEnd: word-timing silence backstop ───────────
                    elif msg_type == "UtteranceEnd":
                        # Only flush if speech_final didn't already do it.
                        await _flush_utterance("UtteranceEnd")

                    # ── Metadata / ack — ignore silently ─────────────────────
                    elif msg_type in ("Metadata",):
                        pass

                    elif msg_type == "Error":
                        print(f"[STT] Deepgram error: {data}")

            sender_task    = asyncio.create_task(_sender())
            receiver_task  = asyncio.create_task(_receiver())
            keepalive_task = asyncio.create_task(_keepalive())

            # Wait for sender to finish (triggered by None sentinel from twilio_receiver).
            await sender_task

            # Give the receiver a short grace window to drain any in-flight
            # final Results / UtteranceEnd before cancelling — otherwise the
            # last user utterance can be lost when the call ends.
            try:
                await asyncio.wait_for(receiver_task, timeout=0.5)
            except asyncio.TimeoutError:
                receiver_task.cancel()
            except Exception:
                pass

            keepalive_task.cancel()

    except Exception as e:
        print(f"[STT] ❌ Connection to Deepgram failed: {type(e).__name__}: {e}")
        print("[STT] 💡 Check your DEEPGRAM_API_KEY in .env")

    finally:
        await transcript_queue.put(None)
        print("[STT] Disconnected from Deepgram STT")
