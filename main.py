"""
Orchestration Layer — main.py

This is the only file that knows about all three pipeline stages.
It wires them together using async queues and a shared barge-in event.

Pipeline topology (per call):

  Twilio (inbound mulaw)
       │
       ▼  audio_queue
  ┌──────────────────┐
  │  STT             │  stt/deepgram_stt.py   ← swap by changing import + run_stt impl
  │  Deepgram Nova-3 │
  └────────┬─────────┘
           │ transcript_queue          also sets barge_in_event ──────────────────┐
           ▼                                                                       │
  ┌──────────────────┐                                                             │
  │  LLM             │  llm/groq_llm.py       ← swap by changing import + run_llm impl
  │  Groq Llama 3.1  │
  └────────┬─────────┘
           │ text_queue
           ▼
  ┌──────────────────┐   ◄── barge_in_event ──────────────────────────────────────┘
  │  TTS             │  tts/elevenlabs_tts.py ← swap by changing import + run_tts impl
  │  ElevenLabs      │
  └────────┬─────────┘
           │ mulaw audio (base64-encoded media messages)
           ▼
  Twilio (outbound audio → phone speaker)
"""

import asyncio
import base64
import json
import traceback

import websockets

import config
from stt.deepgram_stt import run_stt
from llm.groq_llm import run_llm
from tts.elevenlabs_tts import run_tts, synthesize_and_stream
from logs.transcript_logger import TranscriptLogger


# ─────────────────────────────────────────────────────────────────────────────
# Twilio receiver — unchanged interface from original project
# ─────────────────────────────────────────────────────────────────────────────

async def twilio_receiver(
    twilio_ws,
    audio_queue: asyncio.Queue,
    streamsid_queue: asyncio.Queue,
) -> None:
    """
    Reads Twilio media-stream WebSocket events.
    - Extracts the stream SID from the 'start' event and puts it in streamsid_queue.
    - Forwards each inbound mulaw frame (~160 bytes / 20ms) directly to audio_queue
      so Deepgram's endpointer sees silence in isolation (size-based batching breaks
      endpointing: a 300ms pause hidden inside a 400ms chunk is never detected).
    - Guarantees a None sentinel into audio_queue on exit so the downstream STT
      sender never deadlocks on a parse error.
    """
    try:
        async for message in twilio_ws:
            try:
                data  = json.loads(message)
            except json.JSONDecodeError as e:
                print(f"[TWILIO] JSON parse error: {e}")
                continue

            event = data.get("event", "")

            if event == "start":
                streamsid = data["start"]["streamSid"]
                streamsid_queue.put_nowait(streamsid)
                print(f"[TWILIO] Stream started  SID={streamsid}")

            elif event == "media":
                media = data["media"]
                if media.get("track") == "inbound":
                    audio_queue.put_nowait(base64.b64decode(media["payload"]))

            elif event == "stop":
                print("[TWILIO] Stream stopped")
                break

    except Exception as e:
        print(f"[TWILIO] Receiver error: {type(e).__name__}: {e}")
    finally:
        # Always signal STT to shut down, even on parse errors or unexpected disconnects.
        await audio_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# Per-call handler
# ─────────────────────────────────────────────────────────────────────────────

async def twilio_handler(twilio_ws) -> None:
    """
    Spawned once per incoming Twilio call.
    Creates the queues, fires up all pipeline tasks, plays the greeting,
    then waits for the call to end.
    """
    # ── Queues — the only coupling between pipeline stages ───────────────────
    audio_queue      = asyncio.Queue()  # raw mulaw bytes: Twilio → STT
    transcript_queue = asyncio.Queue()  # transcript str:  STT    → LLM
    text_queue       = asyncio.Queue()  # text tokens:     LLM    → TTS
    streamsid_queue  = asyncio.Queue()  # stream SID extracted from Twilio start event

    # ── Shared interrupt signal ───────────────────────────────────────────────
    # STT sets this when VAD detects the user starting to speak mid-response.
    # TTS reads this to cancel in-flight synthesis and clear Twilio's audio buffer.
    barge_in_event = asyncio.Event()

    # ── Start Twilio receiver immediately so it can capture the stream SID ───
    receiver_task = asyncio.create_task(
        twilio_receiver(twilio_ws, audio_queue, streamsid_queue)
    )

    # ── Wait for Twilio to send the 'start' event ────────────────────────────
    print("[MAIN] Waiting for stream SID from Twilio...")
    streamsid = await streamsid_queue.get()

    # ── Per-call transcript logger (writes to transcripts.jsonl) ─────────────
    logger = TranscriptLogger(call_id=streamsid)

    # ── Play greeting via ElevenLabs before the pipeline starts listening ────
    print("[MAIN] Playing greeting...")
    logger.log_bot(config.GREETING)
    await synthesize_and_stream(config.GREETING, twilio_ws, streamsid)

    # ── Launch STT, LLM, TTS concurrently alongside the already-running receiver.
    # return_exceptions=True keeps a single task's crash from cancelling siblings,
    # so the downstream queues can drain their sentinels and exit cleanly.
    print("[MAIN] Pipeline active — call in progress")
    stt_task = asyncio.create_task(run_stt(audio_queue, transcript_queue, barge_in_event, logger))
    llm_task = asyncio.create_task(run_llm(transcript_queue, text_queue, logger))
    tts_task = asyncio.create_task(run_tts(text_queue, twilio_ws, streamsid, barge_in_event, logger))

    try:
        results = await asyncio.gather(
            receiver_task, stt_task, llm_task, tts_task,
            return_exceptions=True,
        )
        for name, r in zip(["receiver", "stt", "llm", "tts"], results):
            if isinstance(r, Exception):
                print(f"[MAIN] ❌ {name} task crashed: {type(r).__name__}: {r}")
    except Exception as e:
        print(f"[MAIN] ❌ Pipeline error: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # Push sentinels so any stuck task can unwind.
        try:
            audio_queue.put_nowait(None)
            transcript_queue.put_nowait(None)
            text_queue.put_nowait(None)
        except Exception:
            pass

        # Cancel anything still running so we don't leak tasks.
        for t in (receiver_task, stt_task, llm_task, tts_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(
            receiver_task, stt_task, llm_task, tts_task,
            return_exceptions=True,
        )

        try:
            await twilio_ws.close()
        except Exception:
            pass
        logger.close()
        print("[MAIN] Call ended — all tasks complete")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("[MAIN] Starting modular voice pipeline server on port 5000")
    print(f"       STT → Deepgram {config.DEEPGRAM_MODEL}")
    print(f"       LLM → Groq     {config.GROQ_MODEL}")
    print(f"       TTS → ElevenLabs {config.ELEVENLABS_MODEL_ID}")
    async with websockets.serve(twilio_handler, "localhost", 5000):
        print("[MAIN] Waiting for Twilio calls...")
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())