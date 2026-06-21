"""
Per-call transcript logger.

Writes one JSON file per call into the logs/transcripts/ directory.
Each file is named with the call start datetime, e.g.:
    logs/transcripts/2026-06-21_17-14-42.json

The JSON file contains a top-level object with metadata and an array of events:
    {
        "call_id": "MZ...",
        "started_at": "2026-06-21T17:14:42.665+00:00",
        "ended_at": "2026-06-21T17:20:02.560+00:00",
        "events": [
            {"ts": "...", "speaker": "bot", "text": "Hi! ..."},
            {"ts": "...", "speaker": "user", "text": "Hello"},
            ...
        ]
    }

Important properties:
- One file per call — easy to browse, archive, or ingest.
- Thread/async safe via a process-wide lock.
- The file is written in full on close() so a single well-formed JSON
  document is always produced.  A crash mid-call still leaves the
  previous calls intact.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"
_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

_WRITE_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _now_filename() -> str:
    """Return a filesystem-safe datetime string for use as a filename."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


class TranscriptLogger:
    """One instance per call. Cheap to create."""

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self._closed = False
        self._started_at = _now_iso()
        self._events: list[dict] = []
        self._filename = _now_filename()
        # Per-turn timing markers — cleared after each timing event is logged.
        self._t_user_final: float | None = None     # monotonic seconds
        self._t_llm_first_token: float | None = None

    def log_user(self, text: str) -> None:
        """Log a finalized user utterance (one row = one consolidated turn)."""
        text = (text or "").strip()
        if not text or self._closed:
            return
        self._events.append({
            "ts": _now_iso(),
            "speaker": "user",
            "text": text,
        })

    def log_bot(self, text: str) -> None:
        """Log a finalized bot utterance (one row = one consolidated turn)."""
        text = (text or "").strip()
        if not text or self._closed:
            return
        self._events.append({
            "ts": _now_iso(),
            "speaker": "bot",
            "text": text,
        })

    # ── Per-turn latency markers ─────────────────────────────────────────────
    # Call sequence per turn:
    #   1) STT calls mark_user_finalized() the moment a finalized transcript
    #      is handed to the LLM.
    #   2) LLM calls mark_llm_first_token() on the first content delta from
    #      Groq for that turn.
    #   3) TTS calls mark_tts_first_audio() right before the first audio
    #      chunk goes out to Twilio. That call writes a single 'timing' event
    #      with three deltas and clears the markers for the next turn.
    # All three are idempotent / safe to call multiple times; only the first
    # call in a turn is recorded.

    def mark_user_finalized(self) -> None:
        if self._closed:
            return
        self._t_user_final = time.monotonic()
        self._t_llm_first_token = None

    def mark_llm_first_token(self) -> None:
        if self._closed or self._t_user_final is None:
            return
        if self._t_llm_first_token is None:
            self._t_llm_first_token = time.monotonic()

    def mark_tts_first_audio(self) -> None:
        if self._closed or self._t_user_final is None:
            return
        now = time.monotonic()
        if self._t_llm_first_token is None:
            # No content was produced (e.g. tool-only turn). Skip.
            self._t_user_final = None
            return
        self._events.append({
            "ts": _now_iso(),
            "event": "timing",
            "user_final_to_llm_first_token_ms": round((self._t_llm_first_token - self._t_user_final) * 1000),
            "llm_first_token_to_tts_first_audio_ms": round((now - self._t_llm_first_token) * 1000),
            "user_final_to_tts_first_audio_ms": round((now - self._t_user_final) * 1000),
        })
        self._t_user_final = None
        self._t_llm_first_token = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        ended_at = _now_iso()

        document = {
            "call_id": self.call_id,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "events": self._events,
        }

        filepath = _TRANSCRIPTS_DIR / f"{self._filename}.json"

        # Avoid overwriting if two calls start in the same second
        counter = 1
        while filepath.exists():
            filepath = _TRANSCRIPTS_DIR / f"{self._filename}_{counter}.json"
            counter += 1

        with _WRITE_LOCK:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

        print(f"[LOG] Transcript saved → {filepath}")
