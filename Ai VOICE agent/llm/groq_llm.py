"""
LLM Module: Groq llama-3.1-8b-instant (free tier)

Groq exposes an OpenAI-compatible REST API, so we reuse the openai SDK
but point it at https://api.groq.com/openai/v1.

To swap provider:
  1. Create llm/<new_provider>_llm.py
  2. Implement: async def run_llm(transcript_queue, text_queue)
  3. Update the import in main.py — nothing else changes.

Contract:
  - Reads transcript strings from `transcript_queue`
  - Pushes response text CHUNKS (token-by-token) into `text_queue`
  - Executes pharmacy tool calls internally using FUNCTION_MAP
  - Maintains per-call conversation history
  - Stops cleanly when `transcript_queue` yields None (sentinel)
  - Forwards None sentinel into `text_queue` to shut down TTS
"""

import asyncio
import json

from openai import AsyncOpenAI, APIError

import config
from pharmacy_functions import FUNCTION_MAP

_FALLBACK_MESSAGE = "Sorry, I had a brief issue. Could you say that again?"

_client = AsyncOpenAI(
    api_key=config.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def run_llm(
    transcript_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
) -> None:
    """
    Main LLM coroutine.  Persists conversation history for the entire call.
    Wraps the whole loop in try/finally so the TTS sentinel always fires —
    even if a turn explodes in a way _chat_turn's own handler missed.
    """
    conversation_history = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ]

    try:
        while True:
            user_text = await transcript_queue.get()

            if user_text is None:           # sentinel → end of call
                print("[LLM] Shutting down")
                return

            print(f"[LLM] User said: '{user_text}'")
            conversation_history.append({"role": "user", "content": user_text})

            # One turn = possibly multiple Groq calls if tool use triggers a follow-up.
            # _chat_turn handles its own errors and always emits a "\n" flush.
            await _chat_turn(conversation_history, text_queue)
    finally:
        # Defense in depth — TTS must always see a None sentinel so it can exit.
        await text_queue.put(None)


async def _chat_turn(
    history: list,
    text_queue: asyncio.Queue,
) -> None:
    """
    Call Groq with streaming.  If the model requests tool calls, execute them
    and recursively call Groq again with the results.

    Crash-tolerant: any error (malformed tool call, network blip, etc.)
    is caught, a fallback message is queued, history is left consistent,
    and the turn ends with a "\n" flush so TTS does not stall.
    """
    response_text      = ""
    tool_calls_buffer: dict[int, dict] = {}   # index → {id, name, arguments}

    try:
        stream = await _client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            messages=history,
            tools=config.TOOLS,
            tool_choice="auto",
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # ── Stream text tokens straight to TTS ───────────────────────────
            if delta.content:
                response_text += delta.content
                await text_queue.put(delta.content)

            # ── Accumulate tool call fragments ───────────────────────────────
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                    buf = tool_calls_buffer[idx]
                    if tc.id:
                        buf["id"] += tc.id
                    if tc.function and tc.function.name:
                        buf["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        buf["arguments"] += tc.function.arguments

    except APIError as e:
        # Most common cause: Groq's `tool_use_failed` (model emitted malformed
        # tool-call syntax). Keep the call alive — speak a fallback and bail.
        failed = getattr(e, "body", None)
        print(f"[LLM] Groq APIError: {e}")
        if failed:
            print(f"[LLM] failed_generation body: {failed}")
        # If we got some text before the crash, keep it as the assistant turn.
        if response_text.strip():
            history.append({"role": "assistant", "content": response_text})
        else:
            # Nothing to say — push a fallback so the caller hears something.
            await text_queue.put(_FALLBACK_MESSAGE)
            history.append({"role": "assistant", "content": _FALLBACK_MESSAGE})
        await text_queue.put("\n")
        return
    except Exception as e:
        print(f"[LLM] Unexpected stream error: {type(e).__name__}: {e}")
        if response_text.strip():
            history.append({"role": "assistant", "content": response_text})
        else:
            await text_queue.put(_FALLBACK_MESSAGE)
            history.append({"role": "assistant", "content": _FALLBACK_MESSAGE})
        await text_queue.put("\n")
        return

    # ── No tool calls → add response to history and signal TTS flush ─────────
    if not tool_calls_buffer:
        if response_text:
            history.append({"role": "assistant", "content": response_text})
        await text_queue.put("\n")
        return

    # ── Tool call path ────────────────────────────────────────────────────────
    # 1. Record assistant message with tool call requests
    tool_calls_list = [
        {
            "id": buf["id"],
            "type": "function",
            "function": {"name": buf["name"], "arguments": buf["arguments"]},
        }
        for buf in (tool_calls_buffer[i] for i in sorted(tool_calls_buffer))
    ]
    history.append({
        "role": "assistant",
        "content": response_text or None,
        "tool_calls": tool_calls_list,
    })

    # 2. Execute each tool and record results
    for tc_info in tool_calls_list:
        func_name = tc_info["function"]["name"]
        try:
            arguments = json.loads(tc_info["function"]["arguments"])
        except (json.JSONDecodeError, ValueError):
            arguments = {}

        print(f"[LLM] Tool call → {func_name}({arguments})")

        try:
            if func_name in FUNCTION_MAP:
                result = FUNCTION_MAP[func_name](**arguments)
            else:
                result = {"error": f"Function '{func_name}' not found"}
        except Exception as e:
            print(f"[LLM] Tool execution error in {func_name}: {type(e).__name__}: {e}")
            result = {"error": f"{func_name} failed: {e}"}

        print(f"[LLM] Tool result ← {result}")

        history.append({
            "role": "tool",
            "tool_call_id": tc_info["id"],
            "content": json.dumps(result),
        })

    # 3. Call Groq again with tool results to get the final spoken response
    await _chat_turn(history, text_queue)
