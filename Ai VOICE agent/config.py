import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# STT  →  currently: Deepgram Nova-3
# To swap: point stt/deepgram_stt.py to a new provider and keep the same
#          run_stt(audio_queue, transcript_queue, barge_in_event) signature.
# ─────────────────────────────────────────────────────────────────────────────
STT_PROVIDER   = "deepgram"
DEEPGRAM_MODEL = "nova-3"

# ─────────────────────────────────────────────────────────────────────────────
# LLM  →  currently: Groq llama-3.1-8b-instant (free)
# To swap: implement llm/<provider>_llm.py with the same
#          run_llm(transcript_queue, text_queue) signature.
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER       = "groq"
# llama-3.1-8b-instant is Groq's worst tool-caller and is deprecating
# 2026-08-16. openai/gpt-oss-20b is a near-drop-in replacement with much
# better tool-calling reliability on the free tier.
# Other free-tier alternatives:
#   moonshotai/kimi-k2-instruct-0905  — best for agentic tool use
#   openai/gpt-oss-120b               — flagship replacement for llama-3.3-70b
GROQ_MODEL         = "llama-3.1-8b-instant"
GROQ_TEMPERATURE   = 0.3   # lower temp → fewer malformed tool calls

SYSTEM_PROMPT = (
    "You are a professional pharmacy assistant. You can: "
    "1) Get drug info with get_drug_info, "
    "2) Place orders with place_order, "
    "3) Look up orders with lookup_order. "
    "IMPORTANT: Always ask users to spell out their full name clearly when placing orders. "
    "Confirm all order details before processing - including customer name, drug name, and quantity. "
    "Be thorough and professional in collecting information. "
    "If a user provides a name that is unclear, ask them to spell it out letter by letter. "
    "Always confirm the complete order details before finalizing any transaction. "
    "CRITICAL: You are a voice assistant. Never use markdown formatting in your responses. "
    "Do not use asterisks, pound signs, bullet points, or any other markdown symbols. "
    "Speak in plain natural conversational English only, as your responses will be read aloud."
)

GREETING = (
    "Hello! I am your pharmacy assistant. I can help you with drug information, "
    "placing orders, and checking order status. How can I assist you today?"
)

# OpenAI-format tool definitions (matches pharmacy_functions.py exactly)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_drug_info",
            "description": (
                "Get detailed information about a specific drug including name, description, "
                "price, and quantity. Use when a customer asks about a drug, its side effects, "
                "pricing, or what it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Name of the drug in lowercase. Examples: 'aspirin', 'ibuprofen'."
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Place a new prescription order for a customer. Use when a customer wants to order "
                "or refill medication. Always verify the drug exists first and confirm all details "
                "with the customer before calling this function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's full name as provided."
                    },
                    "drug_name": {
                        "type": "string",
                        "description": "Name of the drug in lowercase. Must exist in the system."
                    }
                },
                "required": ["customer_name", "drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an existing order by its numeric ID. Use when a customer asks about "
                "a specific order or wants to check order status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The numeric order ID to look up."
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# TTS  →  currently: ElevenLabs (eleven_flash_v2_5 for low latency)
# To swap: implement tts/<provider>_tts.py with the same
#          run_tts(text_queue, twilio_ws, streamsid, barge_in_event) signature.
# ─────────────────────────────────────────────────────────────────────────────
TTS_PROVIDER             = "elevenlabs"
# Premade (default) voice IDs — accessible on Free tier. Library/shared voices
# return HTTP 402 on Free, so we default to one of the premades shipped with
# every account. Run `python list_voices.py` to see what your account has.
#   21m00Tcm4TlvDq8ikWAM  Rachel   (calm, female)
#   pNInz6obpgDQGcFmaJgB  Adam     (deep, male)
#   EXAVITQu4vr4xnSDxMaL  Sarah    (soft, female)
#   nPczCjzI2devNBz1zQrb  Brian    (warm, male)
#   9BWtsMINqrJLrRacOk9x  Aria     (expressive, female)
ELEVENLABS_VOICE_ID      = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (premade)
ELEVENLABS_MODEL_ID      = "eleven_flash_v2_5"  # ~75ms latency
ELEVENLABS_OUTPUT_FORMAT = "ulaw_8000"           # raw mulaw 8kHz → Twilio-ready, no conversion needed

# ─────────────────────────────────────────────────────────────────────────────
# Audio  (Twilio phone calls use mulaw 8kHz)
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_ENCODING = "mulaw"
SAMPLE_RATE    = 8000
BUFFER_SIZE    = 20 * 160  # 20 frames × 160 bytes each = standard mulaw chunk
