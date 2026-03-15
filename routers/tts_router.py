# """
# routers/tts_router.py
# Text-to-Speech using Gemini 2.5 Flash Lite Preview TTS via Vertex AI.
# Streams PCM chunks — iOS starts playing on first chunk for low perceived latency.
# """
# import re
# import base64
# import struct
# from fastapi import APIRouter
# from fastapi.responses import StreamingResponse, Response
# from pydantic import BaseModel
# import google.auth
# from google.auth.transport.requests import Request
# import requests as http_requests
# from config import settings
#
# router = APIRouter(prefix="/tts", tags=["tts"])
#
# TTS_MODEL   = "gemini-2.5-flash-lite-preview-tts"
# VOICE_NAME  = "Fenrir"
# TTS_REGION  = "us-central1"
# SAMPLE_RATE = 22050
# CHANNELS    = 1
# BIT_DEPTH   = 16
# CHUNK_SIZE  = 8192   # 8KB ≈ 185ms of audio at 22050Hz mono 16bit
#
#
# def _clean_for_speech(text: str) -> str:
#     replacements = [
#         (r'(\d+):(\d+)\s*/km',      r'\1 minutes \2 per kilometre'),
#         (r'(\d+):(\d+)\s*min/km',   r'\1 minutes \2 per kilometre'),
#         (r'(\d+):(\d+)\s*/k\b',     r'\1 minutes \2 per kilometre'),
#         (r'(\d+):(\d+)\s*per\s*km', r'\1 minutes \2 per kilometre'),
#         (r'\bHR\b',                 'heart rate'),
#         (r'\bhr\b',                 'heart rate'),
#         (r'\bBPM\b',                'beats per minute'),
#         (r'\bbpm\b',                'beats per minute'),
#         (r'(\d+\.?\d*)\s*km\b',     r'\1 kilometres'),
#         (r'(\d+)\s*m\b(?!\w)',      r'\1 metres'),
#         (r'\bVO2\b',                'V O 2'),
#         (r'\bPB\b',                 'personal best'),
#         (r'\bPR\b',                 'personal record'),
#         (r'\bRPE\b',                'effort level'),
#         (r'\bmin\b',                'minutes'),
#         (r'\bsec\b',                'seconds'),
#         (r'\bN/A\b',                'not available'),
#         (r'COACHING_DATA:.*',       ''),
#     ]
#     for pattern, replacement in replacements:
#         text = re.sub(pattern, replacement, text)
#     return re.sub(r'\s+', ' ', text).strip()
#
#
# def _wav_header(data_size: int) -> bytes:
#     """44-byte WAV/RIFF header for LINEAR16 PCM."""
#     byte_rate   = SAMPLE_RATE * CHANNELS * BIT_DEPTH // 8
#     block_align = CHANNELS * BIT_DEPTH // 8
#     return struct.pack(
#         '<4sI4s4sIHHIIHH4sI',
#         b'RIFF', 36 + data_size, b'WAVE',
#         b'fmt ', 16, 1, CHANNELS, SAMPLE_RATE,
#         byte_rate, block_align, BIT_DEPTH,
#         b'data', data_size
#     )
#
#
# def _get_token() -> str:
#     credentials, _ = google.auth.default(
#         scopes=["https://www.googleapis.com/auth/cloud-platform"]
#     )
#     credentials.refresh(Request())
#     return credentials.token
#
#
# class TTSRequest(BaseModel):
#     text:     str
#     language: str = "en"
#
#
# @router.post("/speak")
# async def speak(req: TTSRequest):
#     clean_text = _clean_for_speech(req.text)
#     print(f"🔊 TTS: '{clean_text[:80]}'")
#
#     try:
#         url = (
#             f"https://{TTS_REGION}-aiplatform.googleapis.com/v1/"
#             f"projects/{settings.GCP_PROJECT}/locations/{TTS_REGION}/"
#             f"publishers/google/models/{TTS_MODEL}:generateContent"
#         )
#
#         payload = {
#             "contents": [{"role": "user", "parts": [{"text": clean_text}]}],
#             "generationConfig": {
#                 "responseModalities": ["AUDIO"],
#                 "speechConfig": {
#                     "voiceConfig": {
#                         "prebuiltVoiceConfig": {"voiceName": VOICE_NAME}
#                     }
#                 }
#             }
#         }
#
#         headers = {
#             "Authorization": f"Bearer {_get_token()}",
#             "Content-Type":  "application/json",
#         }
#
#         resp = http_requests.post(url, json=payload, headers=headers, timeout=30)
#
#         if not resp.ok:
#             print(f"❌ TTS API error {resp.status_code}: {resp.text[:300]}")
#             resp.raise_for_status()
#
#         result      = resp.json()
#         audio_b64   = result["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
#         audio_bytes = base64.b64decode(audio_b64)
#         print(f"✅ TTS audio: {len(audio_bytes)} bytes, streaming {len(audio_bytes)//CHUNK_SIZE + 1} chunks")
#
#         def pcm_stream():
#             # First yield: WAV header + first audio chunk so iOS can start immediately
#             header = _wav_header(len(audio_bytes))
#             yield header + audio_bytes[:CHUNK_SIZE]
#             # Rest of audio
#             offset = CHUNK_SIZE
#             while offset < len(audio_bytes):
#                 yield audio_bytes[offset:offset + CHUNK_SIZE]
#                 offset += CHUNK_SIZE
#
#         return StreamingResponse(
#             pcm_stream(),
#             media_type="audio/wav",
#             headers={
#                 "X-Audio-Sample-Rate": str(SAMPLE_RATE),
#                 "X-Audio-Channels":    str(CHANNELS),
#                 "X-Audio-Bit-Depth":   str(BIT_DEPTH),
#                 "Transfer-Encoding":   "chunked",
#             }
#         )
#
#     except Exception as e:
#         print(f"❌ Gemini TTS error: {e}")
#         return Response(status_code=500, content=str(e).encode())
#
#
# @router.get("/voices")
# async def list_voices():
#     return {
#         "model":       TTS_MODEL,
#         "voice":       VOICE_NAME,
#         "voices":      ["Zephyr", "Aoede", "Charon", "Fenrir", "Kore", "Leda", "Orus", "Puck"],
#         "encoding":    "LINEAR16",
#         "sample_rate": SAMPLE_RATE,
#         "chunk_size":  CHUNK_SIZE,
#     }

"""
routers/tts_router.py
Text-to-Speech using Google Cloud TTS.
Fast ~200ms response. Available in all regions including europe-west1.
Authenticated via Cloud Run service account — no API key needed.
"""
import re
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from google.cloud import texttospeech

router = APIRouter(prefix="/tts", tags=["tts"])

# Best voices for a running coach — clear, warm, natural
VOICE_EN = "en-US-Journey-F"   # warm conversational female — best for coaching
VOICE_AR = "ar-XA-Wavenet-A"   # Arabic female


def _clean_for_speech(text: str) -> str:
    """Expand abbreviations so TTS pronounces them correctly."""
    replacements = [
        (r'(\d+):(\d+)\s*/km',      r'\1 minutes \2 per kilometre'),
        (r'(\d+):(\d+)\s*min/km',   r'\1 minutes \2 per kilometre'),
        (r'(\d+):(\d+)\s*/k\b',     r'\1 minutes \2 per kilometre'),
        (r'(\d+):(\d+)\s*per\s*km', r'\1 minutes \2 per kilometre'),
        (r'\bHR\b',                 'heart rate'),
        (r'\bhr\b',                 'heart rate'),
        (r'\bBPM\b',                'beats per minute'),
        (r'\bbpm\b',                'beats per minute'),
        (r'(\d+\.?\d*)\s*km\b',     r'\1 kilometres'),
        (r'(\d+)\s*m\b(?!\w)',      r'\1 metres'),
        (r'\bVO2\b',                'V O 2'),
        (r'\bPB\b',                 'personal best'),
        (r'\bPR\b',                 'personal record'),
        (r'\bRPE\b',                'effort level'),
        (r'\bmin\b',                'minutes'),
        (r'\bsec\b',                'seconds'),
        (r'\bN/A\b',                'not available'),
        (r'COACHING_DATA:.*',       ''),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text).strip()


class TTSRequest(BaseModel):
    text:     str
    language: str = "en"


@router.post("/speak")
async def speak(req: TTSRequest):
    client     = texttospeech.TextToSpeechClient()
    clean_text = _clean_for_speech(req.text)

    # Google Cloud TTS limit is 5000 bytes — truncate at sentence boundary
    if len(clean_text) > 800:
        # Find last sentence end before 800 chars
        cutoff = clean_text[:800].rfind('.')
        if cutoff > 400:
            clean_text = clean_text[:cutoff + 1]
        else:
            clean_text = clean_text[:800]
        print(f"✂️ TTS truncated to {len(clean_text)} chars")

    print(f"🔊 TTS: '{clean_text[:80]}'")

    is_arabic = req.language == "ar"
    voice_name = VOICE_AR if is_arabic else VOICE_EN
    lang_code  = "ar-XA"  if is_arabic else "en-US"

    try:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=clean_text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.05,
                pitch=0.0,
            ),
        )
        print(f"✅ TTS audio: {len(response.audio_content)} bytes ({voice_name})")
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except Exception as e:
        print(f"❌ TTS error (Journey): {e}, falling back to Standard voice")
        # Fallback to standard voice if Journey not available
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=clean_text),
                voice=texttospeech.VoiceSelectionParams(
                    language_code=lang_code,
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=1.05,
                ),
            )
            return Response(content=response.audio_content, media_type="audio/mpeg")
        except Exception as e2:
            print(f"❌ TTS fallback error: {e2}")
            return Response(status_code=500, content=str(e2).encode())


@router.get("/voices")
async def list_voices():
    return {
        "en": VOICE_EN,
        "ar": VOICE_AR,
        "options": {
            "en_female": ["en-US-Journey-F", "en-US-Neural2-F", "en-US-Wavenet-F"],
            "en_male":   ["en-US-Journey-D", "en-US-Neural2-D", "en-US-Wavenet-D"],
            "ar_female": ["ar-XA-Wavenet-A", "ar-XA-Wavenet-C"],
            "ar_male":   ["ar-XA-Wavenet-B"],
        }
    }