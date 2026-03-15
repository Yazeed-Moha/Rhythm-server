"""
routers/whisper_router.py
Speech-to-Text using Google Cloud STT.
Authenticated via Cloud Run service account — no API key needed.
"""
import os
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from google.cloud import speech

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

ENCODING_MAP = {
    ".m4a":  speech.RecognitionConfig.AudioEncoding.MP3,
    ".mp4":  speech.RecognitionConfig.AudioEncoding.MP3,
    ".mp3":  speech.RecognitionConfig.AudioEncoding.MP3,
    ".wav":  speech.RecognitionConfig.AudioEncoding.LINEAR16,
    ".flac": speech.RecognitionConfig.AudioEncoding.FLAC,
    ".ogg":  speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
    ".webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
}


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file:     UploadFile = File(...),
    language: str        = Form(default="en"),
):
    client      = speech.SpeechClient()
    audio_bytes = await file.read()

    ext      = os.path.splitext(file.filename or "audio.m4a")[1].lower()
    encoding = ENCODING_MAP.get(ext, speech.RecognitionConfig.AudioEncoding.MP3)
    lang     = "ar-SA" if language == "ar" else "en-US"

    audio  = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=44100,
        language_code=lang,
        enable_automatic_punctuation=True,
        model="latest_long",
    )

    try:
        response = client.recognize(config=config, audio=audio)
        text = " ".join(
            r.alternatives[0].transcript
            for r in response.results
            if r.alternatives
        ).strip()
        print(f"🎤 STT ({lang}): '{text[:80]}'")
        return TranscriptionResponse(text=text)
    except Exception as e:
        print(f"❌ STT error: {e}")
        return TranscriptionResponse(text="")