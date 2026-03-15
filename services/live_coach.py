"""
services/live_coach.py
Manages one persistent Gemini Live API session per run.

Flow:
  1. On run start  → get_or_create(run_id, system_prompt, audio_callback)
  2. On each turn  → session.send_text(prompt)  → returns transcript str
                     audio chunks arrive at audio_callback in parallel
  3. On run end    → close(run_id)

audio_callback signature:
  async def callback(chunk: bytes) -> None
  - chunk == b""  →  end-of-turn signal (no more audio for this response)
  - chunk != b""  →  raw PCM int16 @ 24 000 Hz mono

The caller (websocket_router) creates the callback as a closure over the
WebSocket, so audio is pushed to iOS as it arrives from Gemini.
On reconnect, call session.update_audio_callback(new_cb) with the new socket.
"""

import asyncio
from typing import Callable, Awaitable, Optional

from google import genai
from google.genai.types import (
    AudioTranscriptionConfig,
    Content,
    LiveConnectConfig,
    Part,
)

from config import settings

# ── Constants ──────────────────────────────────────────────
LIVE_MODEL        = "gemini-2.5-flash-native-audio-latest"
AUDIO_SAMPLE_RATE = 24_000   # Hz  — Gemini Live always outputs 24 kHz PCM int16


class LiveCoachSession:
    """
    A persistent Gemini Live API session tied to one run.

    The internal _session_loop() task stays alive for the run's duration.
    Turns are queued and processed sequentially; the session keeps its own
    conversation memory so we only need to send the current turn each time.
    """

    def __init__(self, system_prompt: str):
        self._system_prompt   = system_prompt
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._queue:          asyncio.Queue  = asyncio.Queue()
        self._ready:          asyncio.Event  = asyncio.Event()
        self._task:           Optional[asyncio.Task] = None
        self._failed:         bool           = False
        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )

    # ── Public API ─────────────────────────────────────────

    async def start(self, audio_callback: Callable[[bytes], Awaitable[None]]):
        """Connect to Gemini and start the background session loop."""
        self._audio_callback = audio_callback
        self._task = asyncio.create_task(self._session_loop())
        await self._ready.wait()
        if self._failed:
            raise RuntimeError("Gemini Live API session failed to connect")
        print(f"✅ Live session ready  model={LIVE_MODEL}")

    def update_audio_callback(self, cb: Callable[[bytes], Awaitable[None]]):
        """
        Swap in a new audio callback (called after a WebSocket reconnect).
        Thread-safe — just replaces the reference; the loop picks it up next chunk.
        """
        self._audio_callback = cb

    async def send_text(
        self,
        text: str,
        on_partial_transcript: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """
        Queue a text turn and wait for the full response.

        Returns the full output transcript (what Gemini said).
        Audio is streamed to audio_callback concurrently while we await.
        on_partial_transcript is called for each transcript chunk as it arrives,
        allowing the caller to stream text to the client before audio_done.
        Raises RuntimeError if the session has died.
        """
        if self._failed or (self._task and self._task.done()):
            raise RuntimeError("Live session is no longer active")

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((text, future, on_partial_transcript))
        return await future

    async def stop(self):
        """Gracefully terminate the session loop."""
        await self._queue.put((None, None, None))  # sentinel
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    @property
    def is_alive(self) -> bool:
        return (
            not self._failed
            and self._task is not None
            and not self._task.done()
        )

    # ── Internal session loop ──────────────────────────────

    async def _session_loop(self):
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self._system_prompt,
            output_audio_transcription=AudioTranscriptionConfig(),
        )
        try:
            async with self._client.aio.live.connect(
                model=LIVE_MODEL,
                config=config,
            ) as session:
                self._ready.set()

                while True:
                    item = await self._queue.get()
                    text, future, on_partial = item

                    # None is the stop sentinel
                    if text is None:
                        break

                    try:
                        await session.send_client_content(
                            turns=Content(
                                role="user",
                                parts=[Part(text=text)],
                            )
                        )

                        transcript_parts: list[str] = []

                        async for msg in session.receive():
                            sc = msg.server_content

                            # Stream audio chunks to iOS
                            if sc.model_turn and sc.model_turn.parts:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and self._audio_callback:
                                        await self._audio_callback(part.inline_data.data)

                            # Stream transcript chunks as they arrive
                            if sc.output_transcription and sc.output_transcription.text:
                                chunk = sc.output_transcription.text
                                transcript_parts.append(chunk)
                                if on_partial:
                                    try:
                                        await on_partial(chunk)
                                    except Exception:
                                        pass  # never let transcript callback crash the loop

                            # Turn complete — signal end of audio stream
                            if sc.turn_complete:
                                if self._audio_callback:
                                    await self._audio_callback(b"")  # end-of-turn
                                break

                        transcript = "".join(transcript_parts).strip()
                        if future and not future.done():
                            future.set_result(transcript)

                    except Exception as e:
                        print(f"❌ Live turn error: {e}")
                        # 1008 = Gemini closed the WebSocket (context limit / policy)
                        # Mark session dead so get_or_create will recreate it next turn
                        err_str = str(e)
                        if "1008" in err_str or "1011" in err_str or "policy" in err_str.lower():
                            print(f"⚠️  Gemini session terminated ({err_str[:60]}) — marking for recreation")
                            self._failed = True
                        if future and not future.done():
                            future.set_exception(e)
                        if self._failed:
                            break  # exit the while loop so the session_loop exits cleanly

        except Exception as e:
            print(f"❌ Live session failed to connect: {e}")
            self._failed = True
            self._ready.set()   # unblock anyone awaiting start()


# ── Session registry (one per run_id) ─────────────────────
_sessions: dict[int, LiveCoachSession] = {}


async def get_or_create(
    run_id:         int,
    system_prompt:  str,
    audio_callback: Callable[[bytes], Awaitable[None]],
) -> LiveCoachSession:
    """
    Return the existing live session for this run, or create a new one.
    Safe to call on reconnect — just updates the audio callback.
    """
    existing = _sessions.get(run_id)
    if existing and existing.is_alive:
        existing.update_audio_callback(audio_callback)
        print(f"🔄 Reused live session for run {run_id}")
        return existing

    # Create fresh session (new run or dead session)
    session = LiveCoachSession(system_prompt)
    await session.start(audio_callback)
    _sessions[run_id] = session
    print(f"🎙️  Live session created for run {run_id}")
    return session


async def close(run_id: int):
    """Gracefully close and remove the session for a run."""
    session = _sessions.pop(run_id, None)
    if session:
        await session.stop()
        print(f"🔚 Live session closed for run {run_id}")