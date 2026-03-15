"""
routers/websocket_router.py
WebSocket coaching session — now powered by Gemini Live API.

Audio flow (per coaching turn):
  1. coach_service builds a text prompt (vitals + user message or event context)
  2. live_coach.LiveCoachSession.send_text() sends it to Gemini Live API
  3. Gemini streams PCM audio chunks → _audio_callback → iOS receives audio_chunk messages
  4. Gemini also produces a text transcript → returned to us → saved to history & sent as coach message

New outgoing message types (server → iOS):
  audio_chunk  { data: <base64 PCM int16>, sample_rate: 24000, encoding: "pcm_s16le" }
  audio_done   {}   ← end-of-turn signal; iOS can stop buffering

Existing message types are unchanged so the rest of the iOS app keeps working:
  coach        { message, urgency, suggested_pace, coaching_action }
  analysis     { text }
  run_ended    { run_id, total_distance_km, ... }
  reconnected  { elapsed_minutes }
  ping / pong
  error        { detail }
"""

import asyncio
import base64
import json
import time
from typing import Callable, Awaitable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import VitalSignInput, WSOutgoing
from services import coach_service, run_service
from services import live_coach
from services.run_state import RunState
from services.interval_manager import check_interval_events, check_steady_events

router = APIRouter()

# In-memory run states — survive reconnects within the same Cloud Run instance
_active_sessions: dict[int, RunState] = {}


# ── Helpers ────────────────────────────────────────────────

async def send(ws: WebSocket, msg_type: str, payload: dict):
    try:
        await ws.send_text(WSOutgoing(type=msg_type, payload=payload).model_dump_json())
    except Exception:
        pass


def _make_audio_callback(ws: WebSocket) -> Callable[[bytes], Awaitable[None]]:
    """
    Returns a coroutine that:
    - On b"" → sends audio_done  (end-of-turn)
    - Otherwise → base64-encodes the PCM chunk and sends audio_chunk
    On the very first chunk of a turn, also sends coach_speaking so iOS can
    show a placeholder bubble immediately — before transcript arrives.
    """
    first_chunk = True

    async def on_audio(chunk: bytes):
        nonlocal first_chunk
        if chunk == b"":
            first_chunk = True   # reset for next turn
            await send(ws, "audio_done", {})
        else:
            if first_chunk:
                first_chunk = False
                await send(ws, "coach_speaking", {})  # iOS: show placeholder now
            await send(ws, "audio_chunk", {
                "data":        base64.b64encode(chunk).decode(),
                "sample_rate": live_coach.AUDIO_SAMPLE_RATE,
                "encoding":    "pcm_s16le",
            })
    return on_audio


def _make_partial_transcript_callback(ws: WebSocket):
    """
    Returns a coroutine called for each transcript chunk as Gemini generates it.
    Sends a coach_partial message so iOS can display text while audio plays.
    """
    async def on_partial(chunk: str):
        await send(ws, "coach_partial", {"text": chunk})
    return on_partial


# ── Main WebSocket endpoint ────────────────────────────────

@router.websocket("/ws/run/{run_id}")
async def run_websocket(
    websocket: WebSocket,
    run_id:    int,
    db:        Session = Depends(get_db),
):
    await websocket.accept()

    run = run_service.get_run(db, run_id)
    if not run:
        await send(websocket, "error", {"detail": f"Run {run_id} not found."})
        await websocket.close()
        return

    audio_cb   = _make_audio_callback(websocket)
    partial_cb = _make_partial_transcript_callback(websocket)

    # ── Resume or create run state ─────────────────────────
    is_reconnect = run_id in _active_sessions
    if is_reconnect:
        state = _active_sessions[run_id]
        print(f"🔄 Reconnected run {run_id} — {state.elapsed_minutes:.1f}min elapsed")
        await send(websocket, "reconnected", {"elapsed_minutes": state.elapsed_minutes})

        # Update partial callback so streaming text goes to the new WS
        state._partial_cb  = partial_cb
        state._audio_cb    = audio_cb

        # Plug the new WebSocket's audio callback into the existing Live session
        existing = live_coach._sessions.get(run_id)
        if existing and existing.is_alive:
            existing.update_audio_callback(audio_cb)
            print(f"🔌 Audio callback updated for reconnected run {run_id}")
        else:
            # Live session died; recreate with restored system prompt
            system_prompt = coach_service._build_system_prompt(state)
            try:
                await live_coach.get_or_create(run_id, system_prompt, audio_cb)
                print(f"♻️  Recreated live session for reconnected run {run_id}")
            except Exception as e:
                print(f"⚠️  Could not recreate live session: {e}")

        # Brief pause to let iOS WebSocket fully establish before sending audio
        await asyncio.sleep(0.5)

    else:
        run_already_started = getattr(run, "started_at", None) is not None
        past  = run_service.build_past_runs_summary(db)
        state = RunState(
            run_id=run_id,
            run_type=getattr(run, "run_type", "easy") or "easy",
            goal_distance_km=run.goal_distance_km,
            goal_duration_min=run.goal_duration_min,
            goal_description=run.goal_description,
            past_runs_summary=past,
        )

        if run_already_started:
            saved_messages = run_service.get_messages_for_run(db, run_id)
            if saved_messages:
                state.conversation_history = [
                    {"role": m.role, "content": m.content} for m in saved_messages
                ]
                import datetime
                first_msg_time = saved_messages[0].created_at
                elapsed_sec = (datetime.datetime.utcnow() - first_msg_time).total_seconds()
                state._start_time = time.time() - elapsed_sec
                print(f"♻️  Restored run {run_id} — {len(saved_messages)} messages")
                await send(websocket, "reconnected", {"elapsed_minutes": state.elapsed_minutes})
            else:
                run_already_started = False

        raw_cfg = getattr(run, "interval_config", None)
        if raw_cfg and state.run_type == "interval":
            state.interval_config = raw_cfg if isinstance(raw_cfg, dict) else {}

        checkpoint_m = getattr(run, "checkpoint_interval_m", None)
        if checkpoint_m and isinstance(checkpoint_m, int) and checkpoint_m > 0:
            state.checkpoint_interval_m = checkpoint_m

        _active_sessions[run_id] = state

        # ── Start the Gemini Live session ──────────────────
        system_prompt = coach_service._build_system_prompt(state)
        session_obj = None
        try:
            session_obj = await live_coach.get_or_create(run_id, system_prompt, audio_cb)
        except Exception as e:
            print(f"⚠️  Live session failed, will use text fallback: {e}")

        # Store on state so _message_loop can recreate the session if Gemini kills it
        state._partial_cb    = partial_cb
        state._audio_cb      = audio_cb
        state._system_prompt = system_prompt

        if not run_already_started:
            run_service.start_run(db, run_id)
            opening = await coach_service.get_opening_message(
                state, live_session=session_obj, partial_cb=partial_cb
            )
            # Send full text for iOS chat history (audio + partial text already streamed)
            await send(websocket, "coach", {
                "message":         opening,
                "urgency":         "normal",
                "suggested_pace":  None,
                "coaching_action": None,
                "is_live":         session_obj is not None,
            })
            state.conversation_history.append({"role": "assistant", "content": opening})

    ping_task = asyncio.ensure_future(_ping_loop(websocket))
    try:
        await _message_loop(websocket, db, state, run_id)
    except WebSocketDisconnect:
        print(f"⚡ Disconnected run {run_id} — state & live session preserved")
    except Exception as e:
        print(f"❌ WS error run {run_id}: {e}")
        import traceback; traceback.print_exc()
    finally:
        ping_task.cancel()


# ── Ping loop ──────────────────────────────────────────────

async def _ping_loop(ws: WebSocket):
    try:
        while True:
            await asyncio.sleep(15)
            await ws.send_text('{"type":"ping","payload":{}}')
    except Exception:
        pass


# ── Main message loop ──────────────────────────────────────

async def _message_loop(ws: WebSocket, db: Session, state: RunState, run_id: int):
    while True:
        raw      = await ws.receive_text()
        data     = json.loads(raw)
        msg_type = data.get("type")
        payload  = data.get("payload", {})

        # Retrieve live session — auto-recreate if Gemini killed it (1008/timeout)
        live_session = live_coach._sessions.get(run_id)
        if live_session and not live_session.is_alive:
            # Session died (1008 policy violation / context limit) — recreate silently
            print(f"🔄 Live session dead for run {run_id} — recreating...")
            try:
                live_session = await live_coach.get_or_create(
                    run_id,
                    getattr(state, "_system_prompt", ""),
                    getattr(state, "_audio_cb", lambda _: None),
                )
            except Exception as _e:
                print(f"⚠️  Session recreation failed: {_e} — using text fallback")
                live_session = None

        # ── Keepalive ──────────────────────────────────────
        if msg_type in ("ping", "pong"):
            if msg_type == "ping":
                await ws.send_text('{"type":"pong","payload":{}}')
            continue

        # ── Vitals update ──────────────────────────────────
        elif msg_type == "vitals":
            vitals = VitalSignInput(**payload)
            state.update_vitals(vitals)
            run_service.save_vital(db, run_id, vitals)

            event = (
                check_interval_events(state)
                if state.run_type == "interval"
                else check_steady_events(state)
            )

            if event:
                partial_cb = getattr(state, "_partial_cb", None)
                message = await coach_service.get_event_coaching(
                    event, state, live_session=live_session, partial_cb=partial_cb
                )
                if message:
                    await send(ws, "coach", {
                        "message":         message,
                        "urgency":         _event_urgency(event.event_type),
                        "suggested_pace":  None,
                        "coaching_action": _event_action(event.event_type),
                        "is_live":         live_session is not None,
                    })
                    state.conversation_history.append({"role": "assistant", "content": message})
                    run_service.save_message(db, run_id, "assistant", message, vitals_snapshot=payload)
                    state.last_proactive_at = time.time()
            else:
                now = time.time()
                hr  = state.current_hr or 0
                if hr > 175 and (now - state.last_hr_warning_at) > 45:
                    state.last_hr_warning_at = now
                    message = await coach_service.get_proactive_coaching(
                        state, live_session=live_session
                    )
                    if message:
                        await send(ws, "coach", {
                            "message":         message,
                            "urgency":         "critical" if hr > 185 else "warning",
                            "suggested_pace":  None,
                            "coaching_action": "slow_down",
                            "is_live":         live_session is not None,
                        })
                        state.conversation_history.append({"role": "assistant", "content": message})
                        run_service.save_message(db, run_id, "assistant", message, vitals_snapshot=payload)

        # ── Chat message (text from iOS STT) ──────────────
        elif msg_type == "chat":
            user_text = payload.get("content", "").strip()
            if not user_text:
                continue

            print(f"🎤 User [{run_id}]: {user_text}")
            state.conversation_history.append({"role": "user", "content": user_text})
            run_service.save_message(db, run_id, "user", user_text,
                                     vitals_snapshot=state.latest_vitals.model_dump())

            partial_cb = getattr(state, "_partial_cb", None)
            response = await coach_service.get_conversational_response(
                user_text, state, live_session=live_session, partial_cb=partial_cb
            )
            # Text message for display — audio already arrived via audio_chunk
            coach_payload = response.model_dump()
            coach_payload["is_live"] = live_session is not None
            await send(ws, "coach", coach_payload)
            state.conversation_history.append({"role": "assistant", "content": response.message})
            run_service.save_message(db, run_id, "assistant", response.message,
                                     vitals_snapshot=state.latest_vitals.model_dump())

        # ── Language switch ────────────────────────────────
        elif msg_type == "set_language":
            lang = payload.get("language", "en")
            state.language_instruction = (
                "Always respond in English."
                if lang == "en"
                else "تحدث دائماً باللغة العربية الفصحى البسيطة. كن ودوداً وطبيعياً مثل صديق يركض بجانبك."
            )

        # ── End run ────────────────────────────────────────
        elif msg_type == "end_run":
            await _finish_run(ws, db, state, run_id)
            await live_coach.close(run_id)          # close Live session cleanly
            _active_sessions.pop(run_id, None)
            break


# ── Run finish ─────────────────────────────────────────────

async def _finish_run(ws, db, state, run_id):
    # Signal iOS immediately — navigate to analysis page NOW, before audio plays.
    # The analysis text will follow via the "analysis" message once ready.
    await send(ws, "run_ending", {})
    print(f'📊 Generating post-run analysis for run {run_id}...')
    all_vitals = run_service.get_vitals_for_run(db, run_id)
    sampled = [
        {
            "recorded_at": str(v.recorded_at),
            "heart_rate":  v.heart_rate,
            "pace_min_km": v.pace_min_km,
            "distance_km": v.distance_km,
        }
        for i, v in enumerate(all_vitals) if i % 2 == 0
    ]

    # Live session may have been recreated mid-run — get current one from registry
    live_session = live_coach._sessions.get(run_id)
    if live_session and not live_session.is_alive:
        try:
            live_session = await live_coach.get_or_create(
                run_id,
                getattr(state, "_system_prompt", ""),
                getattr(state, "_audio_cb", lambda _: None),
            )
        except Exception:
            live_session = None

    # Stream analysis text to iOS as Gemini transcribes it — text populates while audio plays
    async def on_analysis_partial(chunk: str):
        await send(ws, "analysis_partial", {"text": chunk})

    try:
        analysis = await coach_service.generate_post_run_analysis(
            state, sampled,
            live_session=live_session,
            partial_cb=on_analysis_partial if live_session else None,
        )
        print(f'✅ Post-run analysis ready ({len(analysis)} chars)')
    except Exception as e:
        print(f'⚠️  Post-run analysis failed ({e}) — using fallback summary')
        dist    = state.current_distance
        elapsed = state.elapsed_minutes
        analysis = (
            f"Great effort — {dist:.1f} kilometres in {elapsed:.0f} minutes. "
            f"Keep building on this consistency."
        )

    completed = run_service.end_run(db, run_id, analysis)
    print(f'✅ Run {run_id} saved to DB')

    # Send final complete text — iOS replaces streamed partial with this
    await send(ws, "analysis", {"text": analysis})
    await send(ws, "run_ended", {
        "run_id":             run_id,
        "total_distance_km":  completed.total_distance_km  if completed else state.current_distance,
        "total_duration_min": completed.total_duration_min if completed else state.elapsed_minutes,
        "avg_heart_rate":     completed.avg_heart_rate     if completed else None,
        "goal_achieved":      completed.goal_achieved      if completed else False,
    })


# ── Urgency / action helpers ───────────────────────────────

def _event_urgency(event_type: str) -> str:
    if event_type in ("hr_critical", "hr_critical_during_work"): return "critical"
    if event_type in ("hr_high", "pace_off"):                    return "warning"
    if event_type in ("interval_start", "rest_countdown"):       return "warning"
    return "normal"


def _event_action(event_type: str) -> str | None:
    if event_type in ("hr_critical", "hr_critical_during_work"): return "slow_down"
    if event_type == "hr_high":                                  return "slow_down"
    if event_type == "interval_start":                           return "speed_up"
    if event_type in ("interval_done", "rest_countdown"):        return "slow_down"
    return None