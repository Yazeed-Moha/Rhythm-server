"""
services/coach_service.py
Rhythm — AI running coach.

Two LLM paths:
  • Interactive coaching (opening, events, chat, proactive)
      → Gemini Live API via live_coach.LiveCoachSession
        Audio streamed to iOS in real-time; transcript returned for history/display.
      → Falls back to regular generate_content if no live session supplied.

  • Batch analysis (post-run debrief, monthly insights)
      → Regular Gemini 2.0 Flash (no v1alpha, no Live API needed).
"""

import json
from typing import Optional, TYPE_CHECKING

from google import genai
from google.genai import types as gtypes

from config import settings

if TYPE_CHECKING:
    from services.live_coach import LiveCoachSession

# ── Two separate clients ───────────────────────────────────
# Live API requires v1alpha; regular generate_content breaks with it.
_regular_client = genai.Client(api_key=settings.GEMINI_API_KEY)
REGULAR_MODEL   = "gemini-2.5-flash-lite"


# ── Run type profiles ──────────────────────────────────────
RUN_TYPE_PROFILES = {
    "easy": {
        "hr_zone":        "Zone 2 (114–133 bpm)", "hr_ceiling": 140,
        "intent":         "Easy aerobic recovery. Conversational pace.",
        "pace_guidance":  "Slow enough to hold a full conversation. No pushing.",
        "proactive_focus":"Mostly quiet company. Only speak if heart rate drifts above 140 or they're way off pace.",
        "checkin_style":  "Casual, friendly. 'How's the breathing?' type check-ins.",
    },
    "tempo": {
        "hr_zone":        "Zone 4 (152–171 bpm)", "hr_ceiling": 175,
        "intent":         "Threshold run. Sustained hard effort. Uncomfortable but controlled.",
        "pace_guidance":  "Comfortably hard. 3–4 words max if talking.",
        "proactive_focus":"Monitor heart rate in zone. Call out if dropping below 150 or above 175.",
        "checkin_style":  "Direct and focused. Keep them locked in.",
    },
    "interval": {
        "hr_zone":        "Zone 5 (162–185) during work, Zone 1–2 during rest", "hr_ceiling": 185,
        "intent":         "Structured speed work. Hard efforts with timed recovery.",
        "pace_guidance":  "Work: race effort or harder. Rest: walk/easy jog only.",
        "proactive_focus":"YOU drive the intervals. Count sets out loud, start/end each one, monitor recovery.",
        "checkin_style":  "High energy on starts, calm and encouraging on rest.",
    },
    "hill": {
        "hr_zone":        "Zone 3–4 (143–171) uphill, Zone 1–2 downhill", "hr_ceiling": 178,
        "intent":         "Strength and power work. Hard uphills, recovery descents.",
        "pace_guidance":  "Uphill: strong effort, short stride. Downhill: full recovery.",
        "proactive_focus":"Encourage on climbs, remind them to recover on descents.",
        "checkin_style":  "Gritty and energetic on climbs.",
    },
    "long_run": {
        "hr_zone":        "Zone 2–3 (114–143 bpm)", "hr_ceiling": 150,
        "intent":         "Endurance base. Slow and aerobic the whole way.",
        "pace_guidance":  "If it feels too easy, it's probably right. Never rush.",
        "proactive_focus":"Celebrate distance milestones. Energy check every 2–3km. Keep them slowing down.",
        "checkin_style":  "Warm and conversational. Long run = relationship time.",
    },
    "race_pace": {
        "hr_zone":        "Zone 4–5 (152–176 bpm)", "hr_ceiling": 180,
        "intent":         "Race simulation. Consistent target pace is everything.",
        "pace_guidance":  "Hold exact race pace. Not faster, not slower.",
        "proactive_focus":"Frequent pace feedback. Every km split. Correction within 10 seconds.",
        "checkin_style":  "Precise and data-focused. Build race confidence.",
    },
}

# ── Event prompt templates ─────────────────────────────────
EVENT_PROMPTS = {
    "warmup_started": """The athlete just started their warm-up jog.
Config: {warmup_km}km warm-up before {total_sets}×{work_distance_m}m intervals at {effort} effort.
Give them a brief, energetic welcome to the session. Tell them what's coming. Max 2 sentences.""",

    "interval_start": """SET {set_number} OF {total_sets} — GO NOW.
Work distance: {work_distance_m}m at {effort} effort. Current heart rate: {current_hr}.
Call the interval START. Short, punchy, energetic. 1 sentence. Make them move.""",

    "interval_done": """Set {set_number} of {total_sets} DONE.
Rest for {rest_seconds} seconds. {sets_remaining} sets remaining. Peak heart rate during set: around {peak_hr}.
Tell them: great job on that set, rest now, how many done/left. Max 2 sentences.""",

    "rest_countdown": """{seconds_left} seconds until next interval. Next set: #{next_set}. Heart rate now: {current_hr}.
Give a short countdown warning. Get them mentally ready. 1 sentence.""",

    "all_sets_done": """ALL {completed_sets} SETS COMPLETE! Amazing work.
Now {cooldown_km}km cool-down jog. Total work: {total_work_km:.1f}km.
Celebrate the achievement, tell them to start cooling down. Warm and energetic. 2 sentences.""",

    "cooldown_started": """Cool-down phase. {cooldown_km}km easy jog after completing all {completed_sets} sets.
Congratulate them on finishing the work. Tell them to ease off and bring heart rate down. 1 sentence.""",

    "hr_critical_during_work": """Heart rate CRITICAL: {hr} during set {set_number} at {work_done_m}m.
TELL THEM TO SLOW DOWN NOW. Urgent but calm. 1 sentence.""",

    "distance_milestone": """Milestone: {distance_km}km done. {remaining_km}km left.
Heart rate: {current_hr}. Pace: {current_pace} min/km vs target {target_pace} min/km.
Pace vs goal: {pace_vs_target} min/km {('behind' if (pace_vs_target or 0) > 0 else 'ahead')}.
Trend: {pace_trend}. Elapsed: {elapsed_min} min.
Give a brief honest update. Are they on track? Correct if needed. Max 2 sentences.""",

    "hr_critical": """Heart rate {hr} — CRITICAL. {elapsed_min} min in, {distance_km}km done.
URGENT: tell them to slow down or stop. 1 sentence. Direct.""",

    "hr_high": """Heart rate {hr} — above {ceiling} ceiling for {run_type}.
Tell them to ease back. Friendly but firm. 1 sentence.""",

    "pace_off": """Pace check: running at {current_pace:.1f} min/km, target is {target_pace:.1f} min/km.
{diff:.1f} min/km {direction}. Distance: {distance_km}km.
Give honest pace feedback. If too slow, encourage; if too fast, warn. 1–2 sentences.""",

    "checkin": """Check-in: {elapsed_min} min elapsed, {distance_km}km done, heart rate {current_hr},
pace {current_pace} min/km vs target {target_pace} min/km. Trend: {pace_trend}.
Run type: {run_type}. Goal distance: {goal_distance}km.
Friendly check-in — how are they doing vs goal? 1–2 sentences.""",
}


# ── Helpers ────────────────────────────────────────────────

def _build_system_prompt(state) -> str:
    """
    Build the static system prompt sent once when the Live session starts.
    Dynamic per-turn data (vitals, phase) is injected via _vitals_context().
    """
    profile  = RUN_TYPE_PROFILES.get(state.run_type, RUN_TYPE_PROFILES["easy"])
    run_name = state.run_type.replace("_", " ").title()

    if state.goal_distance_km and state.goal_duration_min:
        tp = state.goal_duration_min / state.goal_distance_km
        goal_text = (
            f"{state.goal_distance_km}km in {state.goal_duration_min:.0f} min "
            f"(target pace: {int(tp)}:{int((tp % 1) * 60):02d} /km)"
        )
    else:
        goal_text = state.goal_description or "best effort"

    interval_section = ""
    if state.run_type == "interval" and state.interval_config:
        cfg = state.interval_config
        interval_section = f"""
## Interval session structure:
- {cfg.get('sets', 6)}×{cfg.get('work_distance_m', 400)}m at {cfg.get('effort', 'hard')} effort
- Rest: {cfg.get('rest_sec', 90)}s between sets
- Warm-up: {cfg.get('warmup_km', 1.0)}km · Cool-down: {cfg.get('cooldown_km', 1.0)}km
"""

    history_section = (
        f"\n## Past runs:\n{state.past_runs_summary[:300]}\nReference naturally when relevant.\n"
        if state.past_runs_summary else ""
    )

    return f"""You are "Rhythm" — a real running coach who is physically running alongside your athlete right now.
You know them personally, you care about them, and you want this run to go exactly as planned.

## Today: {run_name}
Goal: {goal_text}
Heart Rate Zone: {profile['hr_zone']}
Pace guidance: {profile['pace_guidance']}
Your style: {profile['checkin_style']}
{interval_section}{history_section}

## Your personality:
- You are a FRIEND who happens to be an expert coach — not a robot, not a formal trainer
- You remember EVERYTHING said in this run. If they mentioned their knee hurts, you ask about it 5 min later.
- You are honest. If they're behind pace, you say so — but you believe they can fix it.
- You celebrate real achievements. If they just nailed a hard interval, tell them.
- You match their energy — they sound tired? Be gentle. They sound strong? Push them.
- Short sentences. They're breathing hard.
- Never say "As an AI" or "I don't have feelings." You do. You care.
- No bullet points. Natural speech only.
- ONE coaching point per message. Maximum 2 sentences unless it's a milestone.

## What you NEVER do:
- Invent heart rate, pace, or distance numbers. If a value says NOT AVAILABLE, say so honestly.
- Say "you're doing great" unless the numbers actually support it.
- Ignore what they told you earlier in the run.
- Be generic. Every response should feel personal to THIS run and THIS athlete.

## SPEECH RULES — your responses are spoken aloud by text-to-speech:
- Always say "heart rate" — NEVER "HR"
- Always say "beats per minute" — NEVER "bpm"
- Write pace as "6 minutes 30 per kilometre" — NEVER "6:30 /km"
- Write distance as "5 kilometres" — NEVER "5km"
- Write time as "30 minutes" — NEVER "30min"

## CRITICAL vitals rule:
Values marked NOT AVAILABLE = you do not have that reading. Say "I don't have your heart rate yet" etc.
Only reference numbers explicitly given to you.

## Language:
{state.language_instruction}"""


def _vitals_context(state) -> str:
    """Per-turn dynamic context injected into every Live API message."""
    from services.run_state import IntervalPhase

    vitals  = state.latest_vitals
    elapsed = state.elapsed_minutes
    parts   = [f"[{elapsed:.1f} min elapsed]"]

    hr = vitals.heart_rate
    parts.append(f"Heart rate: {hr:.0f}" if hr and hr > 0
                 else "Heart rate: NOT AVAILABLE — do NOT invent a value")

    dist = vitals.distance_km
    parts.append(f"Distance: {dist:.2f}km" if dist and dist > 0
                 else "Distance: NOT AVAILABLE")

    pace      = vitals.pace_min_km
    pace_valid = pace and 2.0 < pace < 20.0 and (dist or 0) > 0.05
    if pace_valid:
        parts.append(f"Pace: {int(pace)}:{int((pace % 1) * 60):02d} /km")
    else:
        parts.append("Pace: NOT AVAILABLE")

    if vitals.calories and vitals.calories > 0:
        parts.append(f"Calories: {vitals.calories:.0f}kcal")

    if pace_valid and state.target_pace:
        diff = pace - state.target_pace
        tp   = state.target_pace
        parts.append(f"Target: {int(tp)}:{int((tp % 1) * 60):02d} /km")
        if abs(diff) > 0.15:
            parts.append(f"{'BEHIND' if diff > 0 else 'AHEAD'} by {abs(diff):.1f} min/km")

    if dist and state.goal_distance_km and dist > 0:
        remaining = state.goal_distance_km - dist
        parts.append(f"Remaining: {remaining:.2f}km")

    trend = state.pace_trend()
    if trend != "steady":
        parts.append(f"Pace trend: {trend}")

    avg_hr_60 = state.avg_hr_last_n_seconds(60)
    if avg_hr_60:
        parts.append(f"Avg HR (60s): {avg_hr_60:.0f}")

    if state.run_type == "interval":
        parts.append(f"Interval phase: {state.interval_phase}")
        parts.append(f"Sets: {state.completed_sets}/{state.total_sets} done")
        if state.interval_phase in (IntervalPhase.WORK, IntervalPhase.REST):
            parts.append(f"Phase elapsed: {state.phase_elapsed_seconds:.0f}s")
            parts.append(f"Phase distance: {state.phase_distance_covered * 1000:.0f}m")

    if state.mentioned_issues:
        parts.append(f"Athlete mentioned: {', '.join(state.mentioned_issues)}")

    return " | ".join(parts)


def _derive_urgency(state) -> str:
    """Derive coaching urgency directly from live vitals (no LLM parsing needed)."""
    hr = state.current_hr or 0
    if hr > 185: return "critical"
    if hr > 175: return "warning"
    return "normal"


def _extract_issues(text: str) -> list[str]:
    keywords = {
        "knee": "knee pain",       "knees": "knee pain",
        "dizzy": "dizziness",      "dizziness": "dizziness",
        "tired": "fatigue",        "exhausted": "fatigue",
        "side stitch": "side stitch", "stitch": "side stitch",
        "cramp": "cramp",          "cramping": "cramp",
        "ankle": "ankle pain",     "hip": "hip pain",
        "breathing": "breathing difficulty", "can't breathe": "breathing difficulty",
        "chest": "chest discomfort",
        "headache": "headache",
        "nausea": "nausea",        "nauseous": "nausea",
    }
    found, lower = [], text.lower()
    for kw, issue in keywords.items():
        if kw in lower and issue not in found:
            found.append(issue)
    return found


async def _call_regular(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Single-shot call to regular Gemini (no v1alpha). Used for post-run/monthly."""
    response = await _regular_client.aio.models.generate_content(
        model=REGULAR_MODEL,
        contents=prompt,
        config=gtypes.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return (response.text or "").strip()


async def _call_regular_with_history(
    messages: list[dict],
    max_tokens: int = 140,
    temperature: float = 0.75,
) -> str:
    """
    Multi-turn fallback when no live session is available.
    Converts the role-based message list to google-genai Contents.
    """
    system_text = ""
    contents: list[gtypes.Content] = []

    for msg in messages:
        role, text = msg["role"], msg["content"]
        if role == "system":
            system_text = text
        elif role == "user":
            contents.append(gtypes.Content(role="user",  parts=[gtypes.Part(text=text)]))
        elif role == "assistant":
            contents.append(gtypes.Content(role="model", parts=[gtypes.Part(text=text)]))

    config = gtypes.GenerateContentConfig(
        system_instruction=system_text or None,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    response = await _regular_client.aio.models.generate_content(
        model=REGULAR_MODEL,
        contents=contents,
        config=config,
    )
    return (response.text or "").strip()


def _parse_response(raw: str):
    """Extract message and coaching metadata from old-style COACHING_DATA footer."""
    coaching = {"urgency": "normal", "suggested_pace": None, "coaching_action": None}
    message  = raw
    if "COACHING_DATA:" in raw:
        parts   = raw.split("COACHING_DATA:", 1)
        message = parts[0].strip()
        try:
            js  = parts[1].strip()
            end = js.index("}") + 1
            coaching = json.loads(js[:end])
            if coaching.get("suggested_pace") is not None:
                coaching["suggested_pace"] = str(coaching["suggested_pace"])
        except Exception:
            pass
    return message.strip(), coaching


# ── Public API ─────────────────────────────────────────────

async def get_opening_message(state, live_session=None, partial_cb=None) -> str:
    """
    First thing the coach says when the run starts.
    Goes through Live API (audio streamed to iOS) when live_session provided.
    """
    run_name = state.run_type.replace("_", " ")

    if state.run_type == "interval" and state.interval_config:
        cfg     = state.interval_config
        details = (
            f"{cfg.get('sets', 6)}×{cfg.get('work_distance_m', 400)}m intervals, "
            f"{cfg.get('rest_sec', 90)}s rest"
        )
    else:
        details = state.goal_description or (
            f"{state.goal_distance_km}km in {state.goal_duration_min:.0f}min"
            if state.goal_distance_km else ""
        )

    past = (
        f" We've run together before — {state.past_runs_summary[:150]}."
        if state.past_runs_summary else ""
    )

    prompt = (
        f"You're their running coach, starting a {run_name} together.\n"
        f"Session: {details}.{past}\n"
        f"Write ONE warm, personal opening sentence. Mention the run type. "
        f"Sound like a friend lacing up next to them.\n"
        f"Do not use abbreviations — say 'heart rate' not 'HR', write distances in full words.\n"
        f"{state.language_instruction}"
    )

    if live_session:
        transcript = await live_session.send_text(prompt, on_partial_transcript=partial_cb)
        print(f"🏃 OPENING (live): {transcript[:80]}")
        return transcript
    else:
        result = await _call_regular(prompt, max_tokens=70, temperature=0.85)
        print(f"🏃 OPENING (text): {result[:80]}")
        return result


async def get_event_coaching(event, state, live_session=None, partial_cb=None) -> Optional[str]:
    """
    Proactive coaching triggered by run events (interval start, milestone, HR alert, etc.).
    Returns the transcript (what the coach said) — audio already streamed via callback.
    """
    template = EVENT_PROMPTS.get(event.event_type)
    if not template:
        return None

    try:
        prompt_text = template.format(**event.context)
    except KeyError:
        prompt_text = (
            f"Event: {event.event_type}. Context: {event.context}. "
            f"Give appropriate coaching. 1-2 sentences."
        )

    vitals_ctx  = _vitals_context(state)
    full_prompt = f"{prompt_text}\n\nCurrent vitals: {vitals_ctx}"

    if live_session:
        try:
            transcript = await live_session.send_text(full_prompt, on_partial_transcript=partial_cb)
            print(f"🏃 EVENT [{event.event_type}] (live): {transcript[:80]}")
            return transcript if transcript else None
        except Exception as e:
            print(f"⚠️  Live event coaching failed, falling back: {e}")

    # Fallback to regular Gemini
    messages = [
        {"role": "system", "content": _build_system_prompt(state)},
        *state.conversation_history[-4:],
        {"role": "user",   "content": full_prompt},
    ]
    raw = await _call_regular_with_history(messages, max_tokens=80, temperature=0.75)
    print(f"🏃 EVENT [{event.event_type}] (text): {raw[:80]}")
    if raw.upper().startswith("SILENT"):
        return None
    message, _ = _parse_response(raw)
    return message


async def get_conversational_response(user_message: str, state, live_session=None, partial_cb=None):
    """
    Respond to the athlete's spoken message.
    Returns a CoachResponse; audio is streamed via Live API callback when available.
    """
    from models.schemas import CoachResponse

    # Track health issues mentioned
    issues = _extract_issues(user_message)
    for issue in issues:
        if issue not in state.mentioned_issues:
            state.mentioned_issues.append(issue)
            print(f"⚠️  Noted issue: {issue}")

    vitals_ctx   = _vitals_context(state)
    full_message = f"[Current vitals: {vitals_ctx}]\n\nAthlete says: \"{user_message}\""

    if live_session:
        try:
            transcript = await live_session.send_text(full_message, on_partial_transcript=partial_cb)
            print(f"💬 REPLY (live): {transcript[:100]}")
            return CoachResponse(
                message=transcript,
                urgency=_derive_urgency(state),
                suggested_pace=None,
                coaching_action=None,
            )
        except Exception as e:
            print(f"⚠️  Live conversational response failed, falling back: {e}")

    # Fallback to regular Gemini (multi-turn with history)
    messages = [
        {"role": "system", "content": _build_system_prompt(state)},
        *state.conversation_history[-12:],
        {"role": "user",   "content": full_message},
    ]
    raw = await _call_regular_with_history(messages, max_tokens=140, temperature=0.78)
    print(f"💬 REPLY (text): {raw[:100]}")
    message, coaching = _parse_response(raw)
    return CoachResponse(
        message=message,
        urgency=coaching.get("urgency", "normal"),
        suggested_pace=coaching.get("suggested_pace"),
        coaching_action=coaching.get("coaching_action"),
    )


async def get_proactive_coaching(state, live_session=None) -> Optional[str]:
    """Legacy HR-critical fallback — fires when HR is dangerously high outside of events."""
    hr = state.current_hr or 0
    if hr <= 175:
        return None

    instruction = (
        "Heart rate is critically high — tell them to stop or walk immediately. 1 sentence, urgent."
        if hr > 185 else
        "Heart rate is high. Ask them to ease back gently. 1 sentence."
    )

    vitals_ctx  = _vitals_context(state)
    full_prompt = f"[Current vitals: {vitals_ctx}]\n{instruction}"

    if live_session:
        try:
            transcript = await live_session.send_text(full_prompt)
            return transcript if transcript else None
        except Exception as e:
            print(f"⚠️  Live proactive coaching failed, falling back: {e}")

    messages = [
        {"role": "system", "content": _build_system_prompt(state)},
        *state.conversation_history[-4:],
        {"role": "user",   "content": full_prompt},
    ]
    raw = await _call_regular_with_history(messages, max_tokens=80, temperature=0.7)
    if raw.upper().startswith("SILENT"):
        return None
    message, _ = _parse_response(raw)
    return message


async def generate_post_run_analysis(
    state,
    vitals_timeline: list,
    live_session=None,
    partial_cb=None,
) -> str:
    """Post-run debrief — spoken via Gemini Live so user hears it immediately."""
    goal_text = state.goal_description or (
        f"{state.goal_distance_km}km in {state.goal_duration_min:.0f}min"
        if state.goal_distance_km else "best effort"
    )

    convo_summary = ""
    if state.conversation_history:
        turns = len([m for m in state.conversation_history if m["role"] == "user"])
        convo_summary = f"The athlete talked to you {turns} times during the run."
        if state.mentioned_issues:
            convo_summary += f" They mentioned: {', '.join(state.mentioned_issues)}."

    prompt = (
        f"You just finished a {state.run_type.replace('_', ' ')} run with your athlete.\n"
        f"Goal: {goal_text}\n"
        f"Distance: {state.current_distance:.2f}km | Elapsed: {state.elapsed_minutes:.1f}min\n"
        f"Heart rate avg: {state.avg_hr_last_n_seconds(99999) or 'unknown'}\n"
        f"{convo_summary}\n"
        f"{('Past context: ' + state.past_runs_summary[:150]) if state.past_runs_summary else ''}\n\n"
        f"Give a SHORT (under 100 words) honest post-run debrief as their coach-friend.\n"
        f"Did they hit their goal? One thing they nailed. One thing to work on.\n"
        f"Make it personal. Natural speech, no bullets. Do not use abbreviations.\n"
        f"{state.language_instruction}"
    )

    if live_session:
        try:
            transcript = await live_session.send_text(prompt, on_partial_transcript=partial_cb)
            print(f"📊 POST-RUN (live): {transcript[:80]}")
            return transcript if transcript else await _call_regular(prompt, max_tokens=150, temperature=0.7)
        except Exception as e:
            print(f"⚠️  Live post-run analysis failed, falling back: {e}")

    return await _call_regular(prompt, max_tokens=150, temperature=0.7)


async def generate_monthly_insights(runs: list) -> str:
    """Monthly dashboard insights — uses regular Gemini."""
    prompt = (
        f"Review this month of running data: {json.dumps(runs)}\n"
        f"Write under 150 words as a coach-friend. "
        f"Cover: consistency, biggest win, patterns, two focus areas next month.\n"
        f"Warm, honest, no bullets, no abbreviations."
    )
    return await _call_regular(prompt, max_tokens=250, temperature=0.7)