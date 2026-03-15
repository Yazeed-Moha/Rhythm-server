"""
services/interval_manager.py
Drives interval session: warmup → work → rest → ... → cooldown.
Returns coaching events that trigger AI responses.
"""

import time
from dataclasses import dataclass
from typing import Optional
from .run_state import RunState, IntervalPhase, IntervalSet


@dataclass
class CoachingEvent:
    """A trigger that tells the coach to say something specific."""
    event_type: str   # "interval_start" | "interval_end" | "rest_end" | "warmup_done"
                      # "cooldown_start" | "all_done" | "hr_too_high" | "checkin"
    context:    dict  # extra info for the AI


def check_interval_events(state: RunState) -> Optional[CoachingEvent]:
    """
    Called on every vitals update for interval runs.
    Detects phase transitions and returns coaching events.
    """
    if state.run_type != "interval" or not state.interval_config:
        return None

    dist = state.current_distance
    hr   = state.current_hr or 0

    # ── WARMUP phase ──────────────────────────────────────
    if state.interval_phase == IntervalPhase.WARMUP:
        if not state.warmup_announced and state.has_started_moving:
            state.warmup_announced = True
            return CoachingEvent("warmup_started", {
                "warmup_km": state.warmup_km,
                "total_sets": state.total_sets,
                "work_distance_m": state.work_distance_m,
                "effort": state.effort_level,
            })

        # Warmup done when distance reached
        if dist >= state.warmup_km and state.has_started_moving:
            state.start_phase(IntervalPhase.WORK)
            state.current_set = 1
            return CoachingEvent("interval_start", {
                "set_number": state.current_set,
                "total_sets": state.total_sets,
                "work_distance_m": state.work_distance_m,
                "effort": state.effort_level,
                "current_hr": hr,
            })

    # ── WORK phase ────────────────────────────────────────
    elif state.interval_phase == IntervalPhase.WORK:
        work_done_m = state.phase_distance_covered * 1000

        # HR check during work — if too high, flag it
        if hr > 188 and (time.time() - state.last_hr_warning_at) > 45:
            state.last_hr_warning_at = time.time()
            return CoachingEvent("hr_critical_during_work", {
                "hr": hr, "set_number": state.current_set,
                "work_done_m": int(work_done_m),
            })

        # Work interval complete
        if work_done_m >= state.work_distance_m:
            state.completed_sets += 1
            state.sets_history.append(IntervalSet(
                set_number=state.current_set,
                started_at=state.phase_started_at,
                ended_at=time.time(),
                peak_hr=max((hr for _, hr in state.hr_history[-20:]), default=None),
                completed=True,
            ))

            if state.completed_sets >= state.total_sets:
                # All sets done — go to cooldown
                total_work_km = state.warmup_km + (state.work_distance_m * state.total_sets / 1000)
                state.start_phase(IntervalPhase.COOLDOWN)
                return CoachingEvent("all_sets_done", {
                    "completed_sets": state.completed_sets,
                    "cooldown_km": state.cooldown_km,
                    "total_work_km": total_work_km,
                })
            else:
                # Go to rest
                state.start_phase(IntervalPhase.REST)
                return CoachingEvent("interval_done", {
                    "set_number": state.current_set,
                    "total_sets": state.total_sets,
                    "rest_seconds": state.rest_seconds,
                    "sets_remaining": state.total_sets - state.completed_sets,
                    "peak_hr": hr,
                })

    # ── REST phase ────────────────────────────────────────
    elif state.interval_phase == IntervalPhase.REST:
        rest_elapsed = state.phase_elapsed_seconds
        hr_recovered = hr < 130 if hr > 0 else False
        rest_done    = rest_elapsed >= state.rest_seconds

        # 10-second countdown warning
        if rest_elapsed >= (state.rest_seconds - 10) and rest_elapsed < state.rest_seconds:
            if (time.time() - state.last_checkin_at) > 12:
                state.last_checkin_at = time.time()
                return CoachingEvent("rest_countdown", {
                    "seconds_left": int(state.rest_seconds - rest_elapsed),
                    "next_set": state.current_set + 1,
                    "current_hr": hr,
                })

        if rest_done or (hr_recovered and rest_elapsed >= state.rest_seconds * 0.8):
            state.current_set += 1
            state.start_phase(IntervalPhase.WORK)
            return CoachingEvent("interval_start", {
                "set_number": state.current_set,
                "total_sets": state.total_sets,
                "work_distance_m": state.work_distance_m,
                "effort": state.effort_level,
                "current_hr": hr,
                "recovered_hr": hr,
            })

    # ── COOLDOWN phase ────────────────────────────────────
    elif state.interval_phase == IntervalPhase.COOLDOWN:
        if not state.cooldown_announced:
            state.cooldown_announced = True
            return CoachingEvent("cooldown_started", {
                "cooldown_km": state.cooldown_km,
                "completed_sets": state.completed_sets,
            })

    return None


def check_steady_events(state: RunState) -> Optional[CoachingEvent]:
    """
    Checks for proactive coaching events in non-interval runs.
    Distance milestones, pace drift, HR zones, check-ins.
    """
    if not state.has_started_moving:
        return None

    dist    = state.current_distance
    hr      = state.current_hr or 0
    pace    = state.current_pace
    elapsed = state.elapsed_minutes
    now     = time.time()

    # ── Distance milestones (user-configured interval) ───
    milestone_interval_km = state.checkpoint_interval_m / 1000.0
    next_milestone = state.last_milestone_km + milestone_interval_km
    if dist >= next_milestone and (now - state.last_proactive_at) > 20:
        state.last_milestone_km = next_milestone
        state.last_proactive_at = now
        remaining = (state.goal_distance_km or 0) - dist
        pace_vs_target = None
        if pace and state.target_pace:
            pace_vs_target = pace - state.target_pace

        # Goal progress percentage
        goal_progress_pct = None
        if state.goal_distance_km and state.goal_distance_km > 0:
            goal_progress_pct = round((dist / state.goal_distance_km) * 100, 1)

        # Projected finish time based on current pace
        projected_finish_min = None
        if pace and pace > 0 and remaining > 0:
            projected_finish_min = round(elapsed + (remaining * pace), 1)

        return CoachingEvent("distance_milestone", {
            "distance_km":          round(dist, 2),
            "checkpoint_m":         state.checkpoint_interval_m,
            "remaining_km":         round(remaining, 2) if remaining > 0 else 0,
            "current_hr":           hr,
            "current_pace":         pace,
            "target_pace":          state.target_pace,
            "pace_vs_target":       round(pace_vs_target, 2) if pace_vs_target else None,
            "pace_trend":           state.pace_trend(),
            "elapsed_min":          round(elapsed, 1),
            "goal_progress_pct":    goal_progress_pct,
            "projected_finish_min": projected_finish_min,
            "goal_duration_min":    state.goal_duration_min,
        })

    # ── HR critical ──────────────────────────────────────
    if hr > 185 and (now - state.last_hr_warning_at) > 30:
        state.last_hr_warning_at = now
        state.last_proactive_at  = now
        return CoachingEvent("hr_critical", {
            "hr": hr, "distance_km": dist, "elapsed_min": elapsed
        })

    if hr > 175 and (now - state.last_hr_warning_at) > 60:
        state.last_hr_warning_at = now
        state.last_proactive_at  = now
        profile_ceiling = _hr_ceiling_for_type(state.run_type)
        if hr > profile_ceiling:
            return CoachingEvent("hr_high", {
                "hr": hr, "ceiling": profile_ceiling,
                "distance_km": dist, "run_type": state.run_type
            })

    # ── Pace significantly off target ────────────────────
    if pace and state.target_pace and (now - state.last_pace_alert_at) > 90:
        diff = pace - state.target_pace
        if abs(diff) > 1.0 and dist > 0.5:  # only after 500m
            state.last_pace_alert_at = now
            state.last_proactive_at  = now
            return CoachingEvent("pace_off", {
                "current_pace": pace,
                "target_pace":  state.target_pace,
                "diff":         round(diff, 2),
                "direction":    "too slow" if diff > 0 else "too fast",
                "distance_km":  dist,
                "run_type":     state.run_type,
            })

    # ── Periodic check-in (long/easy runs) ───────────────
    checkin_interval = 180 if state.run_type in ("long_run", "easy") else 120
    if (now - state.last_checkin_at) > checkin_interval and (now - state.last_proactive_at) > 60:
        state.last_checkin_at = now
        state.last_proactive_at = now
        return CoachingEvent("checkin", {
            "elapsed_min":  round(elapsed, 1),
            "distance_km":  round(dist, 2),
            "current_hr":   hr,
            "current_pace": pace,
            "target_pace":  state.target_pace,
            "goal_distance": state.goal_distance_km,
            "run_type":     state.run_type,
            "pace_trend":   state.pace_trend(),
        })

    return None


def _hr_ceiling_for_type(run_type: str) -> int:
    ceilings = {
        "easy": 140, "long_run": 150, "tempo": 175,
        "race_pace": 180, "hill": 178, "interval": 185
    }
    return ceilings.get(run_type, 175)