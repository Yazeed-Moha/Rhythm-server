"""
services/run_state.py
Active run state machine — tracks phases, milestones, interval sets,
pace splits, and builds rich context for the AI coach.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from models.schemas import VitalSignInput


# ── Interval phase ─────────────────────────────────────────
class IntervalPhase:
    WARMUP   = "warmup"
    WORK     = "work"
    REST     = "rest"
    COOLDOWN = "cooldown"
    DONE     = "done"


@dataclass
class IntervalSet:
    set_number: int
    started_at: float
    ended_at: Optional[float] = None
    peak_hr: Optional[float] = None
    avg_hr: Optional[float] = None
    completed: bool = False


@dataclass
class RunState:
    """
    Full live state of a run session.
    Updated on every vitals message, drives proactive coaching decisions.
    """

    # ── Identity ────────────────────────────────────────────
    run_id:   int
    run_type: str
    started_at: float = field(default_factory=time.time)

    # ── Goal ────────────────────────────────────────────────
    goal_distance_km:  Optional[float] = None
    goal_duration_min: Optional[float] = None
    goal_description:  Optional[str]   = None
    interval_config:        Optional[dict]  = None  # sets, work_distance_m, rest_sec, warmup_km, cooldown_km, effort
    checkpoint_interval_m:  int             = 500   # user-configured coaching checkpoint distance

    # ── Live vitals ─────────────────────────────────────────
    latest_vitals: VitalSignInput = field(default_factory=VitalSignInput)
    hr_history:    list = field(default_factory=list)     # (timestamp, hr)
    pace_history:  list = field(default_factory=list)     # (timestamp, pace, distance)

    # ── Milestones ──────────────────────────────────────────
    last_milestone_km:    float = 0.0   # last km we announced
    last_proactive_at:    float = 0.0
    last_hr_warning_at:   float = 0.0
    last_checkin_at:      float = 0.0
    last_pace_alert_at:   float = 0.0

    # ── Interval tracking ───────────────────────────────────
    interval_phase:       str   = IntervalPhase.WARMUP
    current_set:          int   = 0    # 0 = not started, 1..N = active set
    completed_sets:       int   = 0
    sets_history:         list  = field(default_factory=list)  # list[IntervalSet]
    phase_started_at:     float = field(default_factory=time.time)
    phase_distance_start: float = 0.0  # distance when phase began

    # ── Conversation ────────────────────────────────────────
    conversation_history: list  = field(default_factory=list)
    mentioned_issues:     list  = field(default_factory=list)  # ["knee pain", "dizzy"] etc.
    language_instruction: str   = "Always respond in English."
    past_runs_summary:    str   = ""

    # ── Flags ───────────────────────────────────────────────
    has_started_moving:   bool  = False
    warmup_announced:     bool  = False
    cooldown_announced:   bool  = False

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60

    @property
    def target_pace(self) -> Optional[float]:
        if self.goal_distance_km and self.goal_duration_min:
            return self.goal_duration_min / self.goal_distance_km
        return None

    @property
    def current_distance(self) -> float:
        return self.latest_vitals.distance_km or 0.0

    @property
    def current_hr(self) -> Optional[float]:
        return self.latest_vitals.heart_rate

    @property
    def current_pace(self) -> Optional[float]:
        return self.latest_vitals.pace_min_km

    def update_vitals(self, vitals: VitalSignInput):
        self.latest_vitals = vitals
        now = time.time()
        if vitals.heart_rate and vitals.heart_rate > 0:
            self.hr_history.append((now, vitals.heart_rate))
            if len(self.hr_history) > 200:
                self.hr_history = self.hr_history[-100:]
        if vitals.pace_min_km and vitals.pace_min_km > 0:
            self.pace_history.append((now, vitals.pace_min_km, vitals.distance_km or 0))
            if len(self.pace_history) > 200:
                self.pace_history = self.pace_history[-100:]
        if vitals.distance_km and vitals.distance_km > 0.05:
            self.has_started_moving = True

    def avg_hr_last_n_seconds(self, seconds: int) -> Optional[float]:
        cutoff = time.time() - seconds
        recent = [hr for ts, hr in self.hr_history if ts >= cutoff]
        return sum(recent) / len(recent) if recent else None

    def avg_pace_last_n_seconds(self, seconds: int) -> Optional[float]:
        cutoff = time.time() - seconds
        recent = [p for ts, p, _ in self.pace_history if ts >= cutoff]
        return sum(recent) / len(recent) if recent else None

    def pace_trend(self) -> str:
        """Returns 'slowing', 'speeding', 'steady' based on recent pace history."""
        if len(self.pace_history) < 6:
            return "steady"
        recent  = [p for _, p, _ in self.pace_history[-3:]]
        earlier = [p for _, p, _ in self.pace_history[-6:-3]]
        avg_r = sum(recent)  / len(recent)
        avg_e = sum(earlier) / len(earlier)
        diff = avg_r - avg_e  # positive = slower (higher min/km)
        if diff > 0.3:  return "slowing"
        if diff < -0.3: return "speeding"
        return "steady"

    # ── Interval helpers ────────────────────────────────────
    @property
    def total_sets(self) -> int:
        if self.interval_config:
            return self.interval_config.get("sets", 6)
        return 6

    @property
    def work_distance_m(self) -> int:
        if self.interval_config:
            return self.interval_config.get("work_distance_m", 400)
        return 400

    @property
    def rest_seconds(self) -> int:
        if self.interval_config:
            return self.interval_config.get("rest_sec", 90)
        return 90

    @property
    def warmup_km(self) -> float:
        if self.interval_config:
            return self.interval_config.get("warmup_km", 1.0)
        return 1.0

    @property
    def cooldown_km(self) -> float:
        if self.interval_config:
            return self.interval_config.get("cooldown_km", 1.0)
        return 1.0

    @property
    def effort_level(self) -> str:
        if self.interval_config:
            return self.interval_config.get("effort", "hard")
        return "hard"

    @property
    def phase_elapsed_seconds(self) -> float:
        return time.time() - self.phase_started_at

    @property
    def phase_distance_covered(self) -> float:
        return self.current_distance - self.phase_distance_start

    def start_phase(self, phase: str):
        self.interval_phase   = phase
        self.phase_started_at = time.time()
        self.phase_distance_start = self.current_distance