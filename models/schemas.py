"""
models/schemas.py
Pydantic schemas — shapes of data going in and out of the API.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Vital Signs
# ─────────────────────────────────────────────
class VitalSignInput(BaseModel):
    """Sent by the iPhone app every few seconds during a run."""
    heart_rate:             Optional[float] = None
    heart_rate_variability: Optional[float] = None
    distance_km:            Optional[float] = None
    pace_min_km:            Optional[float] = None
    cadence_spm:            Optional[float] = None
    calories:               Optional[float] = None
    vo2_max:                Optional[float] = None
    respiratory_rate:       Optional[float] = None
    elevation_m:            Optional[float] = None

class VitalSignOut(VitalSignInput):
    id:          int
    run_id:      int
    recorded_at: datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# Run Session
# ─────────────────────────────────────────────
class RunGoal(BaseModel):
    """User sets this before starting a run."""
    goal_distance_km:  Optional[float]  = None
    goal_duration_min: Optional[float]  = None
    goal_description:  Optional[str]    = None   # e.g. "Run 10km in 50 minutes"

class RunSessionCreate(RunGoal):
    pass

class RunSessionOut(BaseModel):
    id:                int
    created_at:        datetime
    started_at:        Optional[datetime]
    ended_at:          Optional[datetime]
    goal_distance_km:  Optional[float]
    goal_duration_min: Optional[float]
    goal_description:  Optional[str]
    total_distance_km: Optional[float]
    total_duration_min:Optional[float]
    avg_heart_rate:    Optional[float]
    avg_pace_min_km:   Optional[float]
    max_heart_rate:    Optional[float]
    calories_burned:   Optional[float]
    goal_achieved:     Optional[bool]
    ai_analysis:       Optional[str]

    class Config:
        from_attributes = True

class RunSessionSummary(BaseModel):
    """Minimal card used in lists / dashboard."""
    id:                int
    started_at:        Optional[datetime]
    ended_at:          Optional[datetime]
    goal_distance_km:  Optional[float]
    goal_duration_min: Optional[float]
    total_distance_km: Optional[float]
    total_duration_min:Optional[float]
    avg_heart_rate:    Optional[float]
    goal_achieved:     Optional[bool]

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────
class ChatMessage(BaseModel):
    """A single message sent by the user during a run."""
    content: str

class CoachResponse(BaseModel):
    """What the server sends back to the app (spoken aloud via TTS)."""
    message:          str
    urgency:          str = "normal"   # "normal" | "warning" | "critical"
    suggested_pace:   Optional[str] = None   # min/km target if AI recommends change
    coaching_action:  Optional[str]   = None   # e.g. "slow_down" | "speed_up" | "maintain"


# ─────────────────────────────────────────────
# WebSocket messages (real-time channel)
# ─────────────────────────────────────────────
class WSIncoming(BaseModel):
    """
    Generic envelope the iPhone app sends over the WebSocket.
    type = "vitals"  → payload is VitalSignInput
    type = "chat"    → payload is { "content": "..." }
    type = "end_run" → payload is {}
    """
    type:    str           # "vitals" | "chat" | "end_run"
    payload: dict

class WSOutgoing(BaseModel):
    """
    Generic envelope the server sends back.
    type = "coach"       → payload is CoachResponse
    type = "analysis"    → payload is { "text": "..." }
    type = "run_ended"   → payload is RunSessionOut
    type = "error"       → payload is { "detail": "..." }
    """
    type:    str
    payload: dict


# ─────────────────────────────────────────────
# Dashboard / Analytics
# ─────────────────────────────────────────────
class MonthlyStats(BaseModel):
    year:              int
    month:             int
    total_runs:        int
    total_distance_km: float
    total_duration_min:float
    avg_heart_rate:    Optional[float]
    avg_pace_min_km:   Optional[float]
    goals_achieved:    int
    ai_insights:       Optional[str]   # AI-generated narrative for the month