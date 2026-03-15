"""
services/run_service.py
All database operations for runs, vitals, and messages.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import extract, func

from models.database import RunSession, VitalSign, ConversationMessage
from models.schemas import RunGoal, VitalSignInput


# ─────────────────────────────────────────────
# Run Session CRUD
# ─────────────────────────────────────────────

def create_run(db: Session, goal: RunGoal) -> RunSession:
    run = RunSession(
        goal_distance_km=goal.goal_distance_km,
        goal_duration_min=goal.goal_duration_min,
        goal_description=goal.goal_description,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def start_run(db: Session, run_id: int) -> RunSession:
    run = db.query(RunSession).filter(RunSession.id == run_id).first()
    if run:
        run.started_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
    return run


def end_run(db: Session, run_id: int, ai_analysis: str) -> RunSession:
    run = db.query(RunSession).filter(RunSession.id == run_id).first()
    if not run:
        return None

    run.ended_at = datetime.utcnow()
    run.ai_analysis = ai_analysis

    # Calculate summary stats from stored vitals
    vitals = db.query(VitalSign).filter(VitalSign.run_id == run_id).all()
    if vitals:
        hrs = [v.heart_rate for v in vitals if v.heart_rate]
        paces = [v.pace_min_km for v in vitals if v.pace_min_km]

        run.avg_heart_rate = sum(hrs) / len(hrs) if hrs else None
        run.max_heart_rate = max(hrs) if hrs else None
        run.avg_pace_min_km = sum(paces) / len(paces) if paces else None

        last_vital = vitals[-1]
        run.total_distance_km = last_vital.distance_km
        run.calories_burned = last_vital.calories

        if run.started_at and run.ended_at:
            run.total_duration_min = (run.ended_at - run.started_at).seconds / 60

        # Check if goal was achieved
        if run.goal_distance_km and run.goal_duration_min:
            run.goal_achieved = (
                (run.total_distance_km or 0) >= run.goal_distance_km and
                (run.total_duration_min or 999) <= run.goal_duration_min
            )

    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: int) -> Optional[RunSession]:
    return db.query(RunSession).filter(RunSession.id == run_id).first()


def get_all_runs(db: Session, limit: int = 50) -> List[RunSession]:
    return (
        db.query(RunSession)
        .filter(RunSession.started_at.isnot(None))
        .order_by(RunSession.started_at.desc())
        .limit(limit)
        .all()
    )


def get_runs_by_month(db: Session, year: int, month: int) -> List[RunSession]:
    return (
        db.query(RunSession)
        .filter(
            extract("year",  RunSession.started_at) == year,
            extract("month", RunSession.started_at) == month,
            RunSession.ended_at.isnot(None),
        )
        .order_by(RunSession.started_at)
        .all()
    )


# ─────────────────────────────────────────────
# Vitals CRUD
# ─────────────────────────────────────────────

def save_vital(db: Session, run_id: int, vitals: VitalSignInput) -> VitalSign:
    v = VitalSign(run_id=run_id, **vitals.model_dump())
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


def get_vitals_for_run(db: Session, run_id: int) -> List[VitalSign]:
    return db.query(VitalSign).filter(VitalSign.run_id == run_id).order_by(VitalSign.recorded_at).all()


# ─────────────────────────────────────────────
# Conversation Messages
# ─────────────────────────────────────────────

def save_message(db: Session, run_id: int, role: str, content: str, vitals_snapshot: dict = None):
    msg = ConversationMessage(
        run_id=run_id,
        role=role,
        content=content,
        vitals_snapshot=vitals_snapshot,
    )
    db.add(msg)
    db.commit()


def get_messages_for_run(db: Session, run_id: int) -> List[ConversationMessage]:
    return (
        db.query(ConversationMessage)
        .filter(ConversationMessage.run_id == run_id)
        .order_by(ConversationMessage.created_at)
        .all()
    )


# ─────────────────────────────────────────────
# Historical patterns summary (injected into AI context)
# ─────────────────────────────────────────────

def build_past_runs_summary(db: Session, limit: int = 5) -> Optional[str]:
    """
    Fetches the last N completed runs and formats them as a
    compact text summary for the AI's system prompt.
    """
    runs = (
        db.query(RunSession)
        .filter(RunSession.ended_at.isnot(None))
        .order_by(RunSession.started_at.desc())
        .limit(limit)
        .all()
    )

    if not runs:
        return None

    lines = []
    for r in runs:
        date = r.started_at.strftime("%Y-%m-%d") if r.started_at else "?"
        dist = f"{r.total_distance_km:.1f}" if r.total_distance_km else "?"
        dur  = f"{r.total_duration_min:.0f}" if r.total_duration_min else "?"
        hr   = f"{r.avg_heart_rate:.0f}" if r.avg_heart_rate else "?"
        pace = f"{r.avg_pace_min_km:.2f}" if r.avg_pace_min_km else "?"
        line = (
            f"- {date}: {dist}km in {dur}min | "
            f"avg HR {hr}bpm | avg pace {pace} min/km | "
            f"goal {'✓' if r.goal_achieved else '✗'}"
        )
        if r.ai_analysis:
            line += f"\n  Coach note: {r.ai_analysis[:120]}..."
        lines.append(line)

    return "\n".join(lines)