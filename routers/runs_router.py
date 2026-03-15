"""
routers/runs_router.py
REST endpoints for managing runs and the dashboard.

POST   /runs/             → create a new run (before starting)
GET    /runs/             → list all runs
GET    /runs/{id}         → get one run with full details
POST   /runs/{id}/start   → mark run as started (alternative to WebSocket auto-start)
DELETE /runs/{id}         → delete a run
GET    /runs/{id}/vitals  → get all vitals for a run
GET    /dashboard/monthly → monthly stats + AI insights
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import (
    RunGoal, RunSessionCreate, RunSessionOut, RunSessionSummary,
    VitalSignOut, MonthlyStats
)
from services import run_service, coach_service

router = APIRouter(prefix="/runs", tags=["runs"])
dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ─────────────────────────────────────────────
# Run CRUD
# ─────────────────────────────────────────────

@router.post("/", response_model=RunSessionOut, summary="Create a new run session")
def create_run(goal: RunSessionCreate, db: Session = Depends(get_db)):
    """
    Call this before starting a run.
    Returns the run ID — use it to open the WebSocket: ws://server/ws/run/{id}
    """
    run = run_service.create_run(db, goal)
    return run


@router.get("/", response_model=List[RunSessionSummary], summary="List all runs")
def list_runs(limit: int = 50, db: Session = Depends(get_db)):
    return run_service.get_all_runs(db, limit=limit)


@router.get("/{run_id}", response_model=RunSessionOut, summary="Get run details")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = run_service.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.delete("/{run_id}", summary="Delete a run")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    run = run_service.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(run)
    db.commit()
    return {"deleted": run_id}



@router.patch("/{run_id}/rename", summary="Rename a run")
def rename_run(run_id: int, body: dict, db: Session = Depends(get_db)):
    run = db.query(RunSession).filter(RunSession.id == run_id).first()
    if not run:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    run.name = body.get("name", "")
    db.commit()
    return {"id": run_id, "name": run.name}

@router.get("/{run_id}/vitals", response_model=List[VitalSignOut], summary="Get vitals timeline for a run")
def get_vitals(run_id: int, db: Session = Depends(get_db)):
    run = run_service.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_service.get_vitals_for_run(db, run_id)


# ─────────────────────────────────────────────
# Manual run import (runs recorded outside the app)
# ─────────────────────────────────────────────

@router.post("/import", response_model=RunSessionOut, summary="Import a past run manually")
def import_run(data: RunSessionOut, db: Session = Depends(get_db)):
    """
    Lets the user add runs they did before using the app.
    Accepts a full RunSessionOut payload — all fields optional except id is auto-assigned.
    """
    from models.database import RunSession
    run = RunSession(
        started_at=data.started_at,
        ended_at=data.ended_at,
        goal_distance_km=data.goal_distance_km,
        goal_duration_min=data.goal_duration_min,
        goal_description=data.goal_description,
        total_distance_km=data.total_distance_km,
        total_duration_min=data.total_duration_min,
        avg_heart_rate=data.avg_heart_rate,
        avg_pace_min_km=data.avg_pace_min_km,
        max_heart_rate=data.max_heart_rate,
        calories_burned=data.calories_burned,
        goal_achieved=data.goal_achieved,
        ai_analysis="Manually imported run.",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────

@dashboard_router.get(
    "/monthly",
    response_model=MonthlyStats,
    summary="Monthly stats + AI insights"
)
async def monthly_dashboard(
    year: int,
    month: int,
    db: Session = Depends(get_db),
):
    runs = run_service.get_runs_by_month(db, year, month)

    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No completed runs found for {year}-{month:02d}"
        )

    # Aggregate stats
    distances = [r.total_distance_km for r in runs if r.total_distance_km]
    durations  = [r.total_duration_min for r in runs if r.total_duration_min]
    hrs        = [r.avg_heart_rate for r in runs if r.avg_heart_rate]
    paces      = [r.avg_pace_min_km for r in runs if r.avg_pace_min_km]
    goals_hit  = sum(1 for r in runs if r.goal_achieved)

    # Prepare run dicts for AI
    runs_for_ai = [
        {
            "date": str(r.started_at.date()) if r.started_at else None,
            "distance_km": r.total_distance_km,
            "duration_min": r.total_duration_min,
            "avg_hr": r.avg_heart_rate,
            "avg_pace": r.avg_pace_min_km,
            "goal_achieved": r.goal_achieved,
            "coach_note": r.ai_analysis,
        }
        for r in runs
    ]

    ai_insights = await coach_service.generate_monthly_insights(runs_for_ai)

    return MonthlyStats(
        year=year,
        month=month,
        total_runs=len(runs),
        total_distance_km=sum(distances),
        total_duration_min=sum(durations),
        avg_heart_rate=sum(hrs)/len(hrs) if hrs else None,
        avg_pace_min_km=sum(paces)/len(paces) if paces else None,
        goals_achieved=goals_hit,
        ai_insights=ai_insights,
    )