"""
models/database.py
SQLAlchemy ORM models — defines every table in the database.
"""

from datetime import datetime
from sqlalchemy import (
    text,
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, ForeignKey, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config import settings

Base = declarative_base()

# ─────────────────────────────────────────────
# Run Session  (one row = one run)
# ─────────────────────────────────────────────
class RunSession(Base):
    __tablename__ = "run_sessions"

    id               = Column(Integer, primary_key=True, index=True)
    name             = Column(String(200), nullable=True)
    run_type         = Column(String(50), nullable=True)   # easy, tempo, interval, hill, long_run, race_pace
    coaching_mode    = Column(String(50), nullable=True)   # recovery_friendly, threshold_monitor, etc
    interval_config  = Column(JSON, nullable=True)         # {sets, work_distance_m, rest_seconds}
    created_at       = Column(DateTime, default=datetime.utcnow)
    started_at       = Column(DateTime, nullable=True)
    ended_at         = Column(DateTime, nullable=True)

    # Goal the user set before the run
    goal_distance_km  = Column(Float, nullable=True)
    goal_duration_min = Column(Float, nullable=True)
    goal_description  = Column(String(500), nullable=True)

    # Coaching checkpoint interval (meters) — set by user before run
    checkpoint_interval_m = Column(Integer, nullable=True, default=500)

    # Summary filled in when run ends
    total_distance_km  = Column(Float, nullable=True)
    total_duration_min = Column(Float, nullable=True)
    avg_heart_rate     = Column(Float, nullable=True)
    avg_pace_min_km    = Column(Float, nullable=True)
    max_heart_rate     = Column(Float, nullable=True)
    calories_burned    = Column(Float, nullable=True)
    goal_achieved      = Column(Boolean, nullable=True)

    # AI-generated post-run analysis (stored as text)
    ai_analysis = Column(Text, nullable=True)

    # Relationships
    vitals   = relationship("VitalSign",           back_populates="run", cascade="all, delete-orphan")
    messages = relationship("ConversationMessage", back_populates="run", cascade="all, delete-orphan")


# ─────────────────────────────────────────────
# Vital Sign  (one row = one reading from HealthKit)
# ─────────────────────────────────────────────
class VitalSign(Base):
    __tablename__ = "vital_signs"

    id              = Column(Integer, primary_key=True, index=True)
    run_id          = Column(Integer, ForeignKey("run_sessions.id"), nullable=False)
    recorded_at     = Column(DateTime, default=datetime.utcnow)

    heart_rate             = Column(Float, nullable=True)
    heart_rate_variability = Column(Float, nullable=True)
    distance_km            = Column(Float, nullable=True)
    pace_min_km            = Column(Float, nullable=True)
    cadence_spm            = Column(Float, nullable=True)
    calories               = Column(Float, nullable=True)
    vo2_max                = Column(Float, nullable=True)
    respiratory_rate       = Column(Float, nullable=True)
    elevation_m            = Column(Float, nullable=True)

    run = relationship("RunSession", back_populates="vitals")


# ─────────────────────────────────────────────
# Conversation Message  (one row = one message in the run chat)
# ─────────────────────────────────────────────
class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id              = Column(Integer, primary_key=True, index=True)
    run_id          = Column(Integer, ForeignKey("run_sessions.id"), nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    role            = Column(String(20))
    content         = Column(Text)
    vitals_snapshot = Column(JSON, nullable=True)

    run = relationship("RunSession", back_populates="messages")


# ─────────────────────────────────────────────
# DB engine + session factory
# ─────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,   # test connection before use — kills stale connections silently
    pool_recycle=1800,    # recycle connections after 30min — Cloud SQL times out at ~10min idle
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables and run safe migrations on server start."""
    Base.metadata.create_all(bind=engine)
    _run_migrations()

def _run_migrations():
    """Safely add new columns. Each migration runs in its own transaction."""
    migrations = [
        "ALTER TABLE run_sessions ADD COLUMN IF NOT EXISTS name VARCHAR(200)",
        "ALTER TABLE run_sessions ADD COLUMN IF NOT EXISTS run_type VARCHAR(50)",
        "ALTER TABLE run_sessions ADD COLUMN IF NOT EXISTS coaching_mode VARCHAR(50)",
        "ALTER TABLE run_sessions ADD COLUMN IF NOT EXISTS interval_config JSONB",
        "ALTER TABLE run_sessions ADD COLUMN IF NOT EXISTS checkpoint_interval_m INTEGER DEFAULT 500",
        "ALTER TABLE vital_signs ADD COLUMN IF NOT EXISTS hrv FLOAT",
        "ALTER TABLE vital_signs ADD COLUMN IF NOT EXISTS steps INTEGER",
        "ALTER TABLE vital_signs ADD COLUMN IF NOT EXISTS elevation_gain FLOAT",
    ]
    for sql in migrations:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
            print(f"✅ Migration OK: {sql[:60]}")
        except Exception as e:
            print(f"⚠️  Migration skipped ({sql[:40]}): {e}")