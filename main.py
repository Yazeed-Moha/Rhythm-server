from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.database import init_db
from routers.runs_router import router as runs_router, dashboard_router
from routers.websocket_router import router as ws_router
from routers.whisper_router import router as whisper_router
from routers.tts_router import router as tts_router

init_db()

app = FastAPI(
    title="Running Coach AI Server",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)
app.include_router(dashboard_router)
app.include_router(ws_router)
app.include_router(whisper_router)
app.include_router(tts_router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model": "gemini-2.5-flash-native-audio-latest"}


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Running Coach AI Server is running 🏃",
        "docs": "/docs",
        "websocket": "ws://<host>/ws/run/{run_id}",
    }