"""
CrickBuddy AI Service — FastAPI Application
============================================
Start-up sequence:
  1. Load env vars
  2. Mount timeout middleware (8 min hard cap)
  3. Mount CORS
  4. Register routers
  5. ON STARTUP: pre-load YOLO + MediaPipe in a thread-pool executor
     so the first real request is served without any model-load delay.
"""

import os
import asyncio
import concurrent.futures

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from routers import batting, bowling, posture, chatbot
from routers import match_growth

load_dotenv()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crick-Buddy AI Service",
    description="MediaPipe + OpenCV + YOLOv8 cricket analysis microservice",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)


# ─── Request Timeout Middleware ───────────────────────────────────────────────
# Hard cap per-request to avoid hanging workers on Render's free tier.
# Set REQUEST_TIMEOUT_SECS env var to override (default: 480 = 8 min).
REQUEST_TIMEOUT_SECS = int(os.getenv("REQUEST_TIMEOUT_SECS", "480"))


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "detail": (
                    f"Analysis exceeded the {REQUEST_TIMEOUT_SECS}s server time limit. "
                    "Try uploading a shorter video (< 60 seconds)."
                ),
            },
        )


# ─── CORS ─────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_env_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

ALLOWED_ORIGINS = list(set([
    "https://crickbuddy.tech",
    "https://www.crickbuddy.tech",
    "https://crick-buddy-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
] + _env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(batting.router,     prefix="/analyze", tags=["Batting Analysis"])
app.include_router(bowling.router,     prefix="/analyze", tags=["Bowling Analysis"])
app.include_router(posture.router,     prefix="/analyze", tags=["Posture Analysis"])
app.include_router(chatbot.router,                        tags=["Chatbot"])
app.include_router(match_growth.router,                   tags=["Match Growth"])


# ─── Startup: pre-load all models ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Pre-load YOLO and MediaPipe Pose in a thread-pool executor at startup.
    This ensures every subsequent request is served by already-warm models,
    eliminating the first-request cold-start that caused timeouts.
    """
    print("🚀 CrickBuddy AI Service starting up...")
    loop = asyncio.get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, _run_warmup)
    except Exception as e:
        # Non-fatal — service still works; first request will load models
        print(f"⚠️  Startup warmup failed (non-fatal): {e}")
    print("✅ Startup complete — ready to serve requests")


def _run_warmup():
    """Synchronous warmup helper (runs in thread pool)."""
    try:
        from utils.analysis import warmup_models
        warmup_models()
    except Exception as e:
        print(f"⚠️  warmup_models error: {e}")


# ─── Health & Root ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "CrickBuddy AI Service Running 🚀", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "crick-buddy-ai",
        "version": "2.0.0",
        "timeout_secs": REQUEST_TIMEOUT_SECS,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("AI_SERVICE_PORT", "8000")))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        timeout_keep_alive=300,
        log_level="info",
    )
