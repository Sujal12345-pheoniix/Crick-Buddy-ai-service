import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import batting, bowling, posture, chatbot
from routers import match_growth

load_dotenv()

app = FastAPI(
    title="Crick-Buddy AI Service",
    description="MediaPipe + OpenCV powered cricket analysis microservice",
    version="1.0.0"
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Build allowed origins from environment (comma-separated) + hardcoded known domains
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_env_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

ALLOWED_ORIGINS = list(set([
    "https://crickbuddy.tech",
    "https://www.crickbuddy.tech",
    "https://crick-buddy-frontend.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
] + _env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(batting.router, prefix="/analyze", tags=["Batting Analysis"])
app.include_router(bowling.router, prefix="/analyze", tags=["Bowling Analysis"])
app.include_router(posture.router, prefix="/analyze", tags=["Posture Analysis"])
app.include_router(chatbot.router, tags=["Chatbot"])
app.include_router(match_growth.router, tags=["Match Growth"])


@app.get("/")
def root():
    return {"message": "AI Service Running 🚀", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "crick-buddy-ai", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("AI_SERVICE_PORT", "8000")))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
