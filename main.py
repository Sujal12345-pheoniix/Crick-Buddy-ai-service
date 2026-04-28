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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batting.router, prefix="/analyze", tags=["Batting Analysis"])
app.include_router(bowling.router, prefix="/analyze", tags=["Bowling Analysis"])
app.include_router(posture.router, prefix="/analyze", tags=["Posture Analysis"])
app.include_router(chatbot.router, tags=["Chatbot"])
app.include_router(match_growth.router, tags=["Match Growth"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "crick-buddy-ai", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
