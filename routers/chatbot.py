"""
AI Cricket Coach Chatbot Router
Provides personalized cricket advice based on player profile and question.
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    user_profile: Optional[Dict] = {}

async def get_contextual_response(message: str, profile: dict, history: list) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"Hello {profile.get('name', 'Player')}! Gemini API key is not configured. My best advice for a {profile.get('experienceLevel', 'beginner')} {profile.get('playerType', 'batsman')} is to focus on fundamentals and watch the ball closely."

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "Gemini SDK is not installed. Run: pip install google-genai"

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system_instruction = f"""You are a professional cricket coach named 'CrickBuddy AI'.
You are coaching a player with the following profile:
Name: {profile.get('name', 'Unknown')}
Role: {profile.get('playerType', 'Batsman')}
Experience Level: {profile.get('experienceLevel', 'Beginner')}
Batting Style: {profile.get('battingStyle', 'Right-handed')}
Bowling Style: {profile.get('bowlingStyle', 'Right-arm medium')}

Provide concise, highly customized, and actionable cricket coaching advice.
Format clearly and be encouraging. Keep it under 150 words."""

    turns = []
    for msg in history[-5:]:
        prefix = "User" if msg.role == "user" else "Coach"
        turns.append(f"{prefix}: {msg.content}")
    transcript = "\n".join(turns)
    user_block = f"{transcript}\nUser: {message}\nCoach:"

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=user_block,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=300,
            ),
        )
        return (response.text or "").strip() or "I could not form a reply. Please try again."
    except Exception as e:
        print(f"Gemini error: {e}")
        return "I'm having trouble connecting to my coaching brain right now. Please try again later."

@router.post("/chatbot")
async def cricket_chatbot(request: ChatRequest):
    """AI Cricket Coach chatbot endpoint passing data to Gemini API."""
    try:
        profile = request.user_profile or {}
        history = request.history or []
        reply = await get_contextual_response(request.message, profile, history)
        return {"success": True, "reply": reply}
    except Exception as e:
        return {
            "success": False,
            "reply": "I encountered an issue processing your question. Please try again.",
            "error": str(e)
        }
