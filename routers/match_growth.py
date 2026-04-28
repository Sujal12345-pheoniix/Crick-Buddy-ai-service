"""
Match Growth Analysis Router
Takes player match performance entries and returns trend + feedback.
Uses Gemini when configured, falls back to rule-based feedback otherwise.
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


router = APIRouter()


class MatchPerformance(BaseModel):
    matchDate: Optional[str] = None
    runs: float = 0
    balls: float = 0
    wickets: float = 0
    oversBowled: float = 0
    runsConceded: float = 0
    catches: float = 0
    stumpings: float = 0
    computed: Optional[Dict[str, Any]] = None


class MatchGrowthRequest(BaseModel):
    latest: MatchPerformance
    previous: Optional[MatchPerformance] = None
    history: List[MatchPerformance] = Field(default_factory=list)


def _rule_based(latest: dict, previous: Optional[dict]) -> dict:
    strengths = []
    weaknesses = []
    suggestions = []

    overall = float(latest.get("overallScore", 0) or 0)
    batting = float(latest.get("battingScore", 0) or 0)
    bowling = float(latest.get("bowlingScore", 0) or 0)
    fielding = float(latest.get("fieldingScore", 0) or 0)
    sr = float(latest.get("strikeRate", 0) or 0)
    eco = float(latest.get("economy", 0) or 0)

    if batting >= 70:
        strengths.append(
            "Batting impact is strong - good run conversion and tempo."
        )
    if bowling >= 70:
        strengths.append(
            "Bowling spells are impactful - wicket-taking and/or control is improving."
        )
    if fielding >= 50:
        strengths.append(
            "Fielding contribution is consistent - good involvement in key moments."
        )

    if sr < 110 and batting < 60:
        weaknesses.append("Batting tempo is below match-impact level.")
        suggestions.append(
            "Do 24-ball scenario drills: target SR 120+ with controlled risk."
        )

    if eco > 8 and bowling < 60:
        weaknesses.append("Run containment in bowling needs improvement.")
        suggestions.append(
            "Death-overs accuracy drill: yorker/slower-ball at target zones "
            "(3x18 balls)."
        )

    if fielding < 30:
        weaknesses.append("Low fielding impact in recent matches.")
        suggestions.append(
            "15-minute reaction catching + pickup/throw routine before nets."
        )

    trend = "stable"
    if previous:
        prev_overall = float(previous.get("overallScore", 0) or 0)
        delta = overall - prev_overall
        if delta >= 5:
            trend = "improving"
            suggestions.append(
                "Great progress - add pressure simulations (chase/defend scenarios)."
            )
        elif delta <= -5:
            trend = "declining"
            suggestions.append(
                "Reset baseline - prioritize consistency and lower-risk options "
                "for 1-2 games."
            )

    if not strengths:
        strengths.append("Building consistency across match phases.")
    if not weaknesses:
        weaknesses.append("No major drop detected - focus on repeatable execution.")
    if not suggestions:
        suggestions.append(
            "Maintain routine and review 1 key phase after each match "
            "(PP/middle/death)."
        )

    return {
        "trend": trend,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


@router.post("/match-growth")
async def match_growth(req: MatchGrowthRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    latest = req.latest.computed or {}
    previous = (req.previous.computed if req.previous else None) or None

    if not api_key:
        return {
            "success": True,
            "analysis": _rule_based(latest, previous),
            "source": "fallback",
        }

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "success": True,
            "analysis": _rule_based(latest, previous),
            "source": "fallback",
        }

    client = genai.Client(api_key=api_key)

    prompt = {
        "latest": req.latest.model_dump(),
        "previous": req.previous.model_dump() if req.previous else None,
        "history_len": len(req.history or []),
    }

    system_instruction = (
        "You are a professional cricket performance analyst. "
        "Given structured match stats and computed impact scores, return "
        "precise, actionable feedback."
    )

    user_prompt = (
        "Return JSON ONLY with keys: trend, strengths, weaknesses, suggestions.\n"
        "trend must be one of: improving, stable, declining.\n"
        "strengths/weaknesses/suggestions must be arrays of strings.\n"
        f"Input:\n{prompt}"
    )

    try:
        resp = await client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                response_mime_type="application/json",
                max_output_tokens=400,
            ),
        )
        import json

        parsed = json.loads(resp.text)
        analysis = {
            "trend": parsed.get("trend") or "stable",
            "strengths": parsed.get("strengths") or [],
            "weaknesses": parsed.get("weaknesses") or [],
            "suggestions": parsed.get("suggestions") or [],
        }
        return {"success": True, "analysis": analysis, "source": "ai"}
    except Exception:
        return {
            "success": True,
            "analysis": _rule_based(latest, previous),
            "source": "fallback",
        }

