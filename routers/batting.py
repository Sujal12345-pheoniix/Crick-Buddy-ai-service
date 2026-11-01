"""
Batting video analysis.
Pipeline and metric definitions: utils/pipeline.py (analyze_batting).
"""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from utils.pipeline import analyze_batting as run_batting_engine
from utils.runner import run_analysis

router = APIRouter()


@router.post("/batting")
async def analyze_batting(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    batting_style: Optional[str] = Form(None),
):
    return await run_analysis(
        "batting", file, fileUrl, upload_id, run_batting_engine, ".mp4",
        batting_style=batting_style,
    )
