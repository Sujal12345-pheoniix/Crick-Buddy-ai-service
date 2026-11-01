"""
Bowling video analysis.
Pipeline and metric definitions: utils/pipeline.py (analyze_bowling).
"""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from utils.pipeline import analyze_bowling as run_bowling_engine
from utils.runner import run_analysis

router = APIRouter()


@router.post("/bowling")
async def analyze_bowling(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    bowling_style: Optional[str] = Form(None),
):
    return await run_analysis(
        "bowling", file, fileUrl, upload_id, run_bowling_engine, ".mp4",
        bowling_style=bowling_style,
    )
