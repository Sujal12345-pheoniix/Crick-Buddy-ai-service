"""
Body posture analysis from an image or a short video.
Pipeline and metric definitions: utils/pipeline.py (analyze_posture).
"""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from utils.pipeline import analyze_posture as run_posture_engine
from utils.runner import run_analysis

router = APIRouter()


@router.post("/posture")
async def analyze_posture(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
):
    return await run_analysis("posture", file, fileUrl, upload_id, run_posture_engine, ".jpg")
