"""
Request plumbing shared by the /analyze routers:
  * stream the upload (or allowed fileUrl) to a temp file with a size cap
  * run the CPU-bound engine in a worker thread so the event loop (and /health)
    stays responsive, with a semaphore so only N analyses run at once
  * turn engine output into the response shape the backend stores
"""

import asyncio
import os
import tempfile
import time
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from utils.analysis import generate_report
from utils.pipeline import AnalysisError, EngineResult
from utils.remote_file import validate_file_url

MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp",
                    ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CHUNK = 1024 * 1024

_semaphore = asyncio.Semaphore(max(1, int(os.getenv("ANALYSIS_CONCURRENCY", "1"))))


def _suffix(name: Optional[str], default: str) -> str:
    ext = os.path.splitext(name or "")[1].lower()
    return ext if ext in ALLOWED_SUFFIXES else default


async def save_input(file: Optional[UploadFile], file_url: Optional[str], default_suffix: str) -> str:
    if not file and not file_url:
        raise HTTPException(400, "No file or fileUrl provided")
    if not file:
        validate_file_url(file_url)

    suffix = _suffix(file.filename if file else urlparse(file_url).path, default_suffix)
    fd, path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            if file:
                while chunk := await file.read(_CHUNK):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "File is too large")
                    out.write(chunk)
            else:
                async with httpx.AsyncClient(timeout=300.0, follow_redirects=False) as client:
                    async with client.stream("GET", file_url) as resp:
                        if resp.status_code != 200:
                            raise HTTPException(400, "Failed to download file from URL")
                        async for chunk in resp.aiter_bytes(_CHUNK):
                            written += len(chunk)
                            if written > MAX_UPLOAD_BYTES:
                                raise HTTPException(413, "File is too large")
                            out.write(chunk)
        if written == 0:
            raise HTTPException(400, "Uploaded file is empty")
        return path
    except BaseException:
        _unlink(path)
        raise


def _unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


async def run_analysis(
    kind: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
    upload_id: Optional[str],
    engine: Callable[..., EngineResult],
    default_suffix: str,
    **engine_kwargs,
) -> dict:
    path = await save_input(file, file_url, default_suffix)
    started = time.monotonic()
    try:
        async with _semaphore:
            result: EngineResult = await run_in_threadpool(engine, path, **engine_kwargs)
    except AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _unlink(path)

    report = await generate_report(result.metrics, kind, result.faults)
    overall = result.metrics.get(f"overall{kind.capitalize()}Score", 0)
    meta = dict(result.meta, processing_ms=int((time.monotonic() - started) * 1000))
    print(f"[Analysis] {kind} upload={upload_id} score={overall} frames={meta.get('frames_with_pose')}/"
          f"{meta.get('frames_analysed')} in {meta['processing_ms']} ms")

    return {
        "success": True,
        "type": kind,
        "upload_id": upload_id,
        "frames_analysed": meta.get("frames_analysed"),
        "frames_with_pose": meta.get("frames_with_pose"),
        f"{kind}_metrics": result.metrics,
        "overall_score": overall,
        "faults": result.faults,
        "fault_codes": [f["faultCode"] for f in result.faults],
        "raw_frame_metrics": result.raw_frame_metrics,
        "landmarks": result.landmarks,
        "analysis_meta": meta,
        **report,
    }
