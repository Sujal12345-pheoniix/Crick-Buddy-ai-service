"""
Guards for downloading media from a caller-supplied fileUrl.
Only HTTPS URLs on hosts listed in ALLOWED_FILE_HOSTS are accepted, so the
service cannot be used to fetch internal or arbitrary addresses.
"""

import os
from urllib.parse import urlparse

from fastapi import HTTPException


def _allowed_hosts() -> list:
    raw = os.getenv("ALLOWED_FILE_HOSTS", "res.cloudinary.com")
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def validate_file_url(file_url: str) -> None:
    """Raise HTTP 400 unless file_url is an https URL on an allowed host."""
    try:
        parsed = urlparse(file_url)
    except ValueError:
        raise HTTPException(400, "Invalid fileUrl")

    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise HTTPException(400, "fileUrl must be a plain https URL")

    if not any(host == h or host.endswith("." + h) for h in _allowed_hosts()):
        raise HTTPException(400, "fileUrl host is not allowed")
