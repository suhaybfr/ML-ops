"""Browser entry point and local static assets for the Study Room UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
ASSETS = Path(__file__).with_name("static") / "tutor"

_ALLOWED_ASSETS = {
    "study.css": "text/css; charset=utf-8",
    "study.js": "application/javascript; charset=utf-8",
}


@router.get("/", include_in_schema=False)
def study_room():
    return FileResponse(ASSETS / "index.html", media_type="text/html; charset=utf-8")


@router.get("/assets/{name}", include_in_schema=False)
def tutor_asset(name: str):
    media_type = _ALLOWED_ASSETS.get(name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(ASSETS / name, media_type=media_type)
