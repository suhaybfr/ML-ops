"""Browser entry point and local assets for the tutor."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
ASSETS = Path(__file__).with_name("static") / "tutor"


@router.get("/", include_in_schema=False)
def study_room():
    return FileResponse(ASSETS / "index.html", media_type="text/html")


@router.get("/assets/{name}", include_in_schema=False)
def tutor_asset(name: str):
    types = {"study.css": "text/css", "study.js": "application/javascript"}
    if name not in types:
        raise HTTPException(404)
    return FileResponse(ASSETS / name, media_type=types[name])
