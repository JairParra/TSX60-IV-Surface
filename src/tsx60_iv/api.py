"""FastAPI interface for serving volatility-surface snapshots."""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

app = FastAPI(title="TSX60 IV Surface")
_surface: dict[str, Any] | None = None
_file_cache: tuple[str, dict[str, Any]] | None = None
_PAGE = Path(__file__).parent / "static" / "index.html"


def set_surface(surface: dict[str, Any]) -> None:
    """Set an in-memory surface; it takes precedence over the surface file."""
    global _surface
    _surface = surface


def _current() -> tuple[dict[str, Any] | None, str | None]:
    """Return the surface and its ETag, re-reading the file only when it changes."""
    global _file_cache
    if _surface is not None:
        return _surface, None
    path = Path(os.environ.get("TSX60_SURFACE", "data/surface.json"))
    try:
        info = path.stat()
    except FileNotFoundError:
        return None, None
    # size guards against coarse filesystem timestamps on quick rewrites
    etag = f'"{info.st_mtime_ns}-{info.st_size}"'
    if _file_cache is None or _file_cache[0] != etag:
        _file_cache = (etag, json.loads(path.read_text(encoding="utf-8")))
    return _file_cache[1], etag


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    """Serve the 3D surface page."""
    return FileResponse(_PAGE)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.get(
    "/surface",
    responses={
        200: {
            "description": "Latest calibrated surface document.",
            "headers": {
                "ETag": {
                    "description": (
                        "Entity tag representing the version of the surface file."
                    ),
                    "schema": {"type": "string"},
                }
            },
        },
        304: {
            "description": (
                "Surface not modified; the client's cached version matching "
                "If-None-Match is current."
            ),
            "headers": {
                "ETag": {
                    "description": (
                        "Entity tag representing the version of the surface file."
                    ),
                    "schema": {"type": "string"},
                }
            },
        },
        503: {
            "description": "Surface not loaded yet.",
        },
    },
)
def surface(request: Request) -> Response:
    """Return the latest surface; 304 when the client already has it."""
    data, etag = _current()
    if data is None:
        raise HTTPException(status_code=503, detail="surface not loaded")
    headers = {"ETag": etag} if etag else {}
    if etag and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(data, headers=headers)
