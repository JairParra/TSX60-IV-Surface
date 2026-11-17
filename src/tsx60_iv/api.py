"""FastAPI interface for serving volatility-surface snapshots."""

from typing import Any

from fastapi import FastAPI

app = FastAPI(title="TSX60 IV Surface")
_surface: dict[str, Any] | None = None


def set_surface(surface: dict[str, Any]) -> None:
    """Set the in-memory surface served by the API."""
    global _surface
    _surface = surface


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.get("/surface")
def surface() -> dict[str, Any]:
    """Return the latest loaded surface."""
    return _surface or {}
