"""Tests for the surface-serving API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tsx60_iv import api


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client whose surface file lives in a temp dir and starts missing."""
    monkeypatch.setenv("TSX60_SURFACE", str(tmp_path / "surface.json"))
    monkeypatch.setattr(api, "_surface", None)
    monkeypatch.setattr(api, "_file_cache", None)
    return TestClient(api.app)


def test_surface_503_until_loaded(client: TestClient) -> None:
    """The surface endpoint fails until a surface is set."""
    assert client.get("/surface").status_code == 503
    api.set_surface({"a": 1})
    response = client.get("/surface")
    assert response.status_code == 200
    assert response.json() == {"a": 1}


def test_surface_file_etag_and_reload(client: TestClient, tmp_path: Path) -> None:
    """The file is served, 304 on a matching ETag, and re-read after a rewrite."""
    path = tmp_path / "surface.json"
    path.write_text(json.dumps({"v": 1}))
    first = client.get("/surface")
    assert first.json() == {"v": 1}
    etag = first.headers["etag"]
    assert client.get("/surface", headers={"If-None-Match": etag}).status_code == 304
    path.write_text(json.dumps({"version": 2}))
    stale = {"If-None-Match": etag}
    assert client.get("/surface", headers=stale).json() == {"version": 2}


def test_page_uses_pinned_cdn_libraries(client: TestClient) -> None:
    """The page is served and pins exact CDN versions."""
    html = client.get("/").text
    assert "plotly.js-dist-min@2.35.2" in html
    assert "KaTeX/0.16.9" in html
