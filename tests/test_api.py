"""Tests for the surface-serving API."""

from fastapi.testclient import TestClient

from tsx60_iv import api


def test_surface_503_until_loaded() -> None:
    """The surface endpoint fails until a surface is set."""
    client = TestClient(api.app)
    api._surface = None
    assert client.get("/surface").status_code == 503
    api.set_surface({"a": 1})
    response = client.get("/surface")
    assert response.status_code == 200
    assert response.json() == {"a": 1}
