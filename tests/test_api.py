
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.main as api


@pytest.fixture
def client(monkeypatch):
    """Create a test client without loading the real model."""

    # Supply placeholder state because we intentionally skip model startup.
    api.app.state.model = object()
    api.app.state.config = {}

    # Return a predictable result instead of running neural-network inference.
    def fake_predict(image_bytes, model, config):
        return {
            "pneumonia_probability": 0.95,
            "label": "PNEUMONIA",
            "triage_status": "uncertain",
            "decision_threshold": 0.905344,
            "uncertain_band": {
                "lower": 0.830344,
                "upper": 0.980344,
            },
        }

    monkeypatch.setattr(api, "predict", fake_predict)

    # Without a context manager, TestClient does not run lifespan startup.
    return TestClient(api.app)


def make_png_bytes():
    """Create a small, valid PNG image entirely in memory."""
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_valid_png_returns_expected_response(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.png",
                make_png_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] == "PNEUMONIA"
    assert body["confidence_band"] == "Uncertain"
    assert body["probability_pneumonia"] == 0.95
    assert body["threshold_used"] == 0.905344
    assert body["uncertain_band"] == [0.830344, 0.980344]
    assert body["disclaimer"] == (
        "Screening aid only. Not a medical diagnosis."
    )
    assert body["inference_ms"] >= 0


def test_unsupported_media_type_returns_415(client):
    response = client.post(
        "/predict",
        files={"file": ("sample.gif", b"content", "image/gif")},
    )

    assert response.status_code == 415


def test_oversized_upload_returns_413(client):
    oversized_content = b"x" * (api.MAX_FILE_SIZE + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_corrupt_image_returns_400(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupt.png",
                b"not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_mismatched_image_format_returns_400(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "mislabelled.jpg",
                make_png_bytes(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
