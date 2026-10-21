
"""ChestScan inference: artifact and configuration validation."""

import hashlib
import json
from pathlib import Path

# Resolve paths from the repository root, not the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "artifacts" / "config.json"


def load_config() -> tuple[dict, Path]:
    """Load and validate the frozen model configuration."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    # Resolve the checkpoint path recorded in the configuration.
    checkpoint = PROJECT_ROOT / config["model"]["checkpoint"]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    # Protect the frozen experiment by checking its recorded checksum.
    expected_hash = config["model"].get("sha256")
    if expected_hash:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != expected_hash:
            raise ValueError("Checkpoint SHA-256 does not match config.json.")

    # Confirm that the expected model and frozen inference settings are present.
    assert config["model"]["architecture"] == "MobileNetV2"
    assert config["input"]["image_size"] == [224, 224]
    assert config["input"]["channels"] == 3
    assert config["input"]["preprocessing"] == "embedded_in_model"
    assert config["decision_threshold"] == 0.905344
    assert config["uncertain_band"]["lower"] == 0.830344
    assert config["uncertain_band"]["upper"] == 0.980344

    return config, checkpoint

import io

import numpy as np
from PIL import Image, UnidentifiedImageError


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into one RGB model input batch."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Uploaded content is not a valid image.") from exc

    # The model contains its own Rescaling layer: do not normalize here.
    return np.expand_dims(pixels, axis=0)

def predict(image_bytes: bytes, model, config: dict) -> dict:
    """Run frozen-model inference and apply saved screening thresholds."""
    batch = preprocess(image_bytes)
    outputs = model.predict(batch, verbose=0)
    pneumonia_probability = float(outputs[0][0])

    if not np.isfinite(pneumonia_probability) or not 0.0 <= pneumonia_probability <= 1.0:
        raise ValueError("Model returned an invalid probability.")

    threshold = float(config["decision_threshold"])
    lower = float(config["uncertain_band"]["lower"])
    upper = float(config["uncertain_band"]["upper"])

    label = (
        config["class_names"]["1"]
        if pneumonia_probability >= threshold
        else config["class_names"]["0"]
    )

    triage_status = (
        "uncertain"
        if lower <= pneumonia_probability <= upper
        else "outside_uncertain_band"
    )

    return {
        "pneumonia_probability": pneumonia_probability,
        "label": label,
        "triage_status": triage_status,
        "decision_threshold": threshold,
        "uncertain_band": {"lower": lower, "upper": upper},
    }