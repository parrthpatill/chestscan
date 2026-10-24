import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

import backend.main as api


IMAGE_PATH = Path(
    os.environ.get(
        "CHESTSCAN_PARITY_IMAGE",
        "sample_chest_xray.jpeg",
    )
)

pytestmark = pytest.mark.skipif(
    not IMAGE_PATH.is_file(),
    reason="Set CHESTSCAN_PARITY_IMAGE to a local JPEG sample to run real-model parity.",
)


def test_real_model_api_matches_tensorflow_reference():
    image_bytes = IMAGE_PATH.read_bytes()

    # Use the API's real startup so it loads the configured frozen checkpoint.
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    IMAGE_PATH.name,
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 200, response.text
        api_probability = response.json()["probability_pneumonia"]

        # Independently reproduce src/data.py's evaluation preprocessing.
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32)
        reference_batch = tf.expand_dims(image, axis=0)

        reference_probability = float(
            client.app.state.model.predict(
                reference_batch,
                verbose=0,
            ).reshape(-1)[0]
        )

    assert np.isfinite(api_probability)
    assert abs(api_probability - reference_probability) < 1e-5
