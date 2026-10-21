
from contextlib import asynccontextmanager
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import time

from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_config, predict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictResponse(BaseModel):
    prediction: str
    confidence_band: str
    confidence: float
    probability_pneumonia: float
    threshold_used: float
    uncertain_band: list[float]
    disclaimer: str
    inference_ms: float



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate the frozen configuration and checkpoint.
    config, checkpoint_path = load_config()

    # Load the model once, not for every uploaded image.
    model = tf.keras.models.load_model(
        checkpoint_path,
        compile=False,
    )

    app.state.config = config
    app.state.model = model

    yield

    # Release references when the server shuts down.
    app.state.model = None
    app.state.config = None


app = FastAPI(
    title="ChestScan API",
    description=(
        "Chest X-ray pneumonia screening service. "
        "Screening aid only. Not a medical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = getattr(app.state, "model", None) is not None

    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
    )


MAX_FILE_SIZE = 5 * 1024 * 1024
DISCLAIMER = "Screening aid only. Not a medical diagnosis."


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # Check the declared media type first.
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )

    # Read at most 5 MB plus one byte to detect oversized uploads.
    image_bytes = await file.read(MAX_FILE_SIZE + 1)

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MB limit.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Verify that the actual image format matches the declared MIME type.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            actual_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded content is not a valid image.",
        ) from exc

    expected_formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
    }

    if actual_format != expected_formats[file.content_type]:
        raise HTTPException(
            status_code=400,
            detail="Image content does not match the declared file type.",
        )

    started = time.perf_counter()

    try:
        result = predict(
            image_bytes,
            app.state.model,
            app.state.config,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from exc



    inference_ms = (time.perf_counter() - started) * 1000
    probability = result["pneumonia_probability"]
    band = result["uncertain_band"]

    return PredictResponse(
        prediction=result["label"],
        confidence_band=(
            "Uncertain"
            if result["triage_status"] == "uncertain"
            else result["label"]
        ),
        confidence=(
            probability
            if result["label"] == "PNEUMONIA"
            else 1.0 - probability
        ),
        probability_pneumonia=probability,
        threshold_used=result["decision_threshold"],
        uncertain_band=[band["lower"], band["upper"]],
        disclaimer=DISCLAIMER,
        inference_ms=round(inference_ms, 2),
    )
