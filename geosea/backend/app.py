import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from .remote_sensing import (
        RemoteSensingError,
        get_sentinel2_observation,
        get_sentinel3_observation,
    )
except ImportError:
    from remote_sensing import (
        RemoteSensingError,
        get_sentinel2_observation,
        get_sentinel3_observation,
    )

try:
    from .bio_oracle import BioOracleError, get_environment
    from .analysis import router as analysis_router
except ImportError:
    from bio_oracle import BioOracleError, get_environment
    from analysis import router as analysis_router

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / "models" / "geosea_rf_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "geosea_rf_metadata.json"

FEATURES = [
    "temperature",
    "salinity",
    "chlorophyll",
    "nitrate",
    "phosphate",
    "pH",
    "PAR",
    "kdpar",
]

MODEL_ID = "random_forest"
MODEL_LABEL = "Random Forest"

app = FastAPI(
    title="GeoSea ML Inference API",
    version="1.1.0",
    description="Inference service for the GeoSea Random Forest habitat suitability model.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "GEOSEA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)

_model = None


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


def load_artifacts():
    global _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest artifact not found: {MODEL_PATH}. "
            "Place geosea_rf_model.joblib in backend/models/."
        )

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise TypeError("The loaded artifact is not a compatible scikit-learn classifier.")

    n_features = getattr(model, "n_features_in_", None)
    if n_features != len(FEATURES):
        raise ValueError(
            f"Random Forest expects {n_features} features, but GeoSea requires {len(FEATURES)}."
        )

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != FEATURES:
        raise ValueError(
            "Random Forest feature order does not match the GeoSea production feature order: "
            + ", ".join(FEATURES)
        )

    _model = model


@app.on_event("startup")
def startup():
    try:
        load_artifacts()
    except Exception as exc:
        print(f"[GeoSea] Random Forest not loaded: {exc}")


class SatelliteObservationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    lookback_days: int = Field(365, ge=1, le=1825)


def _satellite_response(fetcher, request: SatelliteObservationRequest):
    try:
        return fetcher(request.latitude, request.longitude, request.lookback_days)
    except RemoteSensingError as exc:
        # Earth Engine setup/coverage issues are returned as readable API details,
        # not a missing-route 404 or a raw server traceback.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Satellite imagery could not be retrieved for this location. Check Earth Engine access and try again.",
        ) from exc


@app.post("/remote-sensing/sentinel2")
def sentinel2_observation(request: SatelliteObservationRequest):
    """Retrieve a Sentinel-2 scene for a selected map location."""
    return _satellite_response(get_sentinel2_observation, request)


@app.post("/remote-sensing/sentinel3")
def sentinel3_observation(request: SatelliteObservationRequest):
    """Retrieve Sentinel-3 OLCI imagery when Sentinel-2 is unavailable."""
    return _satellite_response(get_sentinel3_observation, request)


@app.get("/health")
def health():
    if _model is None:
        return {
            "status": "degraded",
            "model": MODEL_ID,
            "model_label": MODEL_LABEL,
            "ready": False,
            "message": "Random Forest artifact is not loaded.",
        }

    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "ready": True,
        "features": FEATURES,
        "environmental_source": "Bio-ORACLE v3.0",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    global _model

    if _model is None:
        try:
            load_artifacts()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"GeoSea model is unavailable: {exc}") from exc

    try:
        environmental = get_environment(request.latitude, request.longitude)
    except BioOracleError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bio-ORACLE environmental lookup failed: {exc}",
        ) from exc

    values_dict = environmental["values"]
    try:
        values = np.array([[float(values_dict[feature]) for feature in FEATURES]], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Bio-ORACLE returned an incomplete environmental record.") from exc

    if not np.isfinite(values).all():
        raise HTTPException(status_code=502, detail="Bio-ORACLE returned a non-finite environmental value.")

    try:
        # The feature order is validated during artifact loading. A DataFrame
        # preserves the feature names stored by scikit-learn.
        feature_frame = pd.DataFrame(values, columns=FEATURES)
        probability = float(_model.predict_proba(feature_frame)[0, 1])
        prediction = int(_model.predict(feature_frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GeoSea inference failed: {exc}") from exc

    return {
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "probability": probability,
        "prediction": prediction,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "environment": values_dict,
        "environmental_source": environmental["source"],
    }
