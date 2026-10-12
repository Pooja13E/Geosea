import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

try:
    from .bio_oracle import BioOracleError, get_environment
except ImportError:
    from bio_oracle import BioOracleError, get_environment

try:
    from .scientific_names import canonical_binomial_name
except ImportError:
    from scientific_names import canonical_binomial_name

router = APIRouter()

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

class SpeciesCandidate(BaseModel):
    raw_class: str
    scientific_name: str
    algae_group: str
    phylum: str
    common_name: str
    confidence_score: float
    percentage: float
    habitat_summary: str
    known_range: str

class SeaweedAnalysisResponse(BaseModel):
    model_name: str
    model_architecture: str
    model_classes_count: int
    is_uncertain: bool
    uncertainty_note: Optional[str] = None
    top_candidate: Optional[SpeciesCandidate] = None
    alternative_candidates: List[SpeciesCandidate] = []
    classifier_note: str
    scientific_advice_note: str

_GEOSEA_DATA_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "GeoSea_8_Variables.csv"
_GEOSEA_VARIABLES = {
    "temperature": "Temperature",
    "salinity": "Salinity",
    "chlorophyll": "Chlorophyll-a",
    "nitrate": "Nitrate",
    "phosphate": "Phosphate",
    "pH": "pH",
    "PAR": "Photosynthetically Active Radiation (PAR)",
    "kdpar": "Light Attenuation (KdPAR)",
}
_geosea_presence = None

@router.post("/analyze-image", response_model=SeaweedAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    source_type: str = Form("underwater"),
):
    """Classify an uploaded seaweed image with the local EfficientNet-B0 V2 model."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WebP image.")

    contents = await file.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.") from exc

    # Import the heavyweight image stack only when this feature is requested.
    # A missing/broken optional torch/torchvision installation must not prevent
    # the whole FastAPI application (prediction + health + map) from starting.
    try:
        try:
            from .seaweed_service import classify_seaweed_image
        except ImportError:
            from seaweed_service import classify_seaweed_image
        result = classify_seaweed_image(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Seaweed image classifier could not start. Check that backend requirements are installed "
                "and the model file exists in backend/seaweed_model/. Details: " + str(exc)
            ),
        ) from exc

    for candidate in [result.get("top_candidate"), *(result.get("alternative_candidates") or [])]:
        if isinstance(candidate, dict):
            canonical = canonical_binomial_name(candidate.get("scientific_name", ""))
            if canonical:
                candidate["scientific_name"] = canonical

    # Keep this field explicit for the frontend and remind users that a closed-set
    # image prediction should be checked against the observation context.
    result["scientific_advice_note"] = (
        "This is a visual prediction among the 12 supported GeoSea classes. "
        "Review the confidence and compare the observation location's environmental evidence; "
        "the score is not a calibrated ecological probability."
    )
    return result

def _load_geosea_presence():
    global _geosea_presence
    if _geosea_presence is None:
        if not _GEOSEA_DATA_PATH.exists():
            raise FileNotFoundError(f"GeoSea dataset not found: {_GEOSEA_DATA_PATH}")
        cols = ["scientific_name", "latitude", "longitude", "presence", *list(_GEOSEA_VARIABLES)]
        df = pd.read_csv(_GEOSEA_DATA_PATH, usecols=cols)
        df["presence"] = pd.to_numeric(df["presence"], errors="coerce")
        df = df[df["presence"] == 1].copy()
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Use exactly the same conservative binomial formatting as the frontend.
        # Genus-only, higher-rank, and identifier labels are not species evidence.
        df["_binomial"] = df["scientific_name"].map(canonical_binomial_name)
        df = df[df["_binomial"].notna()].copy()
        df["_binomial_key"] = df["_binomial"].str.casefold()
        _geosea_presence = df
    return _geosea_presence

# Species verification: visual classification + GeoSea environmental/geographic
# evidence. This is deliberately a consistency check, not a calibrated species
# probability and it does not modify the production habitat-suitability model.
# -----------------------------------------------------------------------------

_VERIFICATION_VARIABLE_LABELS = {
    "temperature": ("Temperature", "°C"),
    "salinity": ("Salinity", "PSU"),
    "chlorophyll": ("Chlorophyll-a", ""),
    "nitrate": ("Nitrate", "µmol/L"),
    "phosphate": ("Phosphate", "µmol/L"),
    "pH": ("pH", ""),
    "PAR": ("PAR", ""),
    "kdpar": ("KdPAR", ""),
}

def _empirical_variable_consistency(value: float, species_values: pd.Series) -> tuple[float, dict]:
    vals = pd.to_numeric(species_values, errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5 or not np.isfinite(value):
        return 50.0, {"status": "insufficient_data"}

    q01, q10, q25, q50, q75, q90, q99 = np.percentile(vals, [1, 10, 25, 50, 75, 90, 99])

    # A descriptive 0-100 consistency score based on where the current
    # Bio-ORACLE value sits in the species' GeoSea presence distribution.
    # It is intentionally not presented as a probability or suitability score.
    if q25 <= value <= q75:
        score = 100.0
    elif q10 <= value < q25:
        score = 85.0 + 15.0 * ((value - q10) / max(q25 - q10, 1e-12))
    elif q75 < value <= q90:
        score = 85.0 + 15.0 * ((q90 - value) / max(q90 - q75, 1e-12))
    elif q01 <= value < q10:
        score = 60.0 + 25.0 * ((value - q01) / max(q10 - q01, 1e-12))
    elif q90 < value <= q99:
        score = 60.0 + 25.0 * ((q99 - value) / max(q99 - q90, 1e-12))
    else:
        score = 30.0 * (1.0 - min(1.0, abs(value - q50) / max(abs(q50 - q01), abs(q99 - q50), 1e-12)))

    percentile = float(np.mean(vals <= value) * 100.0)
    return float(np.clip(score, 0.0, 100.0)), {
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "percentile": round(percentile, 1),
        "sample_count": int(vals.size),
        "status": "ok",
    }

def _verification_environment(df: pd.DataFrame, species_name: str, environment: dict) -> dict:
    rows = df[df["_binomial_key"] == species_name.casefold()].copy()
    if rows.empty:
        raise HTTPException(status_code=404, detail="No GeoSea presence records were found for this species.")

    details = []
    scores = []
    current_values = environment["values"]
    for key in _GEOSEA_VARIABLES:
        value = float(current_values[key])
        score, profile = _empirical_variable_consistency(value, rows[key])
        scores.append(score)
        label, unit = _VERIFICATION_VARIABLE_LABELS[key]
        details.append({
            "key": key,
            "label": label,
            "unit": unit,
            "current_value": value,
            "consistency_score": round(score, 1),
            **profile,
        })

    overall = float(np.mean(scores)) if scores else 0.0
    if overall >= 80:
        level = "high"
        label = "High consistency"
    elif overall >= 50:
        level = "moderate"
        label = "Moderate consistency"
    else:
        level = "low"
        label = "Low consistency"

    return {
        "score": round(overall, 1),
        "level": level,
        "label": label,
        "variables": details,
        "presence_count": int(len(rows)),
    }

@router.post("/species-verification")
def verify_species_at_location(
    scientific_name: str = Query(...),
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    """
    Compare the selected coordinate's Bio-ORACLE baseline environment with
    GeoSea presence-record environmental distributions for the image-classified species.

    The environmental score is a descriptive empirical consistency measure, not
    a calibrated species-identification probability and not a replacement for
    the production habitat-suitability model.
    """
    name = canonical_binomial_name(scientific_name)
    if not name:
        raise HTTPException(status_code=400, detail="Please provide a valid genus and species name.")

    try:
        df = _load_geosea_presence()
        environment = get_environment(latitude, longitude)
    except BioOracleError as exc:
        raise HTTPException(status_code=502, detail=f"Bio-ORACLE environmental lookup failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load GeoSea verification data: {exc}") from exc

    env_result = _verification_environment(df, name, environment)
    return {
        "scientific_name": name,
        "location": {"latitude": latitude, "longitude": longitude},
        "geosea_presence_count": int(env_result["presence_count"]),
        "environmental_consistency": env_result,
        "environment": environment["values"],
        "environmental_source": environment["source"],
        "method_note": (
            "Environmental consistency compares the selected Bio-ORACLE baseline values with the "
            "distribution of values in GeoSea presence records for the predicted species. It is a "
            "descriptive consistency measure, not a calibrated species-identification probability, "
            "and it does not change the existing habitat-suitability model."
        ),
    }
