import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PIL import Image

try:
    from .seaweed_service import (
        classify_seaweed_image,
        get_species_biodiversity_data,
        ensure_model_and_classes
    )
except ImportError:
    from seaweed_service import (
        classify_seaweed_image,
        get_species_biodiversity_data,
        ensure_model_and_classes
    )

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

class OccurrenceRecord(BaseModel):
    latitude: float
    longitude: float
    country: str
    event_date: str
    dataset: str
    basis_of_record: str

class SeaweedAnalysisResponse(BaseModel):
    model_name: str
    model_architecture: str
    model_classes_count: int
    is_uncertain: bool
    uncertainty_note: Optional[str] = None
    top_candidate: Optional[SpeciesCandidate] = None
    alternative_candidates: List[SpeciesCandidate] = []
    worms: Dict[str, Any] = {}
    gbif: Dict[str, Any] = {}
    occurrences: List[OccurrenceRecord] = []
    total_documented_occurrences: int = 0
    classifier_note: str
    scientific_advice_note: str

@router.post("/analyze-image", response_model=SeaweedAnalysisResponse)
@router.post("/analyze-seaweed", response_model=SeaweedAnalysisResponse)
async def analyze_seaweed(
    file: UploadFile = File(...),
    source_type: str = Form("underwater")
):
    """
    Identifies marine macroalgae species using the 53-species seaweed classifier,
    and queries WoRMS and GBIF for taxonomy and real documented distribution.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a valid image file (JPEG, PNG, or WebP)."
        )
    
    contents = await file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image format: {e}")
        
    try:
        classification = classify_seaweed_image(image, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seaweed inference failed: {e}")
        
    top_candidate = classification["top_candidate"]
    bio_data: Dict[str, Any] = {}
    
    # If confidence is not completely rejected, fetch WoRMS and GBIF data for top candidate
    if top_candidate:
        try:
            bio_data = get_species_biodiversity_data(top_candidate["scientific_name"])
        except Exception as e:
            print(f"[BioData Error] Could not retrieve distribution: {e}")
            bio_data = {"worms": {}, "gbif": {}}
            
    gbif_info = bio_data.get("gbif", {})
    worms_info = bio_data.get("worms", {})
    
    occurrences_raw = gbif_info.get("occurrences_sample", [])
    occurrences = [OccurrenceRecord(**rec) for rec in occurrences_raw]
    
    return SeaweedAnalysisResponse(
        model_name=classification["model_name"],
        model_architecture=classification["model_architecture"],
        model_classes_count=classification["model_classes_count"],
        is_uncertain=classification["is_uncertain"],
        uncertainty_note=classification["uncertainty_note"],
        top_candidate=SpeciesCandidate(**top_candidate) if top_candidate else None,
        alternative_candidates=[SpeciesCandidate(**c) for c in classification["alternative_candidates"]],
        worms=worms_info,
        gbif=gbif_info,
        occurrences=occurrences,
        total_documented_occurrences=gbif_info.get("total_occurrences_documented", 0),
        classifier_note=classification["classifier_note"],
        scientific_advice_note=(
            "Automated visual identification is a machine-learning estimate based on morphology. "
            "Microscopic or molecular validation is required for definitive taxonomic confirmation. "
            "Documented occurrence points represent recorded observations, which are distinct from the full natural range."
        )
    )

@router.get("/species-distribution")
def get_distribution(scientific_name: str = Query(..., description="Scientific name to look up")):
    """Returns WoRMS and GBIF distribution records for a selected species name."""
    try:
        data = get_species_biodiversity_data(scientific_name)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch species distribution: {e}")
