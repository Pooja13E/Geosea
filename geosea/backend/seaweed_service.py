import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import numpy as np

# Ensure PyTorch backend for Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

MODEL_DIR = Path(__file__).resolve().parent / "seaweed_model"
MODEL_PATH = MODEL_DIR / "best_seaweed_aug_model.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"

HF_MODEL_URL = "https://huggingface.co/spaces/Mahmoudsami/Seaweed_Identifier/resolve/main/best_seaweed_aug_model.keras"
HF_CLASSES_URL = "https://huggingface.co/spaces/Mahmoudsami/Seaweed_Identifier/raw/main/class_names.json"

_model = None
_class_names: List[str] = []
_biodiversity_cache: Dict[str, Dict[str, Any]] = {}

# Typo fixes in original training set labels
TYPO_CORRECTIONS = {
    "sinousa": "sinuosa",
    "Cystoseria": "Cystoseira",
    "Chaetomprpha": "Chaetomorpha",
    "Galaxura": "Galaxaura",
}

# Curated biological metadata for the supported genera / species
SPECIES_BIO_METADATA = {
    "Colpomenia sinuosa": {
        "common_name": "Sinuous bubble alga / Oyster thief",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Intertidal and shallow subtidal rocky reefs, tide pools, and epiphytic on seagrasses.",
        "known_range": "Widely distributed in tropical, subtropical, and warm temperate waters worldwide (Indo-Pacific, Mediterranean, Caribbean)."
    },
    "Cystoseira": {
        "common_name": "Bushy bladder alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Rocky intertidal and subtidal reefs forming canopy-forming seaweed forests.",
        "known_range": "Mediterranean Sea, Eastern Atlantic, and Indo-Pacific tropical to warm-temperate coasts."
    },
    "Dictyota dichotoma": {
        "common_name": "Forked ribbon alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Shallow reefs, lagoons, tide pools, growing on rocks, corals, and shells.",
        "known_range": "Cosmopolitan in temperate and tropical oceans including Atlantic, Mediterranean, and Indo-Pacific."
    },
    "Hormophysa cuneiformis": {
        "common_name": "Wedge-leaf bladder alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Intertidal rocky platforms, fringing coral reef flats, and shallow lagoons.",
        "known_range": "Tropical Indo-Pacific and Red Sea coasts."
    },
    "Hydroclathrus clathratus": {
        "common_name": "Lattice / Perforated bubble alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Shallow rocky shores, reef crests, and sandy-rocky lagoons.",
        "known_range": "Widespread in tropical and subtropical waters worldwide."
    },
    "Lobophora variegata": {
        "common_name": "Variegated fan alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Exposed rocky reef crests, deep fore-reef slopes, and coral rubble.",
        "known_range": "Pantropical and subtropical coastal reefs worldwide."
    },
    "Padina pavonica": {
        "common_name": "Peacock's tail alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Calcareous deposits on shallow rocky shores, pools, and calm sandy-rocky bays.",
        "known_range": "Mediterranean Sea, Atlantic Ocean, and warm temperate coasts."
    },
    "Padina boergesenii": {
        "common_name": "Tropical funnel weed",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Intertidal reef flats and shallow subtidal zones with moderate wave action.",
        "known_range": "Tropical Atlantic, Caribbean, and Indo-Pacific."
    },
    "Sargassum": {
        "common_name": "Sargasso weed / Gulfweed",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Rocky subtidal reefs, reef flats; some species form massive pelagic drift mats.",
        "known_range": "Pantropical and temperate oceans globally."
    },
    "Turbinaria ornata": {
        "common_name": "Ornate turban alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "High-energy outer reef crests, rocky intertidal shores, and coral reef flats.",
        "known_range": "Tropical Indo-Pacific, Red Sea, and Hawaiian Archipelago."
    },
    "Turbinaria turbinata": {
        "common_name": "Turban alga",
        "group": "Brown Alga",
        "phylum": "Ochrophyta (Phaeophyceae)",
        "habitat": "Shallow rocky reefs, lagoons, and hard substrates in clear tropical water.",
        "known_range": "Tropical Atlantic, Caribbean, and Gulf of Mexico."
    },
    "Caulerpa racemosa": {
        "common_name": "Sea grapes / Green caviar",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Shallow reef flats, coral rubble, sandy mudflats, and seagrass beds.",
        "known_range": "Tropical and subtropical coasts worldwide; invasive in parts of the Mediterranean."
    },
    "Caulerpa serrulata": {
        "common_name": "Sawtooth sea grape",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Intertidal reef flats, sandy pockets on reefs, and shallow lagoons.",
        "known_range": "Tropical Indo-Pacific and Red Sea."
    },
    "Chaetomorpha linum": {
        "common_name": "Spaghetti algae / Mermaid's hair",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Estuaries, coastal lagoons, shallow tide pools, often unattached entangled mats.",
        "known_range": "Cosmopolitan in marine and brackish waters worldwide."
    },
    "Codium decorticatum": {
        "common_name": "Dead man's fingers alga",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Lower intertidal to subtidal rocks, boulders, and harbor pilings.",
        "known_range": "Warm-temperate and tropical Atlantic Ocean and Mediterranean Sea."
    },
    "Dictyosphaeria cavernosa": {
        "common_name": "Green bubble alga",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Intertidal and shallow subtidal reef flats, coral rubble, and lagoon patches.",
        "known_range": "Pantropical in the Pacific, Indian, and Atlantic oceans."
    },
    "Halimeda macroloba": {
        "common_name": "Broad-lobed cactus alga",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Calcareous green alga anchoring with bulbous holdfast in sandy or silty lagoon bottoms.",
        "known_range": "Indo-West Pacific tropical waters."
    },
    "Halimeda tuna": {
        "common_name": "Cactus alga / Money plant",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Shallow to deep reef slopes and rocky crevices, accumulating calcium carbonate.",
        "known_range": "Mediterranean Sea, Atlantic, and Indo-Pacific tropical to warm-temperate reefs."
    },
    "Ulva lactuca": {
        "common_name": "Sea lettuce",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Intertidal and upper subtidal rocky shores, salt marshes, and nutrient-rich coastal bays.",
        "known_range": "Cosmopolitan, found on coasts across all oceans worldwide."
    },
    "Valonia aegagropila": {
        "common_name": "Sailor's eyeballs alga",
        "group": "Green Alga",
        "phylum": "Chlorophyta",
        "habitat": "Shallow reef flats forming creeping green cushions on rocks and dead coral.",
        "known_range": "Tropical Indo-Pacific, Mediterranean, and Caribbean."
    },
    "Gracilaria gracilis": {
        "common_name": "Slender agar weed",
        "group": "Red Alga",
        "phylum": "Rhodophyta",
        "habitat": "Intertidal and shallow subtidal sand-covered rocks and sheltered estuaries.",
        "known_range": "North Atlantic, Mediterranean, and temperate African coasts."
    },
    "Jania rubens": {
        "common_name": "Red coralline weed",
        "group": "Red Alga",
        "phylum": "Rhodophyta",
        "habitat": "Calcareous articulated red alga growing epiphytically on larger seaweeds in tide pools.",
        "known_range": "Cosmopolitan in temperate and tropical oceans."
    },
    "Lithophyllum incrustans": {
        "common_name": "Encrusting coralline alga",
        "group": "Red Alga",
        "phylum": "Rhodophyta",
        "habitat": "Forms hard pink calcified crusts on wave-exposed rocky shores and tide pools.",
        "known_range": "Northeast Atlantic, Mediterranean, and British Isles."
    },
    "Digenea simplex": {
        "common_name": "Worm alga",
        "group": "Red Alga",
        "phylum": "Rhodophyta",
        "habitat": "Shallow reef flats, wave-swept rocky shores, and coral rubble.",
        "known_range": "Tropical and subtropical seas worldwide."
    }
}

def clean_class_name(raw_class: str) -> tuple[str, str]:
    """Parses raw class string into algae group and clean scientific name."""
    parts = raw_class.split("_")
    group = parts[0]
    genus = parts[1]
    species = " ".join(parts[2:]) if len(parts) > 2 else "sp."
    
    for typo, fix in TYPO_CORRECTIONS.items():
        if typo in genus:
            genus = genus.replace(typo, fix)
        if typo in species:
            species = species.replace(typo, fix)
            
    scientific_name = f"{genus} {species}".strip()
    return group, scientific_name

def ensure_model_and_classes():
    """Ensures model weights and class definitions exist locally."""
    global _class_names
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CLASSES_PATH.exists():
        urllib.request.urlretrieve(HF_CLASSES_URL, CLASSES_PATH)
        
    if not _class_names and CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
            
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)

def load_seaweed_model():
    """Loads the Keras 3 model with PyTorch backend."""
    global _model
    if _model is None:
        ensure_model_and_classes()
        _model = keras.models.load_model(str(MODEL_PATH))
    return _model

def get_bio_metadata(scientific_name: str, group: str) -> Dict[str, str]:
    """Retrieves curated biological context based on species or genus."""
    if scientific_name in SPECIES_BIO_METADATA:
        return SPECIES_BIO_METADATA[scientific_name]
        
    genus = scientific_name.split()[0]
    if genus in SPECIES_BIO_METADATA:
        return SPECIES_BIO_METADATA[genus]
        
    return {
        "common_name": f"{group} Seaweed ({genus})",
        "group": f"{group} Alga",
        "phylum": "Chlorophyta" if group == "Green" else ("Rhodophyta" if group == "Red" else "Ochrophyta (Phaeophyceae)"),
        "habitat": "Marine coastal waters, rocky reefs, or lagoon substrates.",
        "known_range": "Information unavailable in local registry; consult GBIF / WoRMS records."
    }

def classify_seaweed_image(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """Runs EfficientNet-B0 inference on uploaded seaweed image."""
    model = load_seaweed_model()
    ensure_model_and_classes()
    
    # Preprocess
    img = image.convert("RGB").resize((224, 224))
    x = np.array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]
    
    candidates = []
    for idx in top_indices:
        raw_name = _class_names[idx]
        group, sciname = clean_class_name(raw_name)
        score = float(preds[idx])
        bio = get_bio_metadata(sciname, group)
        
        candidates.append({
            "raw_class": raw_name,
            "scientific_name": sciname,
            "algae_group": bio.get("group", f"{group} Alga"),
            "phylum": bio.get("phylum", "Macroalgae"),
            "common_name": bio.get("common_name", sciname),
            "confidence_score": round(score, 4),
            "percentage": round(score * 100, 1),
            "habitat_summary": bio.get("habitat", "Marine coastal habitat."),
            "known_range": bio.get("known_range", "Information unavailable.")
        })
        
    top_candidate = candidates[0]
    # Check for uncertainty: 1/53 baseline is ~0.019. If top score < 0.18, confidence is low.
    is_uncertain = top_candidate["confidence_score"] < 0.18
    uncertainty_note = None
    if is_uncertain:
        uncertainty_note = (
            f"Low model confidence ({top_candidate['percentage']}%). The image does not strongly match "
            f"any of the 53 supported seaweed taxa. It may depict a non-seaweed subject or a species outside "
            f"the training set."
        )
        
    return {
        "model_name": "EfficientNet-B0 (43+ Species Seaweed Classifier, 53 classes)",
        "model_architecture": "Convolutional Neural Network (EfficientNet-B0 transfer learning)",
        "model_classes_count": len(_class_names),
        "is_uncertain": is_uncertain,
        "uncertainty_note": uncertainty_note,
        "top_candidate": top_candidate,
        "alternative_candidates": candidates[1:],
        "classifier_note": (
            "Model scores are Softmax confidence values across a closed set of 53 classes, "
            "not calibrated true probabilities. Whole-image classification cannot count individual plants."
        )
    }

def query_worms_api(scientific_name: str) -> Dict[str, Any]:
    """Queries World Register of Marine Species (WoRMS) REST API."""
    encoded_name = urllib.parse.quote(scientific_name)
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
    req = urllib.request.Request(url, headers={"User-Agent": "GeoSea-OceanResearch/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and len(data) > 0:
                rec = data[0]
                return {
                    "source": "World Register of Marine Species (WoRMS)",
                    "aphia_id": rec.get("AphiaID"),
                    "scientific_name": rec.get("scientificname"),
                    "valid_name": rec.get("valid_name"),
                    "authority": rec.get("valid_authority"),
                    "taxonomic_status": rec.get("status", "accepted"),
                    "rank": rec.get("rank"),
                    "taxonomy": {
                        "kingdom": rec.get("kingdom"),
                        "phylum": rec.get("phylum"),
                        "class": rec.get("class"),
                        "order": rec.get("order"),
                        "family": rec.get("family"),
                        "genus": rec.get("genus"),
                    },
                    "url": rec.get("url") or f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={rec.get('AphiaID')}",
                    "citation": rec.get("citation")
                }
    except Exception as e:
        print(f"[WoRMS API] Lookup failed for {scientific_name}: {e}")
        
    return {
        "source": "World Register of Marine Species (WoRMS)",
        "status": "Information unavailable from WoRMS",
        "url": f"https://www.marinespecies.org/aphia.php?p=taxlist&tName={encoded_name}"
    }

def query_gbif_api(scientific_name: str) -> Dict[str, Any]:
    """Queries GBIF Backbone Taxonomy and documented occurrences."""
    encoded_name = urllib.parse.quote(scientific_name)
    match_url = f"https://api.gbif.org/v1/species/match?name={encoded_name}"
    req_match = urllib.request.Request(match_url, headers={"User-Agent": "GeoSea-OceanResearch/1.0"})
    
    taxon_key = None
    scientific_match = None
    taxonomy_hierarchy = {}
    
    try:
        with urllib.request.urlopen(req_match, timeout=8) as resp:
            match_data = json.loads(resp.read().decode("utf-8"))
            taxon_key = match_data.get("usageKey")
            scientific_match = match_data.get("scientificName")
            taxonomy_hierarchy = {
                "kingdom": match_data.get("kingdom"),
                "phylum": match_data.get("phylum"),
                "class": match_data.get("class"),
                "order": match_data.get("order"),
                "family": match_data.get("family"),
                "genus": match_data.get("genus"),
                "species": match_data.get("species"),
            }
    except Exception as e:
        print(f"[GBIF API] Taxon match failed for {scientific_name}: {e}")
        
    occurrences = []
    total_count = 0
    
    if taxon_key:
        occ_url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={taxon_key}&hasCoordinate=true&limit=60"
        req_occ = urllib.request.Request(occ_url, headers={"User-Agent": "GeoSea-OceanResearch/1.0"})
        try:
            with urllib.request.urlopen(req_occ, timeout=10) as resp:
                occ_data = json.loads(resp.read().decode("utf-8"))
                total_count = occ_data.get("count", 0)
                for item in occ_data.get("results", []):
                    lat = item.get("decimalLatitude")
                    lon = item.get("decimalLongitude")
                    if lat is not None and lon is not None:
                        occurrences.append({
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "country": item.get("country", "Unknown"),
                            "event_date": item.get("eventDate", "Date unavailable"),
                            "dataset": item.get("datasetName", "GBIF Documented Record"),
                            "basis_of_record": item.get("basisOfRecord", "PreservedSpecimen/HumanObservation")
                        })
        except Exception as e:
            print(f"[GBIF API] Occurrence search failed for {scientific_name}: {e}")
            
    return {
        "source": "Global Biodiversity Information Facility (GBIF)",
        "taxon_key": taxon_key,
        "matched_scientific_name": scientific_match or scientific_name,
        "taxonomy": taxonomy_hierarchy,
        "total_occurrences_documented": total_count,
        "occurrences_sample": occurrences,
        "url": f"https://www.gbif.org/species/{taxon_key}" if taxon_key else f"https://www.gbif.org/species/search?q={encoded_name}",
        "data_limitation_note": (
            "Documented occurrence points represent verified recorded museum/field observations. "
            "Absence of records in a given area does not prove species absence. Documented points are distinct from the full natural range."
        )
    }

def get_species_biodiversity_data(scientific_name: str) -> Dict[str, Any]:
    """Queries and caches WoRMS and GBIF data for the species."""
    if scientific_name in _biodiversity_cache:
        return _biodiversity_cache[scientific_name]
        
    worms_data = query_worms_api(scientific_name)
    gbif_data = query_gbif_api(scientific_name)
    
    result = {
        "scientific_name": scientific_name,
        "worms": worms_data,
        "gbif": gbif_data,
    }
    
    _biodiversity_cache[scientific_name] = result
    return result
