import json
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_DIR = Path(__file__).resolve().parent / "seaweed_model"
MODEL_PATH = MODEL_DIR / "best_geosea_efficientnet_b0_v2.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"

DEFAULT_CLASS_NAMES = [
    "Caulerpa_racemosa",
    "Caulerpa_serrulata",
    "Colpomenia_sinuosa",
    "Dictyosphaeria_cavernosa",
    "Dictyota_dichotoma",
    "Halimeda_macroloba",
    "Halimeda_tuna",
    "Hormophysa_cuneiformis",
    "Hydroclathrus_clathratus",
    "Lobophora_variegata",
    "Turbinaria_ornata",
    "Ulva_lactuca",
]

_model = None
_model_preprocess_version = "v2"
_class_names: List[str] = []

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
        "habitat": "Commonly found on intertidal rocky shores and in littoral and sublittoral rock pools; it may also occur in other coastal habitats depending on local conditions.",
        "known_range": "Widespread across many temperate and tropical coasts; occurrence records do not imply presence on every coast."
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
    parts = raw_class.replace("-", "_").split("_")
    # All GeoSea classifier classes are Genus_species.
    genus = parts[0]
    species = " ".join(parts[1:]) if len(parts) > 1 else "sp."
    scientific_name = f"{genus} {species}".strip()
    if genus in {"Caulerpa", "Dictyosphaeria", "Halimeda", "Ulva"}:
        group = "Green"
    elif genus in {"Colpomenia", "Dictyota", "Hormophysa", "Hydroclathrus", "Lobophora", "Turbinaria"}:
        group = "Brown"
    else:
        group = "Macroalgae"
    return group, scientific_name

def ensure_model_and_classes():
    global _class_names, MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
    else:
        _class_names = DEFAULT_CLASS_NAMES.copy()
        with open(CLASSES_PATH, "w", encoding="utf-8") as f:
            json.dump(_class_names, f, indent=2)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Seaweed classifier model not found. Copy best_geosea_efficientnet_b0_v2.pth "
            "into backend/seaweed_model/."
        )

def load_seaweed_model():
    global _model, _model_preprocess_version
    if _model is not None:
        return _model
    ensure_model_and_classes()
    _model_preprocess_version = "v2" if MODEL_PATH.name.endswith("_v2.pth") else "v1"
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(_class_names))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _model = model
    return _model

def get_bio_metadata(scientific_name: str, group: str) -> Dict[str, str]:
    if scientific_name in SPECIES_BIO_METADATA:
        return SPECIES_BIO_METADATA[scientific_name]
    genus = scientific_name.split()[0]
    return {
        "common_name": f"{group} Seaweed ({genus})",
        "group": f"{group} Alga",
        "phylum": "Chlorophyta" if group == "Green" else "Ochrophyta (Phaeophyceae)",
        "habitat": "Marine coastal waters, rocky reefs, or lagoon substrates.",
        "known_range": "Information unavailable in the local classifier registry.",
    }

def classify_seaweed_image(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    model = load_seaweed_model()
    if _model_preprocess_version == "v2":
        # Match V2 training/evaluation: resize shortest side while preserving
        # aspect ratio, then take a deterministic 224x224 center crop.
        resize_and_crop = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        # V1 was trained with direct square resizing; keep compatibility if the
        # legacy model filename is used.
        resize_and_crop = [transforms.Resize((224, 224))]

    preprocess = transforms.Compose([
        *resize_and_crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        preds = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_indices = np.argsort(preds)[::-1][:min(top_k, len(_class_names))]
    candidates = []
    for idx in top_indices:
        raw_name = _class_names[int(idx)]
        group, sciname = clean_class_name(raw_name)
        score = float(preds[int(idx)])
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
            "known_range": bio.get("known_range", "Information unavailable."),
        })
    top_candidate = candidates[0] if candidates else None
    is_uncertain = top_candidate is None or top_candidate["confidence_score"] < 0.25
    uncertainty_note = None
    if top_candidate and is_uncertain:
        uncertainty_note = (
            f"Low model confidence ({top_candidate['percentage']}%). The image does not strongly match "
            "any of the 12 seaweed species in the current GeoSea training set. "
            "The image may show a different species or an appearance not well represented in the dataset."
        )
    return {
        "model_name": "EfficientNet-B0 V2 (12-Class GeoSea Seaweed Classifier)",
        "model_architecture": "EfficientNet-B0 transfer learning",
        "model_classes_count": len(_class_names),
        "is_uncertain": is_uncertain,
        "uncertainty_note": uncertainty_note,
        "top_candidate": top_candidate,
        "alternative_candidates": candidates[1:],
        "classifier_note": (
            "The classifier predicts one of the 12 supported GeoSea species. "
            "Scores are Softmax confidence values across this closed set and are not calibrated ecological probabilities."
        ),
    }
