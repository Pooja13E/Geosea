# GeoSea — Marine Habitat Suitability & Environmental Analysis

GeoSea is a React/Vite + FastAPI decision-support dashboard built around the real
`GeoSea_8_Variables.csv` dataset (336,924 records). It combines occurrence
exploration, environmental analysis, species exploration, occurrence density,
and live habitat-suitability prediction.

## Dashboard features

- **Observed Occurrence Distribution** — presence and pseudo-absence observations with responsive aggregation and fixed-size markers.
- **Species Explorer** — searchable species selection plus an occurrence profile showing record count, geographic coverage, and ranges for all 8 environmental variables.
- **Environmental Conditions** — observed spatially aggregated layers for temperature, salinity, chlorophyll, nitrate, phosphate, pH, PAR, and kdPAR.
- **Occurrence Density** — spatial concentration of loaded occurrence records, with local record/presence/pseudo-absence/species details.
- **Habitat Suitability Prediction** — live coordinate prediction through FastAPI → Bio-ORACLE v3.0 → saved Random Forest.
- **Environmental Map Overlays** — integrated visual layers for NASA GIBS True-Colour satellite imagery, NASA Ocean Chlorophyll-a, NASA Sea Surface Temperature (SST), and official GEBCO WMS bathymetry with opacity and date sliders.
- **Seaweed Identification & Distribution System** — upload marine macroalgae imagery for automated species identification across 53 seaweed taxa (Brown, Green, and Red seaweeds) using an EfficientNet-B0 classifier, integrated with live WoRMS taxonomic validation and GBIF global occurrence mapping.
- **Dataset summary banner** — total occurrences, presence, pseudo-absence, species, and environmental-variable count.
- **Base map / satellite** map styles and explicit pan controls.

The dashboard does **not** use a fabricated suitability heatmap or a simulated comparison tool.
The ML prediction is a point-based live inference workflow using real Bio-ORACLE environmental
lookups and the saved production model. Seaweed weights (~48 MB) are automatically downloaded on first inference into a local git-ignored cache.

## Production model

The selected production model is **Random Forest**.

5-fold spatial cross-validation:

- ROC-AUC: **0.9431 ± 0.0095**
- Accuracy: **0.8193 ± 0.0433**
- Precision: **0.9606 ± 0.0145**
- Recall: **0.6479 ± 0.1085**
- F1: **0.7690 ± 0.0868**

The final artifact is trained on the complete dataset after the spatial-CV evaluation.
These CV metrics are evaluation results and are **not** an independent test score.

The model uses these eight features in this exact order:

1. temperature
2. salinity
3. chlorophyll
4. nitrate
5. phosphate
6. pH
7. PAR
8. kdpar

No StandardScaler is used for the Random Forest production pipeline.

## Prediction flow

1. Select **Habitat Suitability Prediction**.
2. Click a marine coordinate.
3. FastAPI sends the coordinate to Bio-ORACLE ERDDAP.
4. Bio-ORACLE returns the eight baseline environmental variables.
5. FastAPI passes those eight variables to the saved Random Forest.
6. The Random Forest returns its positive-class score and classification.
7. GeoSea displays the model suitability score, suitability class, environmental profile, and provenance.

The displayed 0–100% value is a **model suitability score / positive-class score**. It should not be interpreted as a calibrated ecological probability or as a percentage chance of habitat occurrence.

Bio-ORACLE values are **baseline climatology values, not real-time observations**.

## Running the backend

Open a terminal in the project root:

```powershell
cd backend
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start FastAPI:

```powershell
uvicorn app:app --reload
```

**Use `app:app`, not `main:app`.** The backend entry file is `backend/app.py`.

Check:

```text
http://127.0.0.1:8000/health
```

Expected when the model is loaded:

```json
{
  "status": "ok",
  "model": "random_forest",
  "model_label": "Random Forest",
  "ready": true
}
```

### Backend CORS

The default frontend origins are `http://localhost:5173` and `http://127.0.0.1:5173`.
To use a different frontend origin, set `GEOSEA_CORS_ORIGINS` to a comma-separated list.

## Running the frontend

From the project root:

```bash
npm install
npm run dev
```

Then open the Vite URL, normally `http://localhost:5173`.

The frontend defaults to `http://127.0.0.1:8000` for prediction requests. To change it,
create `.env` from `.env.example` and set `VITE_API_URL`.

## Bio-ORACLE network requirement

Prediction mode requires internet access from the backend machine to Bio-ORACLE ERDDAP.
If Bio-ORACLE is unavailable, GeoSea reports the live prediction as unavailable rather
than substituting a nearest observed GeoSea record.

## Project structure

```text
geosea/
├── public/data/GeoSea_8_Variables.csv
├── backend/
│   ├── app.py
│   ├── analysis.py
│   ├── seaweed_service.py
│   ├── bio_oracle.py
│   ├── requirements.txt
│   └── models/
│       ├── geosea_rf_model.joblib
│       └── geosea_rf_metadata.json
├── src/
│   ├── components/
│   │   ├── analysis/
│   │   ├── layout/
│   │   ├── map/
│   │   └── sidebar/
│   ├── overlays/
│   ├── data/
│   ├── services/
│   ├── types/
│   └── App.tsx
├── package.json
└── README.md
```
