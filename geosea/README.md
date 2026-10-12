# GeoSea — Marine Habitat Suitability & Environmental Analysis

GeoSea is a React/Vite + FastAPI decision-support dashboard built around the real
`GeoSea_8_Variables.csv` dataset (336,924 records). It combines occurrence exploration,
environmental analysis, occurrence density, habitat-suitability prediction, and a
dedicated seaweed image-identification workflow.

## Main dashboard

- **Observed Occurrence Distribution** — presence and pseudo-absence observations with responsive aggregation and fixed-size markers.
- **Species Explorer** — searchable source-name selection plus occurrence profiles.
- **Environmental Conditions** — observed spatially aggregated layers for temperature, salinity, chlorophyll-a, nitrate, phosphate, pH, PAR, and kdPAR.
- **Occurrence Density** — spatial concentration of loaded occurrence records.
- **Habitat Suitability Prediction** — live coordinate prediction through FastAPI → Bio-ORACLE → the saved Random Forest.
- **Map / Satellite** base-map styles and selected-location analysis.

### Species-count terminology

The dataset contains **10,269 distinct source name strings**, but that is not the same as 10,269 species. Many entries contain author citations, varieties/subspecies, genus-level names, or higher-rank taxonomic labels.

For the headline dashboard count, GeoSea now uses a conservative **canonical binomial-name count: 5,615**. This removes author-string variants and non-binomial labels from the headline species count without changing the underlying occurrence rows or ML training data.

This is **name normalization, not authoritative synonym resolution**. The Species Explorer and map labels now use the same conservative binomial normalization at CSV-load time, so author citations and infraspecific suffixes do not create duplicate search entries. The source `scientific_name` column is preserved in the CSV; no source taxon names are overwritten. A final accepted-species count should be produced later using an authoritative taxonomic backbone such as WoRMS or GBIF and a reviewed synonym mapping.

## Seaweed Identification

The **Seaweed Identification** page is a focused three-step workflow inside the existing GeoSea dashboard shell:

1. **Step 1 — Image identification**
   - Upload a JPG, PNG, or WebP seaweed image.
   - Run the local **EfficientNet-B0 V2** 12-class classifier.
   - Show the predicted species and **visual confidence**.

2. **Step 2 — Observation location**
   - Reuse the same GeoSea occurrence map as the main dashboard.
   - Click the actual specimen observation location.

3. **Step 3 — GeoSea environmental evidence**
   - Retrieve the eight Bio-ORACLE baseline environmental variables for the selected coordinate.
   - Compare them with GeoSea presence-record environmental distributions for the predicted species.
   - Show **environmental consistency** and High / Medium / Low bands for the overall score and each variable.

The workflow intentionally ends with environmental evidence. It does not add a second distribution map, geographic-distance score, or separate WoRMS/GBIF page. The image upload endpoint is `POST /analyze-image`; the location evidence endpoint is `POST /species-verification`. Both routes are registered through the FastAPI analysis router.

### Confidence terminology

**Visual confidence** is the classifier's Softmax score within the closed 12-class image model.

**Environmental consistency** is an empirical 0–100 comparison with GeoSea occurrence-record conditions. It is **not** a probability of species identity and not a habitat-suitability probability.

Consistency bands are:
- **High:** 80–100
- **Medium:** 50–79
- **Low:** 0–49

## Production habitat-suitability model

The selected production model is **Random Forest**. Its eight features are:

1. temperature
2. salinity
3. chlorophyll
4. nitrate
5. phosphate
6. pH
7. PAR
8. kdpar

No StandardScaler is used for this Random Forest production pipeline.

The displayed habitat score is a model positive-class/suitability score and should not be interpreted as a calibrated ecological probability. Bio-ORACLE values are baseline climatology values, not real-time observations.

## Running the backend

From the project root:

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Use **`app:app`**, not `main:app`; the backend entry point is `backend/app.py`.

Health check:

```text
http://127.0.0.1:8000/health
```

## Running the frontend

From the project root:

```bash
npm install
npm run dev
```

The frontend normally runs at `http://localhost:5173`. The default backend URL is
`http://127.0.0.1:8000`; change it with `VITE_API_URL` in a local `.env` if needed.

## Bio-ORACLE requirement

Live prediction and environmental verification require backend internet access to Bio-ORACLE ERDDAP. If the service is unavailable, GeoSea reports the lookup as unavailable rather than inventing or substituting a value.

## Project structure

```text
GeoSea_CLEAN_FINAL/
├── public/data/GeoSea_8_Variables.csv
├── backend/
│   ├── app.py
│   ├── analysis.py
│   ├── seaweed_service.py
│   ├── bio_oracle.py
│   ├── remote_sensing.py
│   ├── requirements.txt
│   ├── .env
│   ├── models/
│   │   ├── geosea_rf_model.joblib
│   │   └── geosea_rf_metadata.json
│   └── seaweed_model/
│       ├── best_geosea_efficientnet_b0_v2.pth
│       ├── class_names.json
│       └── README.txt
├── src/
│   ├── components/
│   │   ├── analysis/
│   │   ├── layout/
│   │   ├── map/
│   │   └── sidebar/
│   ├── data/
│   ├── services/
│   ├── types/
│   └── App.tsx
├── package.json
├── vite.config.ts
└── README.md
```

## Important data/model note

The current 12-class image classifier is independent of the 5,615 canonical-name headline count. Changing the dashboard's species-count display does **not** retrain or change either the Random Forest or the EfficientNet model. The underlying occurrence CSV is preserved.

The `.env` file is intentionally retained in this development ZIP as requested. Do not commit real credentials or secrets to a public repository.

### Start both dashboard and backend (Windows)

Run `START_GEOSEA.bat` from the extracted project root. It opens the API in a separate terminal and then starts Vite. Keep both windows open. The API health check is `http://127.0.0.1:8000/health`; the dashboard is `http://localhost:5173`.

If image identification says it cannot connect to the backend, check the backend terminal first. Start the API manually from the `backend` folder with `py -m uvicorn app:app --reload --host 127.0.0.1 --port 8000`. Install backend dependencies in that folder with `py -m pip install -r requirements.txt` if imports are missing.

Scientific names shown in the dashboard are conservatively normalized for consistent formatting. This does not merge synonyms or overwrite the original CSV taxon labels.
