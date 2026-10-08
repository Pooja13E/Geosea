# GeoSea Dashboard Update

This version keeps the real GeoSea dataset and the live Bio-ORACLE prediction
workflow, with **Random Forest** as the production model.

## Current dashboard additions

1. **Species Explorer** — searchable species profile with occurrence count, geographic coverage, and ranges for all eight environmental variables.
2. **Occurrence Density** — spatial concentration of real loaded occurrence records, with click-through local statistics.
3. **Environmental Profile** — the live prediction panel shows the eight environmental values used by the model.
4. **Dataset Summary Banner** — a compact horizontal banner with total occurrences, presence, pseudo-absence, species, and environmental-variable count.
5. **Environmental Variable Explorer** — all eight observed environmental layers remain available.

## Removed

- Suitability Heatmap
- Compare Two Locations

## Production ML

Random Forest is used for live inference. The model artifact is kept in
`backend/models/geosea_rf_model.joblib` and is loaded only by FastAPI.
No scaler is required.

## Run

### Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend

Open a second terminal at the project root:

```bash
npm install
npm run dev
```
