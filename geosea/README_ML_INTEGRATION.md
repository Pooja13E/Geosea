# GeoSea ML Integration

## Production model

GeoSea currently uses the **Random Forest** production model selected from the
same 5-fold spatial cross-validation setup used for model comparison.

CV ROC-AUC: **0.9431 ± 0.0095**.

The production artifact is:

```text
backend/models/geosea_rf_model.joblib
```

Metadata:

```text
backend/models/geosea_rf_metadata.json
```

The model was trained with these eight features in this exact order:
`temperature, salinity, chlorophyll, nitrate, phosphate, pH, PAR, kdpar`.

## Current prediction architecture

1. User clicks a marine coordinate in the React dashboard.
2. FastAPI receives latitude/longitude.
3. FastAPI retrieves the eight Bio-ORACLE v3.0 baseline variables from ERDDAP.
4. FastAPI validates the environmental vector and feature order.
5. FastAPI runs the saved Random Forest with `predict_proba` and `predict`.
6. The frontend displays the positive-class model score, suitability band, environmental values, and provenance.

No StandardScaler or PyTorch model is used in the production inference path.

## Important interpretation

The displayed score is a **Random Forest positive-class score / model suitability score**.
It is not a calibrated ecological probability.

The environmental values are **Bio-ORACLE baseline climatology values**, not current real-time measurements.

## Bio-ORACLE layers

- `thetao_baseline_2000_2019_depthsurf` / `thetao_mean`
- `so_baseline_2000_2019_depthsurf` / `so_mean`
- `chl_baseline_2000_2018_depthsurf` / `chl_mean`
- `no3_baseline_2000_2018_depthsurf` / `no3_mean`
- `po4_baseline_2000_2018_depthsurf` / `po4_mean`
- `ph_baseline_2000_2018_depthsurf` / `ph_mean`
- `par_mean_baseline_2000_2020_depthsurf` / `par_mean_mean`
- `kdpar_mean_baseline_2000_2020_depthsurf` / `kdpar_mean_mean`

The service uses the baseline time slice `2000-01-01T00:00:00Z` and the Bio-ORACLE 0.05° grid.

## Run

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The health endpoint is:

```text
http://127.0.0.1:8000/health
```

The frontend can be started separately with:

```bash
npm install
npm run dev
```
