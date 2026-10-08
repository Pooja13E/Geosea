GeoSea production model artifact
================================

Production model:
  geosea_rf_model.joblib

Metadata:
  geosea_rf_metadata.json

Model type: RandomForestClassifier
Training features (in this exact order):
  temperature, salinity, chlorophyll, nitrate, phosphate, pH, PAR, kdpar

The Random Forest is used directly for inference; no StandardScaler is required.
The model is kept on the FastAPI backend and is never loaded by the React app.

The metadata records the 5-fold spatial cross-validation results and training
sample counts. CV metrics are evaluation results, not an independent test score.
