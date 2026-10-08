import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "GeoSea_8_Variables.csv"

MODEL_DIR = "models"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "geosea_rf_model.joblib"
)

CV_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "rf_spatial_cv_results.csv"
)

CV_SUMMARY_PATH = os.path.join(
    MODEL_DIR,
    "rf_spatial_cv_summary.json"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "geosea_rf_metadata.json"
)

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

TARGET = "presence"
FOLD_COLUMN = "spatial_fold"

EXPECTED_FOLDS = [0, 1, 2, 3, 4]

PREDICTION_THRESHOLD = 0.5


# Random Forest configuration
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 14,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": 42,
}


# ============================================================
# SETUP
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("GEOSSEA RANDOM FOREST")
print("5-FOLD SPATIAL CROSS-VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(f"Dataset loaded: {len(df):,} rows")


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [
    TARGET,
    FOLD_COLUMN,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

missing_values = df[required_columns].isnull().sum()

if missing_values.sum() > 0:

    print("\nMissing values detected:")

    print(
        missing_values[
            missing_values > 0
        ]
    )

    raise ValueError(
        "Dataset contains missing values in required columns. "
        "Please clean the dataset before training."
    )

print("Missing-value check: PASSED")


# ============================================================
# 4. PREPARE FEATURES / TARGET / FOLDS
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()

folds = df[FOLD_COLUMN].copy()


# ============================================================
# 5. VALIDATE TARGET
# ============================================================

unique_targets = sorted(
    y.unique().tolist()
)

print(
    f"Target values: {unique_targets}"
)

if not set(unique_targets).issubset({0, 1}):

    raise ValueError(
        "Target column must contain only 0 and 1."
    )


# ============================================================
# 6. VALIDATE SPATIAL FOLDS
# ============================================================

unique_folds = sorted(
    folds.unique().tolist()
)

print(
    f"Spatial folds found: {unique_folds}"
)

if unique_folds != EXPECTED_FOLDS:

    raise ValueError(
        f"Expected spatial folds "
        f"{EXPECTED_FOLDS}, "
        f"but found {unique_folds}"
    )

print("Spatial-fold check: PASSED")


# ============================================================
# 7. DISPLAY DATA DISTRIBUTION
# ============================================================

print("\nDataset information:")
print("-" * 50)

print(
    f"Total samples       : {len(df):,}"
)

print(
    f"Presence samples    : {(y == 1).sum():,}"
)

print(
    f"Pseudo-absence      : {(y == 0).sum():,}"
)

print(
    f"Number of features  : {len(FEATURES)}"
)

print("\nFold distribution:")

for fold in EXPECTED_FOLDS:

    count = (folds == fold).sum()

    print(
        f"  Fold {fold}: {count:,} samples"
    )


# ============================================================
# 8. SPATIAL CROSS-VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("5-FOLD SPATIAL CROSS-VALIDATION")
print("=" * 70)


fold_results = []


for fold in EXPECTED_FOLDS:

    print("\n")
    print("-" * 70)
    print(f"VALIDATION FOLD {fold}")
    print("-" * 70)


    # --------------------------------------------------------
    # Train / validation masks
    # --------------------------------------------------------

    train_mask = folds != fold

    val_mask = folds == fold


    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]


    print(
        f"Training samples   : {len(X_train):,}"
    )

    print(
        f"Validation samples : {len(X_val):,}"
    )


    # --------------------------------------------------------
    # Create a fresh model for this fold
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(
        **RF_PARAMS
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    rf_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred_proba = (
        rf_model.predict_proba(X_val)[:, 1]
    )

    y_pred = (
        y_pred_proba >= PREDICTION_THRESHOLD
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_val,
        y_pred_proba
    )

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    fold_results.append({

        "fold": fold,

        "train_samples": len(X_train),

        "validation_samples": len(X_val),

        "roc_auc": auc,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,
    })


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nFold results:")

    print(
        f"  ROC-AUC : {auc:.4f}"
    )

    print(
        f"  Accuracy: {accuracy:.4f}"
    )

    print(
        f"  Precision: {precision:.4f}"
    )

    print(
        f"  Recall: {recall:.4f}"
    )

    print(
        f"  F1: {f1:.4f}"
    )


# ============================================================
# 9. CALCULATE CV SUMMARY
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


mean_auc = results_df["roc_auc"].mean()
std_auc = results_df["roc_auc"].std(ddof=0)

mean_accuracy = results_df["accuracy"].mean()
std_accuracy = results_df["accuracy"].std(ddof=0)

mean_precision = results_df["precision"].mean()
std_precision = results_df["precision"].std(ddof=0)

mean_recall = results_df["recall"].mean()
std_recall = results_df["recall"].std(ddof=0)

mean_f1 = results_df["f1"].mean()
std_f1 = results_df["f1"].std(ddof=0)


# ============================================================
# 10. PRINT FINAL CV RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("SPATIAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    f"AUC       : {mean_auc:.4f} ± {std_auc:.4f}"
)

print(
    f"ACCURACY  : "
    f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
)

print(
    f"PRECISION : "
    f"{mean_precision:.4f} ± {std_precision:.4f}"
)

print(
    f"RECALL    : "
    f"{mean_recall:.4f} ± {std_recall:.4f}"
)

print(
    f"F1        : "
    f"{mean_f1:.4f} ± {std_f1:.4f}"
)


# ============================================================
# 11. SAVE FOLD RESULTS
# ============================================================

results_df.to_csv(
    CV_RESULTS_PATH,
    index=False
)

print("\nCV results saved:")
print(
    f"  {CV_RESULTS_PATH}"
)


# ============================================================
# 12. SAVE CV SUMMARY
# ============================================================

cv_summary = {

    "model": "RandomForestClassifier",

    "features": FEATURES,

    "target": TARGET,

    "fold_column": FOLD_COLUMN,

    "number_of_folds": 5,

    "prediction_threshold":
        PREDICTION_THRESHOLD,

    "roc_auc": {
        "mean": mean_auc,
        "std": std_auc
    },

    "accuracy": {
        "mean": mean_accuracy,
        "std": std_accuracy
    },

    "precision": {
        "mean": mean_precision,
        "std": std_precision
    },

    "recall": {
        "mean": mean_recall,
        "std": std_recall
    },

    "f1": {
        "mean": mean_f1,
        "std": std_f1
    },

    "fold_results":
        fold_results,
}


with open(
    CV_SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        cv_summary,
        f,
        indent=4
    )


print(
    f"  {CV_SUMMARY_PATH}"
)


# ============================================================
# 13. TRAIN FINAL PRODUCTION MODEL
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PRODUCTION MODEL")
print("=" * 70)

print(
    "\nThe spatial CV evaluation is complete."
)

print(
    "Training final Random Forest on the complete dataset..."
)


final_rf_model = RandomForestClassifier(
    **RF_PARAMS
)

final_rf_model.fit(
    X,
    y
)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_rf_model,
    FINAL_MODEL_PATH
)

print(
    "\nFinal Random Forest model saved:"
)

print(
    f"  {FINAL_MODEL_PATH}"
)


# ============================================================
# 15. SAVE MODEL METADATA
# ============================================================

metadata = {

    "model_type":
        "RandomForestClassifier",

    "features":
        FEATURES,

    "target":
        TARGET,

    "spatial_fold_column":
        FOLD_COLUMN,

    "number_of_spatial_folds":
        5,

    "prediction_threshold":
        PREDICTION_THRESHOLD,

    "hyperparameters":
        RF_PARAMS,

    "training_samples":
        len(df),

    "presence_samples":
        int((y == 1).sum()),

    "pseudo_absence_samples":
        int((y == 0).sum()),

    "cross_validation": {

        "roc_auc_mean":
            mean_auc,

        "roc_auc_std":
            std_auc,

        "accuracy_mean":
            mean_accuracy,

        "accuracy_std":
            std_accuracy,

        "precision_mean":
            mean_precision,

        "precision_std":
            std_precision,

        "recall_mean":
            mean_recall,

        "recall_std":
            std_recall,

        "f1_mean":
            mean_f1,

        "f1_std":
            std_f1,
    },

    "artifact":
        FINAL_MODEL_PATH,
}


with open(
    METADATA_PATH,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print(
    f"Metadata saved:"
)

print(
    f"  {METADATA_PATH}"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nFinal CV performance:"
)

print(
    f"  ROC-AUC : "
    f"{mean_auc:.4f} ± {std_auc:.4f}"
)

print(
    f"  Accuracy: "
    f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
)

print(
    f"  Precision: "
    f"{mean_precision:.4f} ± {std_precision:.4f}"
)

print(
    f"  Recall: "
    f"{mean_recall:.4f} ± {std_recall:.4f}"
)

print(
    f"  F1: "
    f"{mean_f1:.4f} ± {std_f1:.4f}"
)

print("\nProduction artifact:")
print(
    f"  {FINAL_MODEL_PATH}"
)

print("\nIMPORTANT:")
print(
    "The CV metrics above are the model evaluation results."
)

print(
    "Do NOT report performance of the final full-data "
    "model as an independent test score."
)

print(
    "The final Random Forest is intended for "
    "production inference."
)

print("=" * 70)