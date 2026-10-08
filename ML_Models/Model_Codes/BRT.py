# ============================================================
# GeoSea - BRT (Boosted Regression Trees)
# 5-Fold Spatial Cross-Validation
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "GeoSea_8_Variables.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "geosea_brt_model.joblib"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "brt_spatial_cv_results.csv"
)

FEATURES = [
    "temperature",
    "salinity",
    "chlorophyll",
    "nitrate",
    "phosphate",
    "pH",
    "PAR",
    "kdpar"
]

TARGET = "presence"

FOLD_COLUMN = "spatial_fold"

RANDOM_STATE = 42


# ============================================================
# BRT PARAMETERS
# ============================================================

N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10
SUBSAMPLE = 0.8


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GeoSea - BRT / Boosted Regression Trees")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = (
    FEATURES +
    [TARGET, FOLD_COLUMN]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

# Convert environmental variables
for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

# Convert target
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# Convert fold
df[FOLD_COLUMN] = pd.to_numeric(
    df[FOLD_COLUMN],
    errors="coerce"
)

# Remove missing rows
before = len(df)

df = df.dropna(
    subset=FEATURES + [
        TARGET,
        FOLD_COLUMN
    ]
).copy()

after = len(df)

print(
    f"Removed {before - after} invalid rows"
)

# Keep only binary target
df = df[
    df[TARGET].isin([0, 1])
].copy()

print(
    f"Final dataset shape: {df.shape}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df[
    FEATURES
].values

y = df[
    TARGET
].astype(int).values

folds = df[
    FOLD_COLUMN
].astype(int).values

unique_folds = sorted(
    df[FOLD_COLUMN].unique()
)


# ============================================================
# DATA SUMMARY
# ============================================================

print("\nFeatures:")

for feature in FEATURES:
    print(f"  - {feature}")

print("\nTarget distribution:")

target_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print(
    target_counts.rename(
        index={
            0: "Pseudo-absence",
            1: "Presence"
        }
    )
)


# ============================================================
# SPATIAL CROSS-VALIDATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("5-FOLD SPATIAL CROSS-VALIDATION")
print("=" * 70)

for fold in unique_folds:

    print(
        f"\n--- Fold {fold} ---"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_mask = folds != fold
    test_mask = folds == fold

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples:  {len(X_test):,}"
    )

    # --------------------------------------------------------
    # BRT MODEL
    # --------------------------------------------------------

    model = GradientBoostingClassifier(

        n_estimators=N_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        subsample=SUBSAMPLE,

        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    y_prediction = (
        y_probability >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"AUC:       {auc:.4f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({

        "fold": fold,

        "train_samples":
            len(X_train),

        "test_samples":
            len(X_test),

        "auc":
            auc,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("BRT CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "fold",
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)


# ============================================================
# MEAN ± STANDARD DEVIATION
# ============================================================

metrics = [
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1"
]

print(
    "\nMean ± Standard Deviation:"
)

summary = {}

for metric in metrics:

    mean_value = (
        results_df[metric].mean()
    )

    std_value = (
        results_df[metric].std()
    )

    summary[metric] = {
        "mean": mean_value,
        "std": std_value
    }

    print(
        f"{metric.capitalize():<10}: "
        f"{mean_value:.4f} ± "
        f"{std_value:.4f}"
    )


# ============================================================
# SAVE CV RESULTS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    f"\nCV results saved to:"
    f" {RESULTS_PATH}"
)


# ============================================================
# FINAL PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL BRT MODEL")
print("=" * 70)

final_model = GradientBoostingClassifier(

    n_estimators=N_ESTIMATORS,

    learning_rate=LEARNING_RATE,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    subsample=SUBSAMPLE,

    random_state=RANDOM_STATE
)

final_model.fit(
    X,
    y
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)

print(
    f"\nModel saved to:"
    f"\n  {MODEL_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("BRT FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        final_model.feature_importances_

})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# COMPLETED
# ============================================================

print(
    "\nBRT training completed successfully."
)