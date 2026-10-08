# ============================================================
# GeoSea - GLM (Logistic Regression)
# 5-Fold Spatial Cross-Validation
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    "geosea_glm_model.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "geosea_glm_scaler.joblib"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "glm_spatial_cv_results.csv"
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("GeoSea - GLM / Logistic Regression")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET, FOLD_COLUMN]

missing_columns = [
    col for col in required_columns
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

# Convert features to numeric
for feature in FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

# Convert target to numeric
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# Convert spatial fold
df[FOLD_COLUMN] = pd.to_numeric(
    df[FOLD_COLUMN],
    errors="coerce"
)

# Remove invalid rows
before = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET, FOLD_COLUMN]
).copy()

after = len(df)

print(f"Removed {before - after} invalid rows")

# Ensure binary target
df = df[df[TARGET].isin([0, 1])].copy()

print(f"Final dataset shape: {df.shape}")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].values
y = df[TARGET].astype(int).values
folds = df[FOLD_COLUMN].astype(int).values

unique_folds = sorted(
    df[FOLD_COLUMN].unique()
)

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nTarget distribution:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Pseudo-absence",
        1: "Presence"
    })
)


# ============================================================
# SPATIAL CROSS-VALIDATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("5-FOLD SPATIAL CROSS-VALIDATION")
print("=" * 70)

for fold in unique_folds:

    print(f"\n--- Fold {fold} ---")

    train_mask = folds != fold
    test_mask = folds == fold

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    # --------------------------------------------------------
    # SCALE USING TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # GLM
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test_scaled
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

    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    results.append({
        "fold": fold,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GLM CROSS-VALIDATION RESULTS")
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

print("\nMean ± Standard Deviation:")

summary = {}

for metric in metrics:

    mean_value = results_df[metric].mean()
    std_value = results_df[metric].std()

    summary[metric] = {
        "mean": mean_value,
        "std": std_value
    }

    print(
        f"{metric.capitalize():<10}: "
        f"{mean_value:.4f} ± {std_value:.4f}"
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
    f"\nCV results saved to: {RESULTS_PATH}"
)


# ============================================================
# FINAL PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL GLM MODEL")
print("=" * 70)

# Scale entire dataset
final_scaler = StandardScaler()

X_scaled = final_scaler.fit_transform(
    X
)

# Train final GLM
final_model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE
)

final_model.fit(
    X_scaled,
    y
)

# ============================================================
# SAVE MODEL + SCALER
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)

joblib.dump(
    final_scaler,
    SCALER_PATH
)

print(f"\nModel saved to:")
print(f"  {MODEL_PATH}")

print(f"\nScaler saved to:")
print(f"  {SCALER_PATH}")


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("GLM FEATURE COEFFICIENTS")
print("=" * 70)

coefficients = final_model.coef_[0]

coefficient_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": coefficients
})

coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False
)

print(
    coefficient_df[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

print("\nGLM training completed successfully.")