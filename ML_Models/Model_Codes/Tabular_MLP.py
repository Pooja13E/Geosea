"""
GeoSea - Tabular MLP
====================
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

CSV_PATH = "GeoSea_8_Variables.csv"

MODEL_DIR = "models"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "geosea_mlp_model.pth"
)

FINAL_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "geosea_mlp_scaler.joblib"
)

CV_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "mlp_spatial_cv_results.csv"
)

CV_SUMMARY_PATH = os.path.join(
    MODEL_DIR,
    "mlp_spatial_cv_summary.json"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "geosea_mlp_metadata.json"
)


# Reproducibility
SEED = 42

# Spatial CV
EXPECTED_FOLDS = [0, 1, 2, 3, 4]

# Features used by the model
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

# Training
EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 0.003

# Classification threshold
THRESHOLD = 0.5

# Final production training
FINAL_EPOCHS = 20


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These improve reproducibility on CUDA.
    # They can slightly reduce performance, but this is acceptable
    # for model evaluation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GeoSea - Tabular MLP")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Random seed: {SEED}")
print("=" * 70)


# ============================================================
# 5. MODEL
# ============================================================

class TabularMLP(nn.Module):
    """
    Feed-forward neural network for the eight environmental
    variables used by GeoSea.
    """

    def __init__(self, input_dim: int = 8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# 6. LOAD DATASET
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}\n\n"
        "Place GeoSea_8_Variables.csv in the same directory "
        "as this script or update CSV_PATH."
    )

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 7. VALIDATE DATASET STRUCTURE
# ============================================================

required_columns = FEATURES + [
    TARGET,
    FOLD_COLUMN
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

print("\nRequired columns found:")
for col in required_columns:
    print(f"  ✓ {col}")


# ============================================================
# 8. VALIDATE FEATURES / TARGET
# ============================================================

print("\nChecking missing values...")

feature_missing = df[FEATURES].isna().sum()

if feature_missing.sum() > 0:
    print("\nMissing feature values found:")
    print(feature_missing[feature_missing > 0])

    raise ValueError(
        "Missing values exist in model features. "
        "Do not silently fill them during this validation."
    )

target_missing = df[TARGET].isna().sum()

if target_missing > 0:
    raise ValueError(
        f"{target_missing} target values are missing."
    )

fold_missing = df[FOLD_COLUMN].isna().sum()

if fold_missing > 0:
    raise ValueError(
        f"{fold_missing} spatial fold values are missing."
    )


# ============================================================
# 9. VALIDATE TARGET
# ============================================================

unique_targets = sorted(
    df[TARGET].unique().tolist()
)

print(f"\nTarget values: {unique_targets}")

if unique_targets != [0, 1]:
    raise ValueError(
        "Target must contain exactly the binary values 0 and 1."
    )


# ============================================================
# 10. VALIDATE SPATIAL FOLDS
# ============================================================

fold_values = sorted(
    df[FOLD_COLUMN].unique().tolist()
)

print(f"\nSpatial folds found: {fold_values}")

if fold_values != EXPECTED_FOLDS:
    raise ValueError(
        "\nUnexpected spatial fold structure.\n"
        f"Expected: {EXPECTED_FOLDS}\n"
        f"Found:    {fold_values}\n\n"
        "Do not continue until the fold structure is verified."
    )

print("✓ Spatial fold structure is [0, 1, 2, 3, 4]")


print("\nRecords per spatial fold:")

fold_counts = (
    df[FOLD_COLUMN]
    .value_counts()
    .sort_index()
)

for fold, count in fold_counts.items():
    print(f"  Fold {int(fold)}: {count:,}")


# ============================================================
# 11. PREPARE ARRAYS
# ============================================================

X_raw = df[FEATURES].astype(np.float32).values

y = df[TARGET].astype(np.float32).values

folds = df[FOLD_COLUMN].astype(int).values


print("\nFeature matrix:")
print(f"  Samples:  {X_raw.shape[0]:,}")
print(f"  Features: {X_raw.shape[1]}")

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_model(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
):
    """
    Train a fresh MLP model.
    """

    model = TabularMLP(
        input_dim=len(FEATURES)
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    # Generator ensures reproducible DataLoader shuffling.
    generator = torch.Generator()

    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() *
                batch_X.size(0)
            )

        average_loss = (
            total_loss /
            len(dataset)
        )

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == epochs - 1
        ):
            print(
                f"      Epoch {epoch + 1:02d}/{epochs} "
                f"- Loss: {average_loss:.5f}"
            )

    return model


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model using:
        - ROC-AUC from probabilities
        - Accuracy
        - Precision
        - Recall
        - F1 at threshold 0.5
    """

    model.eval()

    X_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    auc = roc_auc_score(
        y_val,
        probabilities
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 14. SPATIAL CROSS-VALIDATION
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
    # IMPORTANT:
    # Validation fold is completely excluded from training.
    # --------------------------------------------------------

    train_mask = folds != fold
    val_mask = folds == fold

    X_train_raw = X_raw[train_mask]
    X_val_raw = X_raw[val_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]

    print(
        f"Training samples:   {len(X_train_raw):,}"
    )

    print(
        f"Validation samples: {len(X_val_raw):,}"
    )

    # --------------------------------------------------------
    # FOLD-WISE SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    # IMPORTANT:
    # FIT ONLY ON TRAINING DATA.
    X_train = scaler.fit_transform(
        X_train_raw
    )

    # IMPORTANT:
    # VALIDATION IS TRANSFORMED ONLY.
    X_val = scaler.transform(
        X_val_raw
    )

    print("Scaler: fitted on training fold only ✓")

    # --------------------------------------------------------
    # CREATE FRESH MODEL
    # --------------------------------------------------------

    # Reset seed so each fold is reproducible.
    set_seed(SEED + fold)

    print("\nTraining MLP...")

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_val,
        y_val
    )

    print("\nFold results:")
    print(
        f"  ROC-AUC:  {metrics['auc']:.4f}"
    )
    print(
        f"  Accuracy: {metrics['accuracy']:.4f}"
    )
    print(
        f"  Precision:{metrics['precision']:.4f}"
    )
    print(
        f"  Recall:   {metrics['recall']:.4f}"
    )
    print(
        f"  F1:       {metrics['f1']:.4f}"
    )

    fold_results.append({
        "fold": fold,
        **metrics
    })


# ============================================================
# 15. CROSS-VALIDATION SUMMARY
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

print("\n")
print("=" * 70)
print("SPATIAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

metric_columns = [
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1"
]

summary = {}

for metric in metric_columns:

    mean_value = results_df[metric].mean()

    std_value = results_df[metric].std(
        ddof=1
    )

    summary[metric] = {
        "mean": float(mean_value),
        "std": float(std_value)
    }

    print(
        f"{metric.upper():<10}: "
        f"{mean_value:.4f} ± {std_value:.4f}"
    )


# ============================================================
# 16. SAVE CV RESULTS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

results_df.to_csv(
    CV_RESULTS_PATH,
    index=False
)

with open(
    CV_SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        {
            "seed": SEED,
            "features": FEATURES,
            "target": TARGET,
            "fold_column": FOLD_COLUMN,
            "folds": EXPECTED_FOLDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "threshold": THRESHOLD,
            "metrics": summary
        },
        f,
        indent=4
    )


print("\nCV results saved:")
print(f"  {CV_RESULTS_PATH}")
print(f"  {CV_SUMMARY_PATH}")


# ============================================================
# 17. FINAL PRODUCTION MODEL
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PRODUCTION MODEL")
print("=" * 70)

print(
    "\nThe spatial CV evaluation is now complete."
)

print(
    "Training a final model on the complete dataset..."
)

# ------------------------------------------------------------
# IMPORTANT:
#
# This is intentionally different from CV.
#
# CV:
#   scaler fitted separately on each training fold.
#
# Production:
#   scaler fitted on ALL available training records.
#
# This is valid AFTER model evaluation.
# ------------------------------------------------------------

final_scaler = StandardScaler()

X_full = final_scaler.fit_transform(
    X_raw
)

print(
    "Final scaler fitted on complete dataset."
)

# Fresh seed for final model.
set_seed(SEED)

final_model = train_model(
    X_full,
    y,
    epochs=FINAL_EPOCHS
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

torch.save(
    final_model.state_dict(),
    FINAL_MODEL_PATH
)

joblib.dump(
    final_scaler,
    FINAL_SCALER_PATH
)


# ============================================================
# 19. SAVE METADATA
# ============================================================

metadata = {
    "model": "TabularMLP",
    "input_features": FEATURES,
    "target": TARGET,

    "dataset": CSV_PATH,
    "dataset_rows": int(len(df)),

    "spatial_cv": {
        "folds": EXPECTED_FOLDS,
        "scaling": "fit on training fold only",
        "metric_threshold": THRESHOLD
    },

    "training": {
        "epochs_cv": EPOCHS,
        "epochs_final": FINAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED
    },

    "cv_results": summary,

    "production_scaler": (
        "fitted on complete dataset after CV evaluation"
    ),

    "model_path": FINAL_MODEL_PATH,
    "scaler_path": FINAL_SCALER_PATH
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


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nFinal CV performance:")

print(
    f"  ROC-AUC:  "
    f"{summary['auc']['mean']:.4f} "
    f"± {summary['auc']['std']:.4f}"
)

print(
    f"  Accuracy: "
    f"{summary['accuracy']['mean']:.4f} "
    f"± {summary['accuracy']['std']:.4f}"
)

print(
    f"  Precision:"
    f"{summary['precision']['mean']:.4f} "
    f"± {summary['precision']['std']:.4f}"
)

print(
    f"  Recall:   "
    f"{summary['recall']['mean']:.4f} "
    f"± {summary['recall']['std']:.4f}"
)

print(
    f"  F1:       "
    f"{summary['f1']['mean']:.4f} "
    f"± {summary['f1']['std']:.4f}"
)

print("\nSaved production artifacts:")

print(
    f"  Model:    {FINAL_MODEL_PATH}"
)

print(
    f"  Scaler:   {FINAL_SCALER_PATH}"
)

print(
    f"  Metadata: {METADATA_PATH}"
)

print(
    f"  CV:       {CV_RESULTS_PATH}"
)

print("\n")
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "The CV metrics above are the metrics to use for model evaluation."
)

print(
    "Do NOT report the performance of the final full-data model "
    "as an independent test score."
)

print(
    "The final model is intended for production inference only."
)

print("=" * 70)