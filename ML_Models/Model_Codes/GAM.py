import os
import joblib
import numpy as np
import pandas as pd

from pygam import LogisticGAM, s

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("GeoSea_8_Variables.csv")


features = [
    "temperature",
    "salinity",
    "chlorophyll",
    "nitrate",
    "phosphate",
    "pH",
    "PAR",
    "kdpar",
]


X = df[features]

y = df["presence"]

folds = df["spatial_fold"]


print("============================================================")
print("           TRAINING GAM WITH 5-FOLD SPATIAL CV")
print("============================================================")

print(f"Dataset size:        {len(df):,}")
print(f"Environmental features: {len(features)}")

print("\nFeatures:")
for feature in features:
    print(f"  - {feature}")


# ============================================================
# 2. GAM STRUCTURE
# ============================================================

# One smooth spline term for every environmental variable.

gam_terms = (
    s(0)
    + s(1)
    + s(2)
    + s(3)
    + s(4)
    + s(5)
    + s(6)
    + s(7)
)


# ============================================================
# 3. METRICS STORAGE
# ============================================================

metrics = {
    "auc": [],
    "acc": [],
    "prec": [],
    "rec": [],
    "f1": [],
}


# ============================================================
# 4. 5-FOLD SPATIAL CROSS-VALIDATION
# ============================================================

for fold in range(5):

    print("\n------------------------------------------------------------")
    print(f"FOLD {fold}")
    print("------------------------------------------------------------")


    # --------------------------------------------------------
    # Spatial split
    # --------------------------------------------------------

    train_idx = folds != fold
    val_idx = folds == fold


    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]


    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")


    # --------------------------------------------------------
    # Representative training subset
    # --------------------------------------------------------
    #
    # GAM is computationally expensive on the complete
    # 300k+ row dataset.
    #
    # IMPORTANT:
    # The subset is selected ONLY from the training fold.
    #
    # Therefore validation data never enters GAM fitting.
    # --------------------------------------------------------

    sample_size = min(
        40000,
        len(X_train)
    )


    rng = np.random.RandomState(
        SEED + fold
    )


    sample_idx = rng.choice(
        len(X_train),
        size=sample_size,
        replace=False,
    )


    X_train_fit = X_train.iloc[
        sample_idx
    ]

    y_train_fit = y_train.iloc[
        sample_idx
    ]


    print(
        f"GAM fitting subset: {len(X_train_fit):,}"
    )


    # --------------------------------------------------------
    # Train GAM
    # --------------------------------------------------------

    gam = LogisticGAM(
        gam_terms,
        max_iter=100,
    )


    gam.fit(
        X_train_fit.values,
        y_train_fit.values,
    )


    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    y_pred_proba = gam.predict_proba(
        X_val.values
    )


    # --------------------------------------------------------
    # Convert probability to class
    # --------------------------------------------------------
    #
    # Same threshold used by the other models:
    #
    # >= 0.5 -> Presence
    # <  0.5 -> Pseudo-absence
    #
    # ROC-AUC itself does not depend on this threshold.
    # --------------------------------------------------------

    y_pred = (
        y_pred_proba >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_val,
        y_pred_proba,
    )


    acc = accuracy_score(
        y_val,
        y_pred,
    )


    prec = precision_score(
        y_val,
        y_pred,
        zero_division=0,
    )


    rec = recall_score(
        y_val,
        y_pred,
        zero_division=0,
    )


    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0,
    )


    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    metrics["auc"].append(auc)
    metrics["acc"].append(acc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["f1"].append(f1)


    # --------------------------------------------------------
    # Print fold results
    # --------------------------------------------------------

    print(
        f"\nFold {fold} Results"
    )

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print(
        f"Accuracy:  {acc:.4f}"
    )

    print(
        f"Precision: {prec:.4f}"
    )

    print(
        f"Recall:    {rec:.4f}"
    )

    print(
        f"F1-Score:  {f1:.4f}"
    )


# ============================================================
# 5. MEAN SPATIAL CV PERFORMANCE
# ============================================================

print("\n============================================================")
print("             GAM MEAN SPATIAL CV PERFORMANCE")
print("============================================================")


mean_auc = np.mean(
    metrics["auc"]
)

std_auc = np.std(
    metrics["auc"]
)


mean_acc = np.mean(
    metrics["acc"]
)

std_acc = np.std(
    metrics["acc"]
)


mean_prec = np.mean(
    metrics["prec"]
)

std_prec = np.std(
    metrics["prec"]
)


mean_rec = np.mean(
    metrics["rec"]
)

std_rec = np.std(
    metrics["rec"]
)


mean_f1 = np.mean(
    metrics["f1"]
)

std_f1 = np.std(
    metrics["f1"]
)


print(
    f"Mean ROC-AUC:  {mean_auc:.4f} ± {std_auc:.4f}"
)

print(
    f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}"
)

print(
    f"Mean Precision:{mean_prec:.4f} ± {std_prec:.4f}"
)

print(
    f"Mean Recall:   {mean_rec:.4f} ± {std_rec:.4f}"
)

print(
    f"Mean F1:       {mean_f1:.4f} ± {std_f1:.4f}"
)


# ============================================================
# 6. TRAIN FINAL PRODUCTION GAM
# ============================================================

print("\n============================================================")
print("           TRAINING FINAL PRODUCTION GAM")
print("============================================================")


# Use a larger deterministic subset for the final model.

final_sample_size = min(
    60000,
    len(X)
)


rng = np.random.RandomState(
    SEED
)


final_sample_idx = rng.choice(
    len(X),
    size=final_sample_size,
    replace=False,
)


X_final_fit = X.iloc[
    final_sample_idx
]

y_final_fit = y.iloc[
    final_sample_idx
]


print(
    f"Final GAM fitting samples: {len(X_final_fit):,}"
)


final_gam = LogisticGAM(
    gam_terms,
    max_iter=150,
)


final_gam.fit(
    X_final_fit.values,
    y_final_fit.values,
)


# ============================================================
# 7. SAVE MODEL
# ============================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)


model_path = os.path.join(
    current_dir,
    "geosea_gam_model.joblib",
)


joblib.dump(
    final_gam,
    model_path,
    compress=3,
)


# ============================================================
# 8. SAVE CV RESULTS
# ============================================================

cv_results = pd.DataFrame({
    "fold": range(5),
    "roc_auc": metrics["auc"],
    "accuracy": metrics["acc"],
    "precision": metrics["prec"],
    "recall": metrics["rec"],
    "f1": metrics["f1"],
})


cv_path = os.path.join(
    current_dir,
    "gam_spatial_cv_results.csv",
)


cv_results.to_csv(
    cv_path,
    index=False,
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("             GAM TRAINING COMPLETE")
print("============================================================")

print(
    f"Production model: {model_path}"
)

print(
    f"CV results:        {cv_path}"
)

print(
    f"Mean ROC-AUC:      {mean_auc:.4f} ± {std_auc:.4f}"
)

print(
    f"Mean Accuracy:     {mean_acc * 100:.2f}%"
)

print(
    f"Mean Precision:    {mean_prec * 100:.2f}%"
)

print(
    f"Mean Recall:       {mean_rec * 100:.2f}%"
)

print(
    f"Mean F1:           {mean_f1:.4f}"
)

print("\nGAM model saved successfully.")