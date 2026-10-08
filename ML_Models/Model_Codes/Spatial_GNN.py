import os
import random
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "GeoSea_8_Variables.csv"
)


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


X_raw = df[
    features
].values.astype(np.float32)


coords = df[
    [
        "latitude",
        "longitude",
    ]
].values.astype(np.float32)


y = df[
    "presence"
].values.astype(np.float32)


folds = df[
    "spatial_fold"
].values


print("============================================================")
print("        TRAINING SPATIAL GNN WITH 5-FOLD SPATIAL CV")
print("============================================================")

print(
    f"Dataset size: {len(df):,}"
)

print(
    f"Environmental features: {len(features)}"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Device: {device}"
)


# ============================================================
# 3. SPATIAL GRAPH CONVOLUTION
# ============================================================

class SpatialGraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
    ):

        super().__init__()


        self.neighbor_linear = nn.Linear(
            in_features,
            out_features,
            bias=False,
        )


        self.self_linear = nn.Linear(
            in_features,
            out_features,
            bias=True,
        )


    def forward(
        self,
        x,
        adjacency,
    ):

        neighbor_features = torch.matmul(
            adjacency,
            x,
        )


        neighbor_output = (
            self.neighbor_linear(
                neighbor_features
            )
        )


        self_output = (
            self.self_linear(x)
        )


        return (
            neighbor_output
            + self_output
        )


# ============================================================
# 4. SPATIAL GNN MODEL
# ============================================================

class SpatialGNN(nn.Module):

    def __init__(
        self,
        input_dim=8,
        hidden_dim=64,
    ):

        super().__init__()


        self.gnn1 = SpatialGraphConv(
            input_dim,
            hidden_dim,
        )


        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )


        self.gnn2 = SpatialGraphConv(
            hidden_dim,
            hidden_dim,
        )


        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                32,
                1,
            ),
        )


    def forward(
        self,
        x,
        adjacency,
    ):

        x = self.gnn1(
            x,
            adjacency,
        )

        x = self.bn1(x)

        x = F.relu(x)


        x = self.gnn2(
            x,
            adjacency,
        )

        x = self.bn2(x)

        x = F.relu(x)


        return self.classifier(x)


# ============================================================
# 5. BUILD LOCAL KNN SPATIAL GRAPH
# ============================================================

def build_spatial_adj(
    batch_coords,
    k_neighbors=8,
):

    n_samples = len(
        batch_coords
    )


    if n_samples <= 1:

        return torch.eye(
            n_samples,
            dtype=torch.float32,
        )


    k = min(
        k_neighbors,
        n_samples - 1,
    )


    nearest_neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="kd_tree",
    )


    nearest_neighbors.fit(
        batch_coords
    )


    adjacency = (
        nearest_neighbors
        .kneighbors_graph(
            batch_coords,
            mode="connectivity",
        )
        .toarray()
    )


    # Remove self-connections.

    np.fill_diagonal(
        adjacency,
        0,
    )


    # Row-normalize adjacency.

    degree = np.sum(
        adjacency,
        axis=1,
        keepdims=True,
    )


    degree[
        degree == 0
    ] = 1.0


    normalized_adjacency = (
        adjacency / degree
    )


    return torch.tensor(
        normalized_adjacency,
        dtype=torch.float32,
    )


# ============================================================
# 6. TRAINING SETTINGS
# ============================================================

batch_size = 2048

epochs = 12

k_neighbors = 8

learning_rate = 0.005

weight_decay = 1e-4


# ============================================================
# 7. METRICS STORAGE
# ============================================================

metrics = {
    "auc": [],
    "acc": [],
    "prec": [],
    "rec": [],
    "f1": [],
}


# ============================================================
# 8. 5-FOLD SPATIAL CROSS-VALIDATION
# ============================================================

for fold in range(5):

    print("\n------------------------------------------------------------")
    print(f"FOLD {fold}")
    print("------------------------------------------------------------")


    # --------------------------------------------------------
    # Spatial split
    # --------------------------------------------------------

    train_mask = (
        folds != fold
    )

    val_mask = (
        folds == fold
    )


    X_train_raw = X_raw[
        train_mask
    ]

    coords_train = coords[
        train_mask
    ]

    y_train = y[
        train_mask
    ]


    X_val_raw = X_raw[
        val_mask
    ]

    coords_val = coords[
        val_mask
    ]

    y_val = y[
        val_mask
    ]


    print(
        f"Training samples:   {len(X_train_raw):,}"
    )

    print(
        f"Validation samples: {len(X_val_raw):,}"
    )


    # --------------------------------------------------------
    # STRICT FOLD-WISE SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_train_raw
    ).astype(np.float32)


    X_val = scaler.transform(
        X_val_raw
    ).astype(np.float32)


    # --------------------------------------------------------
    # NEW MODEL FOR EACH FOLD
    # --------------------------------------------------------

    model = SpatialGNN(
        input_dim=8,
        hidden_dim=64,
    ).to(device)


    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        epochs
    ):

        model.train()


        indices = np.random.permutation(
            len(X_train)
        )


        n_train_batches = int(
            np.ceil(
                len(X_train)
                / batch_size
            )
        )


        epoch_loss = 0.0

        processed_batches = 0


        for b in range(
            n_train_batches
        ):

            batch_indices = indices[
                b * batch_size:
                (b + 1) * batch_size
            ]


            if len(batch_indices) < 16:
                continue


            batch_x = torch.tensor(
                X_train[
                    batch_indices
                ],
                dtype=torch.float32,
            ).to(device)


            batch_y = torch.tensor(
                y_train[
                    batch_indices
                ],
                dtype=torch.float32,
            ).unsqueeze(1).to(device)


            # ------------------------------------------------
            # Local spatial graph
            # ------------------------------------------------

            batch_adj = build_spatial_adj(
                coords_train[
                    batch_indices
                ],
                k_neighbors=k_neighbors,
            ).to(device)


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            optimizer.zero_grad()


            logits = model(
                batch_x,
                batch_adj,
            )


            loss = criterion(
                logits,
                batch_y,
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()


            epoch_loss += (
                loss.item()
            )

            processed_batches += 1


        if (
            epoch == 0
            or (epoch + 1) % 4 == 0
            or epoch == epochs - 1
        ):

            avg_loss = (
                epoch_loss
                / max(
                    processed_batches,
                    1,
                )
            )


            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"| Loss: {avg_loss:.6f}"
            )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    validation_predictions = []


    n_val_batches = int(
        np.ceil(
            len(X_val)
            / batch_size
        )
    )


    with torch.no_grad():

        for b in range(
            n_val_batches
        ):

            start = (
                b * batch_size
            )


            end = min(
                (b + 1) * batch_size,
                len(X_val),
            )


            batch_x = torch.tensor(
                X_val[
                    start:end
                ],
                dtype=torch.float32,
            ).to(device)


            batch_coords = (
                coords_val[
                    start:end
                ]
            )


            # ------------------------------------------------
            # Validation graph uses validation coordinates only
            # ------------------------------------------------

            batch_adj = build_spatial_adj(
                batch_coords,
                k_neighbors=k_neighbors,
            ).to(device)


            logits = model(
                batch_x,
                batch_adj,
            )


            probabilities = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )


            validation_predictions.append(
                probabilities
            )


    y_pred_proba = (
        np.vstack(
            validation_predictions
        )
        .flatten()
    )


    # --------------------------------------------------------
    # Threshold
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


    metrics["auc"].append(
        auc
    )

    metrics["acc"].append(
        acc
    )

    metrics["prec"].append(
        prec
    )

    metrics["rec"].append(
        rec
    )

    metrics["f1"].append(
        f1
    )


    # --------------------------------------------------------
    # Fold results
    # --------------------------------------------------------

    print(
        "\nFold Results"
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
# 9. MEAN SPATIAL CV PERFORMANCE
# ============================================================

print("\n============================================================")
print("          SPATIAL GNN MEAN SPATIAL CV PERFORMANCE")
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
# 10. SAVE CV RESULTS
# ============================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)


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
    "spatial_gnn_cv_results.csv",
)


cv_results.to_csv(
    cv_path,
    index=False,
)


# ============================================================
# 11. FINAL PRODUCTION MODEL
# ============================================================

print("\n============================================================")
print("        TRAINING FINAL PRODUCTION SPATIAL GNN")
print("============================================================")


final_scaler = StandardScaler()


X_full = final_scaler.fit_transform(
    X_raw
).astype(np.float32)


final_model = SpatialGNN(
    input_dim=8,
    hidden_dim=64,
).to(device)


final_criterion = nn.BCEWithLogitsLoss()


final_optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


# ------------------------------------------------------------
# Initial shuffle
# ------------------------------------------------------------

full_indices = np.random.permutation(
    len(X_full)
)


n_full_batches = int(
    np.ceil(
        len(X_full)
        / batch_size
    )
)


# ------------------------------------------------------------
# Final training
# ------------------------------------------------------------

for epoch in range(
    epochs
):

    final_model.train()


    if epoch > 0:

        full_indices = np.random.permutation(
            len(X_full)
        )


    epoch_loss = 0.0

    processed_batches = 0


    for b in range(
        n_full_batches
    ):

        batch_indices = full_indices[
            b * batch_size:
            (b + 1) * batch_size
        ]


        if len(batch_indices) < 16:
            continue


        batch_x = torch.tensor(
            X_full[
                batch_indices
            ],
            dtype=torch.float32,
        ).to(device)


        batch_y = torch.tensor(
            y[
                batch_indices
            ],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)


        batch_adj = build_spatial_adj(
            coords[
                batch_indices
            ],
            k_neighbors=k_neighbors,
        ).to(device)


        final_optimizer.zero_grad()


        logits = final_model(
            batch_x,
            batch_adj,
        )


        loss = final_criterion(
            logits,
            batch_y,
        )


        loss.backward()

        final_optimizer.step()


        epoch_loss += (
            loss.item()
        )

        processed_batches += 1


    if (
        epoch == 0
        or (epoch + 1) % 4 == 0
        or epoch == epochs - 1
    ):

        avg_loss = (
            epoch_loss
            / max(
                processed_batches,
                1,
            )
        )


        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {avg_loss:.6f}"
        )


# ============================================================
# 12. SAVE PRODUCTION MODEL
# ============================================================

model_path = os.path.join(
    current_dir,
    "geosea_spatial_gnn_model.pth",
)


scaler_path = os.path.join(
    current_dir,
    "geosea_spatial_scaler.joblib",
)


torch.save(
    final_model.state_dict(),
    model_path,
)


joblib.dump(
    final_scaler,
    scaler_path,
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("          SPATIAL GNN TRAINING COMPLETE")
print("============================================================")


print(
    f"Production model: {model_path}"
)


print(
    f"Scaler:            {scaler_path}"
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
    f"Mean F1:            {mean_f1:.4f}"
)


print("\nSpatial GNN model saved successfully.")