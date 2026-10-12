import os
import json
import copy
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

# This script is inside:
# C:\Users\epooj\Downloads\Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset:
# C:\Users\epooj\Downloads\Model\Dataset
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# V2 output folder
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "GeoSea_Model_Output_V2"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

# Stage 1
HEAD_EPOCHS = 5

# Stage 2
FINETUNE_EPOCHS = 25

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

# Stop if validation accuracy does not improve
EARLY_STOPPING_PATIENCE = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GeoSea 12-Class Seaweed Classifier - V2")
print("=" * 70)

print("\nVersion 2 improvements:")
print("  - Aspect-ratio preserving preprocessing")
print("  - Conservative augmentation")
print("  - Label smoothing")
print("  - Learning-rate scheduling")
print("  - Early stopping")

print("\nDataset:")
print(DATASET_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("\nGPU detected:")
    print(torch.cuda.get_device_name(0))

else:

    device = torch.device("cpu")

    print("\nNo NVIDIA GPU detected.")
    print("Using CPU.")


# ============================================================
# TRANSFORMS
# ============================================================

"""
IMPORTANT:

V1 used:

    Resize((224, 224))

That forces every image into a square and can distort
the original seaweed shape.

V2 first resizes while preserving aspect ratio.

Training:
    Resize shortest side to 256
    RandomResizedCrop to 224
    Mild geometric/color augmentation

Validation/Test:
    Resize shortest side to 256
    CenterCrop to 224

No random augmentation is used for validation/test.
"""


# ------------------------------------------------------------
# TRAINING TRANSFORM
# ------------------------------------------------------------

train_transforms = transforms.Compose([

    # Preserve original aspect ratio
    transforms.Resize(256),

    # Conservative crop.
    # Scale stays close to the original image so important
    # seaweed morphology is less likely to be removed.
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0),
        ratio=(0.90, 1.10)
    ),

    # Seaweed can naturally be photographed from different
    # orientations.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    # Mild lighting/color variation
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ------------------------------------------------------------
# VALIDATION / TEST TRANSFORM
# ------------------------------------------------------------

eval_transforms = transforms.Compose([

    # Preserve aspect ratio
    transforms.Resize(256),

    # Deterministic crop
    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET PATHS
# ============================================================

train_dir = os.path.join(
    DATASET_DIR,
    "train"
)

val_dir = os.path.join(
    DATASET_DIR,
    "val"
)

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transforms
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.classes

num_classes = len(
    class_names
)


print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"\nNumber of classes : {num_classes}"
)

print(
    f"Training images  : {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images      : {len(test_dataset)}"
)

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        f"{i:2d} : {class_name}"
    )


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

if train_dataset.class_to_idx != val_dataset.class_to_idx:

    raise RuntimeError(
        "Training and validation class mappings are different!"
    )


if train_dataset.class_to_idx != test_dataset.class_to_idx:

    raise RuntimeError(
        "Training and test class mappings are different!"
    )


# Save mapping

with open(
    os.path.join(
        OUTPUT_DIR,
        "class_mapping.json"
    ),
    "w"
) as f:

    json.dump(
        train_dataset.class_to_idx,
        f,
        indent=4
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE FOR STAGE 1
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)


model = model.to(device)


print("\nModel:")
print("EfficientNet-B0")

print(
    f"Output classes: {num_classes}"
)


# ============================================================
# LOSS
# ============================================================

# Label smoothing helps reduce overconfidence,
# especially with a relatively small dataset.

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_labels.extend(
                labels.cpu().numpy()
            )


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": []
}


# ============================================================
# BEST MODEL
# ============================================================

best_val_accuracy = 0.0

best_epoch = 0

best_model_weights = copy.deepcopy(
    model.state_dict()
)

epochs_without_improvement = 0


# ============================================================
# STAGE 1
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 70)


optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE_HEAD,
    weight_decay=WEIGHT_DECAY
)


for epoch in range(
    HEAD_EPOCHS
):

    start_time = time.time()


    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )


    epoch_time = (
        time.time()
        -
        start_time
    )


    history["epoch"].append(
        len(history["epoch"]) + 1
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_acc
    )


    print(
        f"Epoch {epoch + 1}/{HEAD_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )


    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        best_epoch = (
            len(history["epoch"])
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_geosea_efficientnet_b0_v2.pth"
            )
        )

    else:

        epochs_without_improvement += 1


# ============================================================
# STAGE 2
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING EFFICIENTNET-B0")
print("=" * 70)


# Unfreeze entire backbone

for parameter in model.features.parameters():

    parameter.requires_grad = True


optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE_FINETUNE,
    weight_decay=WEIGHT_DECAY
)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


for epoch in range(
    FINETUNE_EPOCHS
):

    start_time = time.time()


    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )


    scheduler.step(
        val_acc
    )


    epoch_time = (
        time.time()
        -
        start_time
    )


    history["epoch"].append(
        len(history["epoch"]) + 1
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_acc
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.7f} | "
        f"Time: {epoch_time:.1f}s"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        best_epoch = (
            len(history["epoch"])
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_geosea_efficientnet_b0_v2.pth"
            )
        )

        print(
            "  ✓ New best validation accuracy!"
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        print(
            f"No validation improvement for "
            f"{EARLY_STOPPING_PATIENCE} epochs."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST V2 MODEL")
print("=" * 70)


model.load_state_dict(
    best_model_weights
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL V2 TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    model,
    test_loader,
    criterion
)


print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


print("\n" + "=" * 70)
print("V2 CLASSIFICATION REPORT")
print("=" * 70)

print(report)


with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report_v2.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


plt.figure(
    figsize=(13, 11)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "GeoSea 12-Class Confusion Matrix - V2"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_v2.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history_v2.csv"
    ),
    index=False
)


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "GeoSea V2 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve_v2.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)


plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "GeoSea V2 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve_v2.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# PER-CLASS RESULTS
# ============================================================

report_dict = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


rows = []


for class_name in class_names:

    rows.append({

        "class": class_name,

        "precision":
            report_dict[class_name]["precision"],

        "recall":
            report_dict[class_name]["recall"],

        "f1_score":
            report_dict[class_name]["f1-score"],

        "support":
            report_dict[class_name]["support"]

    })


per_class_df = pd.DataFrame(
    rows
)


per_class_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_class_results_v2.csv"
    ),
    index=False
)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_info = {

    "version": "V2",

    "model": "EfficientNet-B0",

    "num_classes": num_classes,

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "head_epochs": HEAD_EPOCHS,

    "finetune_epochs": FINETUNE_EPOCHS,

    "best_epoch": best_epoch,

    "best_validation_accuracy":
        best_val_accuracy,

    "test_accuracy":
        test_accuracy,

    "test_accuracy_percent":
        test_accuracy * 100,

    "label_smoothing": 0.1,

    "seed": SEED,

    "classes": class_names,

    "dataset": DATASET_DIR,

    "preprocessing":
        "Aspect-ratio preserving",

    "training_augmentation":
        [
            "RandomResizedCrop",
            "HorizontalFlip",
            "VerticalFlip",
            "Rotation",
            "ColorJitter"
        ]

}


with open(
    os.path.join(
        OUTPUT_DIR,
        "model_info_v2.json"
    ),
    "w"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "geosea_efficientnet_b0_final_v2.pth"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GEOS EA V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nV1 Test Accuracy: 77.08%"
)

print(
    f"V2 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("\nOutput files saved to:")

print(
    OUTPUT_DIR
)

print("\nImportant files:")

print(
    "  best_geosea_efficientnet_b0_v2.pth"
)

print(
    "  geosea_efficientnet_b0_final_v2.pth"
)

print(
    "  classification_report_v2.txt"
)

print(
    "  per_class_results_v2.csv"
)

print(
    "  confusion_matrix_v2.png"
)

print(
    "  accuracy_curve_v2.png"
)

print(
    "  loss_curve_v2.png"
)

print(
    "  training_history_v2.csv"
)

print(
    "  model_info_v2.json"
)

print(
    "  class_mapping.json"
)

print("\n" + "=" * 70)