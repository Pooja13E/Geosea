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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "GeoSea_ResNet50_Output")

IMAGE_SIZE = 224

# CPU-friendly settings. Reduce BATCH_SIZE to 4 if memory is low.
BATCH_SIZE = 8

# Train the new classification head first, then fine-tune ResNet's
# final layer group (layer4) rather than the entire network.
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 8

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42
EARLY_STOPPING_PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("GeoSea ResNet-50 Image Classifier")
print("=" * 70)
print("Dataset:", DATASET_DIR)
print("Output :", OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)
if device.type == "cpu":
    print("CPU training selected. This may take a while on a large dataset.")


# ============================================================
# TRANSFORMS
# Keep these consistent with the supplied EfficientNet script.
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        IMAGE_SIZE, scale=(0.90, 1.0), ratio=(0.90, 1.10)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# Expected structure:
# Dataset/
#   train/class_name/image.jpg
#   val/class_name/image.jpg
#   test/class_name/image.jpg
# ============================================================

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

for folder in (train_dir, val_dir, test_dir):
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Could not find dataset folder:\n{folder}\n"
            "Check that Dataset contains train, val, and test folders."
        )

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=eval_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=eval_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)

if num_classes < 2:
    raise RuntimeError(f"Expected at least 2 classes; found {num_classes}.")

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise RuntimeError("Training and validation class mappings differ.")
if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise RuntimeError("Training and test class mappings differ.")

print("\nDataset summary")
print("-" * 70)
print("Number of classes :", num_classes)
print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))
print("Test images       :", len(test_dataset))
print("Classes           :", class_names)

with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)


# Keep workers at zero for reliable Windows execution.
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)


# ============================================================
# MODEL: RESNET-50
# ============================================================

print("\nLoading pretrained ResNet-50...")
print("The first run may download ImageNet pretrained weights.")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Freeze the pretrained feature extractor for Stage 1.
for parameter in model.parameters():
    parameter.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print("Model: ResNet-50")
print("Output classes:", num_classes)


# ============================================================
# TRAIN / EVALUATE FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    # Keep frozen backbone batch-normalization layers in evaluation mode
    # during the head-only stage.
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
        all_labels,
        all_predictions
    )


def save_best_model(model, path):
    torch.save(model.state_dict(), path)


history = {
    "epoch": [],
    "stage": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_seconds": []
}

best_val_accuracy = -1.0
best_epoch = 0
best_model_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0
best_model_path = os.path.join(
    OUTPUT_DIR, "best_geosea_resnet50.pth"
)


# ============================================================
# STAGE 1: TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 70)

optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE_HEAD,
    weight_decay=WEIGHT_DECAY
)

global_epoch = 0

for epoch in range(HEAD_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
    elapsed = time.time() - start_time
    global_epoch += 1

    history["epoch"].append(global_epoch)
    history["stage"].append("head")
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["epoch_seconds"].append(elapsed)

    print(
        f"Head Epoch {epoch + 1}/{HEAD_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_epoch = global_epoch
        best_model_weights = copy.deepcopy(model.state_dict())
        save_best_model(model, best_model_path)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1


# ============================================================
# STAGE 2: FINE-TUNE ONLY RESNET layer4 + CLASSIFICATION HEAD
# This is less demanding on a CPU than unfreezing the whole model.
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING RESNET-50 layer4 AND CLASSIFICATION HEAD")
print("=" * 70)

for parameter in model.layer4.parameters():
    parameter.requires_grad = True
for parameter in model.fc.parameters():
    parameter.requires_grad = True

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE_FINETUNE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

for epoch in range(FINETUNE_EPOCHS):
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("Early stopping: validation accuracy has not improved.")
        break

    start_time = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step(val_acc)
    elapsed = time.time() - start_time
    global_epoch += 1

    history["epoch"].append(global_epoch)
    history["stage"].append("fine_tune_layer4")
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["epoch_seconds"].append(elapsed)

    print(
        f"Fine-tune Epoch {epoch + 1}/{FINETUNE_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_epoch = global_epoch
        best_model_weights = copy.deepcopy(model.state_dict())
        save_best_model(model, best_model_path)
        epochs_without_improvement = 0
        print("  New best validation accuracy saved.")
    else:
        epochs_without_improvement += 1


# ============================================================
# FINAL TEST EVALUATION USING BEST VALIDATION CHECKPOINT
# ============================================================

print("\nLoading best ResNet-50 checkpoint...")
model.load_state_dict(best_model_weights)

test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    model, test_loader, criterion
)

print("\n" + "=" * 70)
print("FINAL RESNET-50 TEST RESULTS")
print("=" * 70)
print(f"Test Loss              : {test_loss:.4f}")
print(f"Test Accuracy          : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy * 100:.2f}%)")
print(f"Best Epoch             : {best_epoch}")


# Classification report and macro-F1
report = classification_report(
    test_labels,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0
)
report_dict = classification_report(
    test_labels,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

print("\nClassification report")
print(report)
macro_f1 = report_dict["macro avg"]["f1-score"]
print(f"Macro-F1: {macro_f1:.4f}")

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w", encoding="utf-8"
) as f:
    f.write(report)
    f.write(f"\nMacro-F1: {macro_f1:.4f}\n")


# Confusion matrix
cm = confusion_matrix(
    test_labels, test_predictions, labels=list(range(num_classes))
)
plt.figure(figsize=(13, 11))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("GeoSea ResNet-50 Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()


# Training history and curves
history_df = pd.DataFrame(history)
history_df.to_csv(
    os.path.join(OUTPUT_DIR, "training_history.csv"), index=False
)

plt.figure(figsize=(10, 6))
plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Training Accuracy")
plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-50 Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-50 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
plt.close()


# Per-class metrics
per_class_rows = []
for class_name in class_names:
    metrics = report_dict[class_name]
    per_class_rows.append({
        "class": class_name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1-score"],
        "support": metrics["support"]
    })
pd.DataFrame(per_class_rows).to_csv(
    os.path.join(OUTPUT_DIR, "per_class_results.csv"), index=False
)


# Model size and inference latency
model_path = os.path.join(OUTPUT_DIR, "geosea_resnet50_final.pth")
torch.save(model.state_dict(), model_path)
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

model.eval()
sample_images, _ = next(iter(test_loader))
sample_images = sample_images[:min(4, len(sample_images))].to(device)
with torch.no_grad():
    # Warm-up
    _ = model(sample_images)
    start = time.perf_counter()
    repeats = 3
    for _ in range(repeats):
        _ = model(sample_images)
    elapsed = time.perf_counter() - start

inference_ms_per_image = (elapsed / (repeats * len(sample_images))) * 1000

model_info = {
    "model": "ResNet-50",
    "num_classes": num_classes,
    "classes": class_names,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "device": str(device),
    "head_epochs_configured": HEAD_EPOCHS,
    "finetune_epochs_configured": FINETUNE_EPOCHS,
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_val_accuracy,
    "test_accuracy": test_accuracy,
    "macro_f1": macro_f1,
    "model_size_mb": model_size_mb,
    "approx_inference_ms_per_image": inference_ms_per_image,
    "dataset": DATASET_DIR,
    "preprocessing": "Resize shortest side to 256, crop to 224, ImageNet normalization",
    "fine_tuning": "Classification head, then layer4 and classification head",
    "seed": SEED
}
with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=4)

print("\n" + "=" * 70)
print("GE OSEA RESNET-50 TRAINING COMPLETE")
print("=" * 70)
print("Output folder:", OUTPUT_DIR)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Approx. inference: {inference_ms_per_image:.2f} ms/image")
print("\nSaved files:")
print("  best_geosea_resnet50.pth")
print("  geosea_resnet50_final.pth")
print("  classification_report.txt")
print("  per_class_results.csv")
print("  confusion_matrix.png")
print("  accuracy_curve.png")
print("  loss_curve.png")
print("  training_history.csv")
print("  model_info.json")
print("  class_mapping.json")
