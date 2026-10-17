"""
GeoSea MobileNetV3-Large training script (CPU-friendly).

Expected folders beside this script:
Dataset/train/<class_name>/
Dataset/val/<class_name>/
Dataset/test/<class_name>/

Install:
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn

Run:
python train_geosea_mobilenetv3.py
"""
import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "GeoSea_MobileNetV3_Output"

IMAGE_SIZE = 224
BATCH_SIZE = 8
HEAD_EPOCHS = 4
FINETUNE_EPOCHS = 8
PATIENCE = 3
NUM_WORKERS = 0  # Recommended for Windows
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.90, 1.0), ratio=(0.90, 1.10)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    print("=" * 60, flush=True)
    print("GEOSEA MOBILENETV3-LARGE TRAINING", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Dataset: {DATASET_DIR}", flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)

    for split in ("train", "val", "test"):
        if not (DATASET_DIR / split).is_dir():
            raise FileNotFoundError(
                f"Missing folder: {DATASET_DIR / split}\n"
                "Expected Dataset/train, Dataset/val and Dataset/test."
            )

    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=EVAL_TRANSFORM)
    test_ds = datasets.ImageFolder(DATASET_DIR / "test", transform=EVAL_TRANSFORM)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Train, val and test must contain the same class folder names.")

    classes = train_ds.classes
    n_classes = len(classes)
    print(f"Training images: {len(train_ds)}", flush=True)
    print(f"Validation images: {len(val_ds)}", flush=True)
    print(f"Test images: {len(test_ds)}", flush=True)
    print(f"Classes ({n_classes}): {classes}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("\nLoading pretrained MobileNetV3-Large weights...", flush=True)
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=2)

    history = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    def run_epoch(loader, optimizer=None, training=False, freeze_features_bn=False):
        model.train(training)
        if training and freeze_features_bn:
            # Keep frozen feature extractor BatchNorm statistics stable.
            for module in model.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        total_loss = 0.0
        labels_all, preds_all = [], []
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                labels_all.extend(labels.detach().cpu().tolist())
                preds_all.extend(outputs.argmax(1).detach().cpu().tolist())

        avg_loss = total_loss / max(1, len(loader.dataset))
        acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
        macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0) if labels_all else 0.0
        return avg_loss, acc, macro_f1, labels_all, preds_all

    def train_stage(stage_name, epochs, lr, prefixes, freeze_features_bn):
        nonlocal best_val_acc, best_state, best_epoch
        print(f"\n--- {stage_name} ---", flush=True)

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
        no_improvement = 0

        for epoch in range(1, epochs + 1):
            start = time.time()
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
                train_loader, optimizer, True, freeze_features_bn
            )
            va_loss, va_acc, va_f1, _, _ = run_epoch(val_loader, training=False)
            scheduler.step(va_acc)
            elapsed = time.time() - start

            row = {
                "stage": stage_name, "epoch": epoch,
                "train_loss": tr_loss, "train_accuracy": tr_acc, "train_macro_f1": tr_f1,
                "val_loss": va_loss, "val_accuracy": va_acc, "val_macro_f1": va_f1,
                "seconds": elapsed,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

            print(
                f"Epoch {epoch}/{epochs} | train loss {tr_loss:.4f} | "
                f"train acc {tr_acc*100:.2f}% | val loss {va_loss:.4f} | "
                f"val acc {va_acc*100:.2f}% | val macro-F1 {va_f1:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = len(history)
                torch.save({
                    "model_state_dict": best_state,
                    "class_names": classes,
                    "num_classes": n_classes,
                    "image_size": IMAGE_SIZE,
                    "architecture": "mobilenet_v3_large",
                    "best_val_accuracy": best_val_acc,
                    "best_epoch": best_epoch,
                }, OUTPUT_DIR / "best_geosea_mobilenetv3.pth")
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= PATIENCE:
                print("Early stopping: validation accuracy did not improve.", flush=True)
                break

    train_stage(
        "Stage 1: train classifier head", HEAD_EPOCHS, 1e-3,
        ("classifier.",), freeze_features_bn=True
    )
    train_stage(
        "Stage 2: fine-tune final feature blocks", FINETUNE_EPOCHS, 1e-4,
        ("features.12.", "features.13.", "classifier."), freeze_features_bn=False
    )

    model.load_state_dict(best_state)
    model.eval()
    print("\nEvaluating on the test set...", flush=True)
    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(test_loader, training=False)

    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=classes, digits=4, zero_division=0
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Test loss: {test_loss:.4f}\nTest accuracy: {test_acc*100:.2f}%\n"
        f"Macro-F1: {test_f1:.4f}\n\n{report}", encoding="utf-8"
    )
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(OUTPUT_DIR / "per_class_results.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=list(range(n_classes))),
        display_labels=classes
    ).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(hist_df) + 1), hist_df["train_accuracy"], label="Train accuracy")
    ax.plot(range(1, len(hist_df) + 1), hist_df["val_accuracy"], label="Validation accuracy")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="MobileNetV3-Large Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(hist_df) + 1), hist_df["train_loss"], label="Train loss")
    ax.plot(range(1, len(hist_df) + 1), hist_df["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="MobileNetV3-Large Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)

    final_path = OUTPUT_DIR / "geosea_mobilenetv3_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": classes,
        "num_classes": n_classes,
        "image_size": IMAGE_SIZE,
        "architecture": "mobilenet_v3_large",
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }, final_path)

    # Rough single-image latency estimate on the selected device.
    model.eval()
    sample = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    with torch.no_grad():
        for _ in range(2):
            model(sample)
        start = time.perf_counter()
        repeats = 8
        for _ in range(repeats):
            model(sample)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000 / repeats

    info = {
        "architecture": "MobileNetV3-Large", "device": str(DEVICE),
        "num_classes": n_classes, "class_names": classes,
        "train_images": len(train_ds), "validation_images": len(val_ds),
        "test_images": len(test_ds), "best_validation_accuracy": best_val_acc,
        "best_epoch_in_history": best_epoch, "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "model_size_mb": final_path.stat().st_size / (1024 * 1024),
        "approx_inference_ms_per_image": latency_ms,
    }
    with open(OUTPUT_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("MOBILENETV3-LARGE TRAINING COMPLETE", flush=True)
    print(f"Test accuracy: {test_acc*100:.2f}%", flush=True)
    print(f"Macro-F1: {test_f1:.4f}", flush=True)
    print(f"Model size: {info['model_size_mb']:.2f} MB", flush=True)
    print(f"Approx. inference: {latency_ms:.2f} ms/image", flush=True)
    print(f"Outputs saved in: {OUTPUT_DIR}", flush=True)
    print("Files: best_geosea_mobilenetv3.pth, geosea_mobilenetv3_final.pth,")
    print("classification_report.txt, per_class_results.csv, confusion_matrix.png,")
    print("accuracy_curve.png, loss_curve.png, training_history.csv, model_info.json,")
    print("class_mapping.json")


if __name__ == "__main__":
    main()
