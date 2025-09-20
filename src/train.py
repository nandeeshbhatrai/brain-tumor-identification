import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BrainTumorDataset

# -----------------------------
# Warnings
# -----------------------------
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 2

TRAIN_DIR = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\train"
VAL_DIR   = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\val"
MODEL_PATH = r"D:\Developments\Resume projects\brain-tumor-project\resnet50_best.pth"

# -----------------------------
# Data transforms
# -----------------------------
train_transforms = A.Compose([
    A.Resize(224,224),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=20, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# -----------------------------
# Datasets & Loaders
# -----------------------------
train_ds = BrainTumorDataset(TRAIN_DIR, transform=train_transforms)
val_ds   = BrainTumorDataset(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -----------------------------
# Model: ResNet50 (timm)
# -----------------------------
model = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# -----------------------------
# Training + Validation Loops
# -----------------------------
if __name__ == "__main__":
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        # ---- Training ----
        model.train()
        train_loss = 0
        y_true, y_pred = [], []

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average="macro")
        avg_train_loss = train_loss / len(train_ds)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        avg_val_loss = val_loss / len(val_ds)

        scheduler.step(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved new best model at {MODEL_PATH} (Val Acc: {val_acc:.4f})")

    # -----------------------------
    # Final report
    # -----------------------------
    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("Classification Report on last epoch:")
    print(classification_report(y_true, y_pred, target_names=["no", "yes"]))
