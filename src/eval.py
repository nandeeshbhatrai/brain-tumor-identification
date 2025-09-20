import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BrainTumorDataset

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 2

TEST_DIR = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\test"
MODEL_PATH = r"D:\Developments\Resume projects\brain-tumor-project\resnet50_best.pth"

# -----------------------------
# Data transforms (only resize + normalize)
# -----------------------------
test_transforms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# -----------------------------
# Dataset & Loader
# -----------------------------
test_ds = BrainTumorDataset(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -----------------------------
# Load model
# -----------------------------
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Evaluation
# -----------------------------
if __name__ == "__main__":
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["no", "yes"]))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["no","yes"], yticklabels=["no","yes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.show()
