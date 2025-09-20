import os
import cv2
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from dataset import BrainTumorDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

TEST_DIR = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\test"
MODEL_PATH = r"D:\Developments\Resume projects\brain-tumor-project\resnet50_best.pth"
OUTPUT_DIR = r"D:\Developments\Resume projects\brain-tumor-project\outputs\gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Transforms (for inference)
# -----------------------------
test_transforms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# -----------------------------
# Dataset
# -----------------------------
dataset = BrainTumorDataset(TEST_DIR, transform=test_transforms)

# -----------------------------
# Load trained model
# -----------------------------
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Grad-CAM Setup
# -----------------------------
# For ResNet50, last conv layer = model.layer4[-1]
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

# -----------------------------
# Visualize Grad-CAM for samples
# -----------------------------
def visualize_gradcam(index=0):
    """Generate Grad-CAM heatmap for dataset[index]."""
    image, label = dataset[index]
    img_path, _ = dataset.samples[index]
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img_resized = cv2.resize(orig_img, (224,224)) / 255.0

    input_tensor = image.unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    visualization = show_cam_on_image(orig_img_resized.astype(np.float32), grayscale_cam, use_rgb=True)

    # Save
    fname = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, f"gradcam_{fname}")
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # Show
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(orig_img_resized)
    plt.title(f"Original (Label: {label})")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()

    print(f"✅ Saved Grad-CAM at {out_path}")

# -----------------------------
# Run on first 5 samples
# -----------------------------
if __name__ == "__main__":
    for i in range(5):
        visualize_gradcam(i)
