import torch
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
MODEL_PATH = r"D:\Developments\Resume projects\brain-tumor-project\resnet50_best.pth"

# -----------------------------
# Load model
# -----------------------------
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Preprocessing pipeline
# -----------------------------
transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class_names = ["no", "yes"]

# -----------------------------
# Inference function
# -----------------------------
def predict(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    transformed = transform(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
    
    print(f"Image: {image_path}")
    print(f"Predicted: {class_names[pred_class]} (Confidence: {probs[pred_class]:.4f})")

    return pred_class, probs.cpu().numpy()

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_img = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\test\yes\Y1.jpg"
    predict(test_img)
