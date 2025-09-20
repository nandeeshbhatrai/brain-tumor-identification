import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with images for a split (e.g. data/processed/train).
            transform (albumentations.Compose): Transformations to apply.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Classes are ['no', 'yes']
        self.class_to_idx = {"no": 0, "yes": 1}

        for cls in ["no", "yes"]:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # ensure 3 channels

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # Paths (adjust if needed)
    train_dir = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\train"
    val_dir   = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\val"
    test_dir  = r"D:\Developments\Resume projects\brain-tumor-project\data\processed\test"

    # Define transforms
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

    # Create datasets
    train_ds = BrainTumorDataset(train_dir, transform=train_transforms)
    val_ds   = BrainTumorDataset(val_dir, transform=val_transforms)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    # Quick test
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels}")
