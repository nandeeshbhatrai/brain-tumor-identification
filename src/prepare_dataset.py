import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
RAW_DATASET_DIR = r"D:\Developments\Resume projects\brain-tumor-project\data\raw\archive\brain_tumor_dataset"
PROCESSED_DIR = r"D:\Developments\Resume projects\brain-tumor-project\data\processed"

# Ensure processed dirs
for split in ["train", "val", "test"]:
    for cls in ["yes", "no"]:
        os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

# Collect image paths
data = []
labels = []
for cls in ["yes", "no"]:
    cls_dir = os.path.join(RAW_DATASET_DIR, cls)
    images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('.jpg','.png','.jpeg'))]
    data.extend(images)
    labels.extend([cls]*len(images))

# Stratified split (70/15/15)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=42
)

def copy_files(paths, labels, split):
    for path, label in zip(paths, labels):
        fname = os.path.basename(path)
        dest = os.path.join(PROCESSED_DIR, split, label, fname)
        shutil.copy(path, dest)

copy_files(train_paths, train_labels, "train")
copy_files(val_paths, val_labels, "val")
copy_files(test_paths, test_labels, "test")

print("✅ Dataset prepared in:", PROCESSED_DIR)
