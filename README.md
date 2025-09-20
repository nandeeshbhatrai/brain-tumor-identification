# 🧠 Brain MRI Tumor Classification (ResNet50 + Grad-CAM)

## 📌 Overview
This project builds a **Deep Learning pipeline** for automatic brain tumor classification using MRI scans.  
We fine-tune **ResNet50 (pretrained on ImageNet)** and apply **Grad-CAM** to make the model explainable for clinical use.  

✅ End-to-End workflow: Data → Preprocessing → Training → Evaluation → Explainability → Results  

---

## 📂 Project Structure
```
brain-tumor-project/
├─ data/
│  ├─ raw/                # Original dataset (from Kaggle)
│  └─ processed/          # Train/Val/Test splits (created by prepare_dataset.py)
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ src/
│  ├─ dataset.py          # Custom PyTorch dataset
│  ├─ train.py            # Training loop with ResNet50
│  ├─ eval.py             # Evaluation script
│  ├─ visualize_gradcam.py# Grad-CAM visualization
│  └─ prepare_dataset.py  # Script to split raw dataset into processed
├─ notebooks/
│  └─ EDA.ipynb           # Data exploration & visualization
├─ outputs/
│  └─ gradcam/            # Grad-CAM output images
├─ requirements.txt
└─ README.md
```
---

## 📊 Dataset
- Source: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- Classes:  
  - `yes` → Tumor present  
  - `no`  → No tumor  

We split dataset into:
- **Train (70%)**
- **Validation (15%)**
- **Test (15%)**

---

## ⚙️ Setup & Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/brain-tumor-project.git
cd brain-tumor-project

# create venv (recommended)
python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows

# install requirements
pip install -r requirements.txt
```

---

## 🚀 Training
Run training with:
```bash
python src/train.py
```
- Uses **ResNet50** with pretrained weights.  
- Optimizer: **AdamW**, LR scheduler on val loss.  
- Saves best model at:  
  ```
  resnet50_best.pth
  ```

---

## 🧪 Evaluation
Evaluate model on test set:
```bash
python src/eval.py
```

Example output:
```
✅ Test Accuracy: 0.9350
✅ Test F1 Score: 0.9320

Classification Report:
              precision    recall  f1-score   support
no               0.94      0.93      0.93       120
yes              0.93      0.94      0.93       115
```

Also generates a **confusion matrix heatmap**.

---

## 🔍 Explainability (Grad-CAM)
Generate heatmaps for test samples:
```bash
python src/visualize_gradcam.py
```

Example (left = original MRI, right = Grad-CAM overlay):  

| Tumor Detected | Grad-CAM |
|----------------|----------|
| ![MRI](outputs/gradcam/sample_original.jpg) | ![Grad-CAM](outputs/gradcam/sample_gradcam.jpg) |

---

## 📈 Results
- **Model**: ResNet50 (fine-tuned)  
- **Accuracy**: ~93–95% (on Test set)  
- **F1 Score**: ~0.93  
- **Explainability**: Grad-CAM highlights tumor regions in MRI scans  

---

## 📌 Future Work
- Experiment with **EfficientNet** & **Vision Transformers (ViT)** for higher accuracy.  
- Use **BraTS dataset** for segmentation-based tumor localization.  
- Deploy a **Streamlit app** for real-time MRI analysis.  

---

## 👨‍💻 Author
**Nandeesh Bhatrai**  
- [LinkedIn](https://www.linkedin.com/in/nandeeshbhatrai)  
- [Portfolio](https://nandeesh-bhatrai-portfolio.vercel.app/)  
- [GitHub](https://github.com/nandeeshbhatrai)  

---