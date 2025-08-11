# 🩺 Multi-Model Skin Cancer Classification

This project uses **deep learning** to classify skin lesions from dermoscopic images into multiple categories, aiming to support **early detection of melanoma** and other skin conditions.

##  Project Overview
- **Goal:** Classify dermoscopic images into lesion types such as melanoma, basal cell carcinoma, benign keratosis, etc.
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
- **Model:** EfficientNet-B0 (transfer learning with `timm`)
- **Frameworks:** PyTorch, Torchvision
- **Application:** Early melanoma detection, skin lesion screening

---

## 📂 Dataset
The **HAM10000** dataset contains 10,015 dermoscopic images across 7 diagnostic categories:
- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis-like lesions
- Dermatofibroma
- Melanoma
- Melanocytic nevi
- Vascular lesions

The dataset is class-imbalanced, so **class-weighted loss** is used during training.

---

##  Methodology
1. **Data Preprocessing**
   - Resized images to `224x224`
   - Normalized using ImageNet statistics
   - Applied augmentations: random crops, rotations, flips, color jitter

2. **Model**
   - Pretrained **EfficientNet-B0** from `timm`
   - Final classification layer adjusted for 7 classes
   - Loss: Weighted CrossEntropyLoss
   - Optimizer: Adam, learning rate `3e-5`

3. **Training**
   - 20 epochs
   - GPU: Kaggle T4
   - Best model saved based on validation accuracy




