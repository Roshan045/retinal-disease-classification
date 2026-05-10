# Retinal Disease Classification

A notebook using retinal image scan data to detect ocular disease across eight classes.

---

## Dataset

**ODIR-5K — Ocular Disease Intelligent Recognition**
Available on Kaggle: [Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

> The dataset is not included in this repository. To reproduce results, add the dataset as a Kaggle input (instructions below).

**Diseases in data:**

| Label | Disease |
|---|---|
| N | Normal |
| D | Diabetes |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration (AMD) |
| H | Hypertension |
| M | Myopia |
| O | Other |

The dataset is imbalanced and is addressed during the project.

---

## Approach

### Data preparation
- Dropped non-predictive columns (patient ID, diagnostic keywords, left/right fundus labels)
- Expanded the multi-label target column into individual disease columns
- Removed duplicates and missing values
- Conducted EDA
- Stratified 70/15/15 train/validation/test split on `disease_position` to preserve class ratios across all splits

### Imbalance handling
- Calculated per-class weights using `sklearn.utils.class_weight.compute_class_weight`
- Applied class weights during model training to prevent bias towards majority classes
- Used data augmentation on training images 

### Models
Three pretrained architectures were fine-tuned and compared:

| Model | 
|---|
| EfficientNetB0 | 
| ResNet50 | 
| VGG19 | 

Each model used `GlobalAveragePooling2D` → `Dense` → `Dropout` → 8-class `softmax` output. Training used `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks.

### Evaluation
- Macro and weighted F1-score, precision, recall
- Per-class confusion matrices
- Classification report across all 8 classes

---

## Environment

This notebook was developed on **Kaggle** using a GPU P100 accelerator and the ODIR-5K dataset added as a Kaggle input.

**To reproduce on Kaggle:**
1. Upload the notebook to [Kaggle](https://www.kaggle.com)
2. Add the [Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) dataset as an input
3. In Settings, enable the **GPU P100** accelerator (phone verification may be required)
4. Run all cells

**Dependencies:** `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `opencv-python`
