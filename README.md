# 🍷 Wine Quality Prediction — ML + Streamlit Web App

> **Predicting premium red wines using chemically engineered features and a Random Forest classifier**

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

<br>

## Overview

This project builds a **binary classifier** that identifies *Premium* red wines (quality score ≥ 7) from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). Rather than feeding all 11 raw lab measurements into the model, **5 chemically meaningful features** are engineered from the raw data — each encoding a principle that winemakers and oenologists use to assess wine quality.

The model is deployed as a **Streamlit web application** where users adjust wine chemical parameters via sliders and receive an instant prediction with a feature-contribution explanation.

**Author:** Kavinda Pushpa Kumara  
**Background:** Food Science Student · IBM Certified Data Scientist

---

## Table of Contents

- [Project Highlights](#project-highlights)
- [Engineered Features](#engineered-features)
- [Model & Methodology](#model--methodology)
- [App Features](#app-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Results](#results)
- [References](#references)

---

## Project Highlights

- **Domain-informed feature engineering** — raw measurements combined into chemically meaningful ratios that capture body, aroma, acidity balance, and preservation
- **Correct SMOTE implementation** — applied *inside* each CV fold using `imblearn.Pipeline`, preventing the data leakage that inflates most tutorial-level models
- **Baseline comparison** — a majority-class `DummyClassifier` is included to give all metrics honest context (raw accuracy is misleading with ~14% premium wines)
- **Precision-recall trade-off analysis** — threshold tuning curve showing how to balance false Premium labels against missed premium wines for different business priorities
- **Transparent deployment** — Streamlit app shows per-prediction feature contributions alongside global model importances, with a full methodology explanation

---

## Engineered Features

| Feature | Formula | Winemaking Principle |
|---|---|---|
| `alcohol_density_ratio` | alcohol ÷ density | **Body & mouthfeel** — higher ratio = fuller-bodied wine |
| `flavor_intensity` | sulphates × alcohol | **Aroma complexity** — sulphates protect volatile compounds extracted by alcohol |
| `acidity_quality` | pH × volatile acidity | **Fault detection** — penalises high volatile acidity (vinegar off-flavours) |
| `sugar_acid_balance` | residual sugar ÷ fixed acidity | **Sensory sweetness** — sweetness perception relative to acid backbone |
| `so2_efficiency` | free SO₂ ÷ alcohol | **Preservation efficiency** — normalised SO₂; optimal range avoids oxidation and over-sulfiting |

> `sugar_acid_balance` has a weak marginal correlation with quality (≈ −0.03) but is retained for its established role in winemaking sensory science and its potential interaction effects in the tree model.

---

## Model & Methodology

**Algorithm:** Random Forest Classifier  
**Dataset:** UCI Red Wine Quality — 1,599 Portuguese Vinho Verde wines  
**Class imbalance:** ~14% premium wines → addressed with SMOTE + `class_weight='balanced'`

```
Raw data
   └── Train / Test split (80/20, stratified)
         └── Training set only
               ├── Feature selection analysis (correlation + RF importance)
               ├── ImbPipeline: StandardScaler → SMOTE → RandomForestClassifier
               ├── GridSearchCV (5-fold CV, scoring = F1)
               └── Final evaluation on held-out test set
```

**Key design decisions:**
- Feature selection correlation and importance analysis run on **training data only** — the test set is never seen until final evaluation
- `imblearn.Pipeline` ensures SMOTE synthetic samples are generated per fold, not across the full training set
- Model is evaluated on **F1 score and AUC-ROC**, not accuracy, due to class imbalance

---

## App Features

| Feature | Description |
|---|---|
| Interactive sliders | 5 engineered feature inputs with dataset-mean defaults |
| Instant prediction | Premium / Non-Premium with confidence percentage |
| Decision explanation | Per-prediction feature contribution chart (importance × scaled value) |
| Global importances | RF feature importance bar chart from the trained model |
| Scientific methodology | Full winemaking rationale for each engineered feature |
| Model transparency | Dataset, algorithm, limitations, and threshold guidance disclosed |

---

## Repository Structure

```
wine-quality-prediction/
│
├── app.py                    # Streamlit web application
├── requirements.txt          # Pinned dependencies
├── README.md                 # This file
│
├── notebook/
│   └── wine_quality_prediction.ipynb   # Full analysis notebook
│
├── models/                   # Place your .pkl files here
│   ├── wine_model.pkl        # Trained RandomForestClassifier
│   └── scaler.pkl            # Fitted StandardScaler
│
└── assets/
    └── screenshots/          # App screenshots (optional)
```

> **Note:** `.pkl` files are not committed to the repository. Generate them by running the notebook end-to-end, then place them in the `models/` folder (update the paths in `app.py` accordingly).

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the model files

Run the Jupyter notebook end-to-end:

```bash
jupyter notebook notebook/wine_quality_prediction.ipynb
```

The final cell saves `wine_model.pkl` and `scaler.pkl`. Move them to the project root (same folder as `app.py`).

### 5. Launch the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Saving Model Files from the Notebook

The notebook saves the scaler and Random Forest **separately** so the Streamlit app can use them as independent steps:

```python
scaler_standalone = best_pipe.named_steps['scaler']
rf_standalone     = best_pipe.named_steps['clf']

joblib.dump(rf_standalone,     'wine_model.pkl')
joblib.dump(scaler_standalone, 'scaler.pkl')
```

---

## Results

> *Exact values will vary slightly depending on random seed and environment. Run the notebook to reproduce.*

| Model | Accuracy | F1 (Premium) | AUC-ROC |
|---|---|---|---|
| Baseline (majority class) | ~86% | 0.00 | 0.50 |
| 5 Raw features | — | — | — |
| **5 Engineered features** | — | — | — |

The baseline row illustrates why accuracy is not a useful metric here — a classifier that always predicts "Non-Premium" scores 86% by doing nothing. F1 and AUC-ROC are the meaningful measures.

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository and set the main file to `app.py`
4. Add your `.pkl` files to the repo or configure cloud storage
5. Click **Deploy** — you'll receive a shareable public URL

---

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553.
- Peynaud, E. (1987). *Knowing and Making Wine.* Wiley.
- OIV (2023). *International Code of Oenological Practices.* International Organisation of Vine and Wine.
- UCI Machine Learning Repository: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
