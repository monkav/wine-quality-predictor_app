"""
Wine Quality Prediction — Streamlit Application
Author : Kavinda Pushpa Kumara
Role   : Food Science Student | IBM Certified Data Scientist

Exact model metrics from notebook (wine_quality_prediction_eng_feat.ipynb):
  Accuracy  : 91.56%
  F1 Score  : 0.710
  AUC-ROC   : 0.951
  Avg Prec  : from notebook cell 8
  Baseline  : 86.4% accuracy, 0.00 F1
  Raw model : 90.00% accuracy, 0.686 F1, 0.927 AUC
  Dataset   : 1599 wines, 14.0% premium (quality >= 7)
  Train/Test: 80/20 stratified split
  CV        : 5-fold, scored on F1
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=32&h=32&fit=crop",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Image URLs (Unsplash, free licence) ───────────────────────────────────────
IMG = {
    "hero":     "https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=1600&q=80&fit=crop&crop=center",
    "pour":     "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80&fit=crop&crop=center",
    "cellar":   "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80&fit=crop&crop=center",
    "lab":      "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=800&q=80&fit=crop&crop=center",
    "vineyard": "https://images.unsplash.com/photo-1464207687429-7505649dae38?w=800&q=80&fit=crop&crop=center",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
    -webkit-text-size-adjust: 100%;
    text-size-adjust: 100%;
}
.stApp { background: #faf7f2; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Safe area for notched phones */
.stApp > div:first-child { padding-left: env(safe-area-inset-left); padding-right: env(safe-area-inset-right); }

/* ── Hero ── */
.hero-banner {
    position: relative; width: 100%; height: 160px;
    overflow: hidden; background: #1a0a0e;
}
.hero-banner img {
    width: 100%; height: 100%; object-fit: cover;
    object-position: center 60%; opacity: 0.5; display: block;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(15,4,6,0.88) 0%, rgba(15,4,6,0.5) 55%, rgba(15,4,6,0.1) 100%);
    display: flex; align-items: center;
    padding: 0 3rem; gap: 2rem;
}
.hero-text { flex: 1; min-width: 0; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.3rem, 2.8vw, 2rem);
    font-weight: 700; color: #fff;
    letter-spacing: 0.01em; line-height: 1.2; margin: 0 0 0.3rem;
}
.hero-rule { width: 44px; height: 2px; background: #8b1a2f; margin: 0.35rem 0; }
.hero-author {
    font-size: 0.75rem; color: rgba(255,255,255,0.65);
    letter-spacing: 0.1em; text-transform: uppercase;
}
.hero-stats {
    display: flex; gap: 1rem; flex-shrink: 0;
}
.hero-stat {
    background: rgba(139,26,47,0.75);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 0.5rem 0.9rem;
    text-align: center; color: #fff;
}
.hero-stat strong {
    display: block;
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 0;
}
.hero-stat span {
    font-size: 0.62rem; letter-spacing: 0.07em;
    text-transform: uppercase; opacity: 0.8;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-bottom: 1px solid #e8ddd5;
    padding: 0 2.5rem; gap: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Lato', sans-serif;
    font-size: 0.79rem; font-weight: 500;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: #9b8c84; padding: 0.85rem 1.5rem;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: #8b1a2f !important;
    border-bottom: 2px solid #8b1a2f !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Tab content ── */
.tab-content {
    padding: 1.2rem 3rem 1.5rem;
    max-width: 1320px; margin: 0 auto; width: 100%;
    box-sizing: border-box;
}

/* ── Input panel ── */
.input-panel {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 12px;
    padding: 1.1rem 1.5rem 0.6rem;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
    height: 100%;
    box-sizing: border-box;
}
.panel-header {
    display: flex; align-items: center;
    gap: 0.85rem; margin-bottom: 0.7rem;
    padding-bottom: 0.65rem; border-bottom: 1px solid #f0ebe4;
}
.panel-header img {
    width: 50px; height: 50px;
    border-radius: 7px; object-fit: cover; flex-shrink: 0;
}
.ph-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.97rem; font-weight: 600; color: #2c2118;
}
.ph-sub {
    font-size: 0.68rem; color: #9b8c84;
    letter-spacing: 0.05em; text-transform: uppercase; margin-top: 0.1rem;
}

/* ── Sliders ── */
.stSlider > label {
    font-size: 0.76rem !important; color: #5c4a3a !important;
    font-weight: 500 !important; letter-spacing: 0.02em;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #8b1a2f !important;
}

/* ── Button ── */
.stButton > button {
    background: #8b1a2f !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.84rem !important; font-weight: 500 !important;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.6rem 0 !important; width: 100%;
    transition: background 0.2s !important; margin-top: 0.4rem;
    min-height: 44px;
}
.stButton > button:hover { background: #a02038 !important; }

/* ── Result cards ── */
.result-premium {
    background: linear-gradient(135deg, #fff5f6 0%, #fff 100%);
    border: 2px solid #8b1a2f; border-radius: 12px;
    padding: 1.2rem 1.4rem; text-align: center;
    margin-bottom: 0.8rem; position: relative; overflow: hidden;
}
.result-premium::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b1a2f, #c9406a, #8b1a2f);
}
.result-standard {
    background: #fff; border: 2px solid #d0c8c0;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    text-align: center; margin-bottom: 0.8rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem; font-weight: 700;
    letter-spacing: 0.03em; margin-bottom: 0.2rem; line-height: 1.1;
}
.result-prob { font-size: 0.8rem; color: #9b8c84; }
.result-prob strong { color: #2c2118; }

/* ── Gauge ── */
.gauge-wrap { margin: 0.6rem 0; }
.gauge-label {
    display: flex; justify-content: space-between;
    font-size: 0.65rem; color: #9b8c84;
    letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.gauge-track {
    background: #f0ebe4; border-radius: 99px; height: 7px; overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 99px; }

/* ── Feature chips ── */
.feat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.4rem; margin-top: 0.4rem;
}
.feat-chip {
    background: #faf7f2; border: 1px solid #e8ddd5;
    border-radius: 7px; padding: 0.35rem 0.6rem;
}
.feat-chip .fname {
    color: #9b8c84; font-size: 0.63rem; text-transform: uppercase;
    letter-spacing: 0.05em; display: block; margin-bottom: 0.05rem;
}
.feat-chip .fval {
    color: #2c2118; font-weight: 600;
    font-family: monospace; font-size: 0.78rem;
}

/* ── Section labels ── */
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem; font-weight: 600;
    color: #2c2118; margin: 0 0 0.18rem;
}
.sec-sub {
    font-size: 0.71rem; color: #9b8c84;
    margin-bottom: 0.6rem; line-height: 1.45;
}

/* ── Explanation panel ── */
.explanation-panel {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    margin-top: 0.7rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Metrics strip ── */
.metrics-strip {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 0.6rem; margin-bottom: 1rem;
}
.metric-card {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 10px; padding: 0.75rem 0.9rem;
    text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    border-top: 3px solid #8b1a2f;
}
.metric-card.secondary { border-top-color: #b0a8a0; }
.mc-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9b8c84;
    display: block; margin-bottom: 0.2rem;
}
.mc-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700; color: #8b1a2f; line-height: 1;
}
.mc-val.secondary { color: #5c4a3a; }
.mc-note {
    font-size: 0.62rem; color: #b0a8a0;
    margin-top: 0.2rem; display: block;
}

/* ── Comparison table ── */
.cmp-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.8rem; margin-top: 0.5rem;
}
.cmp-table th {
    background: rgba(139,26,47,0.08); color: #8b1a2f;
    padding: 0.5rem 0.8rem; text-align: left;
    font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; font-size: 0.68rem;
    border-bottom: 2px solid #e8ddd5;
}
.cmp-table td {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid #f0ebe4;
    color: #5c4a3a;
}
.cmp-table tr:hover td { background: rgba(0,0,0,0.012); }
.cmp-table .best { color: #4a7c59; font-weight: 600; }
.cmp-table .highlight-row td { background: #fdf8f8; }

/* ── Methodology ── */
.meth-hero {
    position: relative; width: 100%; height: 110px;
    border-radius: 10px; overflow: hidden; margin-bottom: 1rem;
}
.meth-hero img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center 40%; opacity: 0.6;
}
.meth-hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(15,4,6,0.8) 0%, rgba(15,4,6,0.15) 100%);
    display: flex; align-items: center; padding: 0 1.6rem;
}
.meth-hero-text h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem; font-weight: 700; color: #fff; margin: 0 0 0.15rem;
}
.meth-hero-text p {
    font-size: 0.73rem; opacity: 0.8; margin: 0; color: #fff;
    max-width: 500px; line-height: 1.4;
}
.meth-grid {
    display: grid;
    grid-template-columns: repeat(5,1fr);
    gap: 0.7rem;
}
.meth-card {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 9px; padding: 0.85rem 0.95rem;
    border-top: 3px solid #8b1a2f;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.mc-tag {
    font-family: monospace; font-size: 0.67rem;
    background: #fdf0f2; color: #8b1a2f;
    padding: 0.1rem 0.38rem; border-radius: 3px;
    display: inline-block; margin-bottom: 0.3rem;
}
.mc-formula {
    font-size: 0.72rem; color: #9b8c84;
    font-family: monospace; margin-bottom: 0.3rem;
}
.mc-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.88rem; font-weight: 600;
    color: #2c2118; margin-bottom: 0.25rem;
}
.mc-body { font-size: 0.74rem; color: #5c4a3a; line-height: 1.5; }
.dir-good  { color: #4a7c59; font-size: 0.68rem; font-weight: 500; margin-top: 0.28rem; }
.dir-bad   { color: #8b1a2f; font-size: 0.68rem; font-weight: 500; margin-top: 0.28rem; }
.dir-range { color: #7a6a30; font-size: 0.68rem; font-weight: 500; margin-top: 0.28rem; }

/* ── Insights tab ── */
.img-strip {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.7rem; margin-bottom: 1rem;
}
.img-card {
    position: relative; border-radius: 9px;
    overflow: hidden; height: 80px;
}
.img-card img { width:100%; height:100%; object-fit:cover; opacity:0.65; }
.img-label {
    position: absolute; bottom:0; left:0; right:0;
    background: linear-gradient(0deg,rgba(15,4,6,0.78) 0%,transparent 100%);
    padding: 0.35rem 0.65rem; font-size: 0.66rem; color:#fff;
    letter-spacing: 0.07em; text-transform: uppercase; font-weight:500;
}
.transp-box {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.77rem; color: #5c4a3a; line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.transp-box h4 {
    font-family: 'Playfair Display', serif;
    font-size: 0.88rem; color: #2c2118;
    margin: 0.65rem 0 0.15rem;
}
.transp-box h4:first-child { margin-top: 0; }

/* ── Footer ── */
.app-footer {
    background: #fff; border-top: 1px solid #e8ddd5;
    padding: 0.65rem 1rem; text-align: center;
    font-size: 0.68rem; color: #c8bdb8; letter-spacing: 0.05em;
}

/* ── Streamlit columns mobile stacking ── */
[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
    gap: 1rem;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 320px;
    min-width: 280px;
}

/* ── Make tables scrollable ── */
.cmp-table { display: block; overflow-x: auto; white-space: nowrap; }

/* ── Mobile ── */
@media (max-width: 900px) {
    .tab-content { padding: 1rem 1.5rem; }
    .hero-overlay { padding: 0 1.5rem; }
    .meth-grid { grid-template-columns: repeat(3,1fr); }
}
@media (max-width: 768px) {
    .hero-banner { height: 200px; }
    .hero-overlay {
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        padding: 1rem 1.2rem;
        gap: 0.8rem;
        background: linear-gradient(180deg, rgba(15,4,6,0.9) 0%, rgba(15,4,6,0.6) 100%);
    }
    .hero-stats {
        display: flex;
        width: 100%;
        gap: 0.5rem;
    }
    .hero-stat {
        flex: 1;
        padding: 0.4rem 0.5rem;
        border-radius: 6px;
    }
    .hero-stat strong { font-size: 1.1rem; }
    .stTabs [data-baseweb="tab-list"] {
        padding: 0 0.5rem;
        overflow-x: auto;
        white-space: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] { padding: 0.8rem 1rem; font-size: 0.75rem; }
    .tab-content { padding: 1rem 1rem 1.2rem; }
    .metrics-strip { grid-template-columns: repeat(2,1fr); gap: 0.5rem; }
    .meth-grid { grid-template-columns: 1fr 1fr; }
    .img-strip { grid-template-columns: 1fr; }
    .app-footer { padding: 0.65rem 1rem; font-size: 0.65rem; }
}
@media (max-width: 480px) {
    .hero-banner { height: 220px; }
    .hero-overlay { padding: 0.9rem 1rem; }
    .hero-title { font-size: clamp(1.15rem, 5vw, 1.4rem); }
    .hero-author { font-size: 0.68rem; }
    .tab-content { padding: 0.85rem 0.9rem 1.1rem; }
    .input-panel { padding: 0.9rem 1rem 0.5rem; border-radius: 10px; }
    .panel-header img { width: 44px; height: 44px; }
    .metrics-strip { grid-template-columns: 1fr 1fr; gap: 0.4rem; }
    .mc-val { font-size: 1.3rem; }
    .meth-grid { grid-template-columns: 1fr; }
    .feat-grid { grid-template-columns: 1fr; }
    .stTabs [data-baseweb="tab"] { padding: 0.7rem 0.8rem; font-size: 0.7rem; }
    .result-label { font-size: 1.6rem; }
    .stButton > button { font-size: 0.8rem !important; min-height: 48px; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100%;
        min-width: 100%;
    }
}

/* ── Mobile polish fixes ── */
html, body { overflow-x: hidden; }
.block-container { padding-left: 0 !important; padding-right: 0 !important; }

/* Ensure tab content has safe side padding on phones */
@media (max-width: 768px) {
  .tab-content { padding: 0.9rem 0.9rem 1.2rem !important; }
  .input-panel { padding: 0.9rem 0.9rem 0.6rem !important; border-radius: 10px; }
  /* Give sliders breathing room */
  [data-testid="stSlider"] { padding: 0 0.1rem; }
  .stSlider > label { margin-bottom: 0.25rem !important; }
  /* Prevent edge-to-edge */
  [data-testid="column"] { padding: 0 0.1rem; }
  /* Bottom safe area so button isn't hidden */
  .main .block-container { padding-bottom: 90px !important; }
  /* Hide Streamlit cloud floating buttons */
  [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
  .stAppDeployButton, button[kind="header"], div[data-testid="stAppViewBlockContainer"] > div > div > button {
    display: none !important;
  }
  /* Make Analyse button sticky and full width */
  .stButton > button {
    position: sticky;
    bottom: 12px;
    z-index: 999;
    box-shadow: 0 4px 12px rgba(139,26,47,0.25);
  }
}

/* Tighter spacing for very small screens */
@media (max-width: 480px) {
  .tab-content { padding: 0.75rem 0.75rem 1rem !important; }
  .input-panel { padding: 0.8rem 0.8rem 0.5rem !important; }
  .stSlider > label { font-size: 0.72rem !important; }
}

</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}\n\nPlace wine_model.pkl and scaler.pkl in the same folder as app.py.")
        st.stop()

model, scaler = load_model()

FEATURE_NAMES = [
    "alcohol_density_ratio",
    "flavor_intensity",
    "acidity_quality",
    "sugar_acid_balance",
    "so2_efficiency",
]
importances = model.feature_importances_

# ── Exact metrics from notebook ───────────────────────────────────────────────
MODEL_METRICS = {
    "accuracy":   0.9156,
    "f1":         0.710,
    "auc_roc":    0.951,
    "baseline_acc": 0.864,
    "raw_acc":    0.9000,
    "raw_f1":     0.686,
    "raw_auc":    0.927,
    "n_train":    1279,
    "n_test":     320,
    "n_total":    1599,
    "pct_premium": 14.0,
    "cv_folds":   5,
}

# ── Dataset means (UCI Red Wine) ──────────────────────────────────────────────
MEANS = {
    "alcohol":             10.42,
    "density":              0.9967,
    "sulphates":            0.658,
    "pH":                   3.311,
    "volatile_acidity":     0.528,
    "residual_sugar":       2.539,
    "fixed_acidity":        8.32,
    "free_sulfur_dioxide": 15.87,
}


def engineer_features(alcohol, density, sulphates, pH, volatile_acidity,
                       residual_sugar, fixed_acidity, free_sulfur_dioxide):
    return {
        "alcohol_density_ratio": alcohol / density,
        "flavor_intensity":      sulphates * alcohol,
        "acidity_quality":       pH * volatile_acidity,
        "sugar_acid_balance":    residual_sugar / (fixed_acidity + 1e-6),
        "so2_efficiency":        free_sulfur_dioxide / (alcohol + 1e-6),
    }


def make_prediction(feat_dict):
    df     = pd.DataFrame([feat_dict])
    scaled = scaler.transform(df)
    prob   = model.predict_proba(scaled)[0, 1]
    label  = 1 if prob >= 0.5 else 0
    return label, prob, scaled[0]


def get_food_science_notes(feats):
    """Return food science interpretation of engineered feature values."""
    notes = []
    adr = feats["alcohol_density_ratio"]
    fi  = feats["flavor_intensity"]
    aq  = feats["acidity_quality"]
    so2 = feats["so2_efficiency"]
    sab = feats["sugar_acid_balance"]

    if adr > 12.5:
        notes.append(("Body", "Full-bodied — high alcohol-density ratio indicates strong extract and palate weight.", "good"))
    elif adr < 11.0:
        notes.append(("Body", "Light-bodied — low alcohol-density ratio; may lack structure.", "warn"))
    else:
        notes.append(("Body", f"Medium body (ratio {adr:.2f}) — within typical range for quality reds.", "neutral"))

    if fi > 9.0:
        notes.append(("Aroma", "High aroma complexity — strong sulphates × alcohol product; good aromatic richness.", "good"))
    elif fi < 6.0:
        notes.append(("Aroma", "Low flavour intensity — may lack aromatic complexity; consider sulphate levels.", "warn"))
    else:
        notes.append(("Aroma", f"Moderate flavour intensity ({fi:.2f}) — adequate aromatic presence.", "neutral"))

    if aq < 2.0:
        notes.append(("Acidity", "Clean acidity balance — low volatile acidity; no vinegar off-flavours expected.", "good"))
    elif aq > 3.0:
        notes.append(("Acidity", "High volatile acidity risk — acidity quality index above 3.0; vinegar notes may be detectable.", "warn"))
    else:
        notes.append(("Acidity", f"Acceptable acidity ({aq:.2f}) — borderline; monitor volatile acidity.", "neutral"))

    if 1.5 <= so2 <= 3.5:
        notes.append(("Preservation", f"Optimal SO2 efficiency ({so2:.2f}) — good antimicrobial and antioxidant protection.", "good"))
    elif so2 < 1.5:
        notes.append(("Preservation", "Low SO2 efficiency — oxidation or microbial spoilage risk.", "warn"))
    else:
        notes.append(("Preservation", "High SO2 efficiency — potential sulfurous off-aromas detectable by consumers.", "warn"))

    if sab < 0.3:
        notes.append(("Balance", "Well-balanced — low residual sugar relative to acidity; typical of quality dry reds.", "good"))
    elif sab > 0.8:
        notes.append(("Balance", "Higher sweetness ratio — excess residual sugar relative to fixed acidity.", "warn"))
    else:
        notes.append(("Balance", f"Moderate sugar-acid balance ({sab:.3f}) — within expected range for dry reds.", "neutral"))

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <img src="{IMG['hero']}" alt="Vineyard at golden hour">
    <div class="hero-overlay">
        <div class="hero-text">
            <div class="hero-title">Wine Quality Predictor</div>
            <div class="hero-rule"></div>
            <div class="hero-author">
                Kavinda Pushpa Kumara &nbsp;&middot;&nbsp;
                Food Science Student &nbsp;&middot;&nbsp;
                IBM Certified Data Scientist
            </div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <strong>91.6%</strong>
                <span>Accuracy</span>
            </div>
            <div class="hero-stat">
                <strong>0.951</strong>
                <span>AUC-ROC</span>
            </div>
            <div class="hero-stat">
                <strong>1,599</strong>
                <span>Wines Trained</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_insights, tab_method = st.tabs([
    "  Predict  ",
    "  Model Insights  ",
    "  Methodology  ",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    col_inputs, col_result = st.columns([1, 1], gap="large")

    # ── Left: raw inputs ───────────────────────────────────────────────────────
    with col_inputs:
        st.markdown(f"""
        <div class="input-panel">
            <div class="panel-header">
                <img src="{IMG['pour']}" alt="Wine being poured">
                <div>
                    <div class="ph-title">Raw Chemical Measurements</div>
                    <div class="ph-sub">Enter lab values &mdash; features are engineered automatically</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0,
                                float(MEANS["alcohol"]), 0.1,
                                help="Percentage alcohol by volume")
            sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00,
                                  float(MEANS["sulphates"]), 0.01,
                                  help="Potassium sulphate — preservative and aroma enhancer")
            pH = st.slider("pH", 2.70, 4.50,
                           float(MEANS["pH"]), 0.01,
                           help="Acidity level — lower = more acidic")
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0,
                                       float(MEANS["residual_sugar"]), 0.1,
                                       help="Unfermented sugar remaining after fermentation")
        with c2:
            density = st.slider("Density (g/cm³)", 0.990, 1.004,
                                float(MEANS["density"]), 0.0001,
                                format="%.4f",
                                help="Wine density — decreases as alcohol increases")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60,
                                         float(MEANS["volatile_acidity"]), 0.01,
                                         help="Acetic acid — high levels produce a vinegar taste")
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0,
                                      float(MEANS["fixed_acidity"]), 0.1,
                                      help="Tartaric acid — structural backbone of the wine")
            free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0,
                                 float(MEANS["free_sulfur_dioxide"]), 0.5,
                                 help="Free sulfur dioxide — prevents oxidation and microbial growth")

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyse Wine", use_container_width=True)

    # ── Right: result ──────────────────────────────────────────────────────────
    with col_result:
        feats = engineer_features(alcohol, density, sulphates, pH,
                                   volatile_acidity, residual_sugar,
                                   fixed_acidity, free_so2)

        if predict_btn:
            # store result in session state so explanation button can appear
            label, prob, scaled_vals = make_prediction(feats)
            st.session_state["last_label"]       = label
            st.session_state["last_prob"]        = prob
            st.session_state["last_scaled"]      = scaled_vals
            st.session_state["last_feats"]       = feats
            st.session_state["show_explanation"] = False  # reset on new prediction

        if "last_prob" in st.session_state:
            label       = st.session_state["last_label"]
            prob        = st.session_state["last_prob"]
            scaled_vals = st.session_state["last_scaled"]
            feats_saved = st.session_state["last_feats"]

            # ── Result card ──
            if label == 1:
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label" style="color:#8b1a2f;">Premium</div>
                    <div class="result-prob">Quality predicted &ge; 7</div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence &nbsp;<strong style="color:#2c2118">{prob*100:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{prob*100:.1f}%;background:#8b1a2f;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("Chemical profile meets premium criteria. Strong body, aromatic complexity, and controlled acidity contribute positively.")
            else:
                conf = (1 - prob) * 100
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label" style="color:#5c4a3a;">Non-Premium</div>
                    <div class="result-prob">Quality predicted &lt; 7</div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence &nbsp;<strong style="color:#2c2118">{conf:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{prob*100:.1f}%;background:#b0a8a0;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("Profile does not meet premium threshold. Volatile acidity and flavour intensity are the primary levers for improvement.")

            # ── Explain button — only appears after a prediction ──
            if st.button("Why did the model decide this?", use_container_width=True):
                st.session_state["show_explanation"] = not st.session_state.get("show_explanation", False)

            if st.session_state.get("show_explanation", False):
                st.markdown('<div class="explanation-panel">', unsafe_allow_html=True)

                # Feature contribution chart
                st.markdown('<p class="sec-title">Feature Contributions</p>', unsafe_allow_html=True)
                st.markdown('<p class="sec-sub">Importance &times; scaled value. Burgundy = pushes toward Premium; grey = toward Non-Premium.</p>', unsafe_allow_html=True)

                contributions = importances * scaled_vals
                contrib_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Contribution": contributions,
                }).sort_values("Contribution", key=abs, ascending=True)

                colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in contrib_df["Contribution"]]

                fig, ax = plt.subplots(figsize=(5, 2.2))
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#ffffff")
                ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                        color=colors, height=0.48, edgecolor="none")
                ax.axvline(0, color="#e8ddd5", linewidth=1)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.tick_params(colors="#5c4a3a", labelsize=7.5)
                ax.set_xlabel("Contribution", fontsize=7.5, color="#9b8c84")
                pos_p = mpatches.Patch(color="#8b1a2f", label="Toward Premium")
                neg_p = mpatches.Patch(color="#b0a8a0", label="Toward Non-Premium")
                ax.legend(handles=[pos_p, neg_p], framealpha=0,
                          labelcolor="#5c4a3a", fontsize=7, loc="lower right")
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Food science notes
                st.markdown('<p class="sec-title" style="margin-top:0.6rem;">Food Science Analysis</p>', unsafe_allow_html=True)
                notes = get_food_science_notes(feats_saved)
                color_map = {"good": "#e8f5ee", "warn": "#fdf0f2", "neutral": "#faf7f2"}
                border_map = {"good": "#4a7c59", "warn": "#8b1a2f", "neutral": "#d0c8c0"}
                text_map   = {"good": "#2d5a3d", "warn": "#8b1a2f", "neutral": "#5c4a3a"}
                for aspect, text, tone in notes:
                    st.markdown(
                        f'<div style="background:{color_map[tone]};border-left:3px solid {border_map[tone]};'
                        f'border-radius:0 6px 6px 0;padding:0.4rem 0.7rem;margin-bottom:0.35rem;'
                        f'font-size:0.76rem;color:{text_map[tone]};line-height:1.45;">'
                        f'<strong>{aspect}:</strong> {text}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Awaiting state
            st.markdown(f"""
            <div style="position:relative;border-radius:10px;overflow:hidden;
                        height:148px;margin-bottom:0.8rem;">
                <img src="{IMG['pour']}" style="width:100%;height:100%;
                     object-fit:cover;opacity:0.3;" alt="Wine">
                <div style="position:absolute;inset:0;display:flex;align-items:center;
                            justify-content:center;flex-direction:column;gap:0.35rem;
                            background:rgba(250,247,242,0.55);">
                    <div style="font-family:'Playfair Display',serif;font-size:0.95rem;
                                color:#5c4a3a;font-weight:600;">Awaiting Analysis</div>
                    <div style="font-size:0.72rem;color:#9b8c84;">
                        Set measurements on the left and press
                        <strong style="color:#8b1a2f;">Analyse Wine</strong>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Engineered feature chips (always visible) ──
        st.markdown('<p class="sec-title">Engineered Features</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Calculated live from your inputs. These are the 5 values the model actually receives.</p>', unsafe_allow_html=True)
        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity":      "Flavour Intensity",
            "acidity_quality":       "Acidity Quality",
            "sugar_acid_balance":    "Sugar / Acid Balance",
            "so2_efficiency":        "SO2 Efficiency",
        }
        chips = '<div class="feat-grid">'
        for k, v in feats.items():
            chips += f'<div class="feat-chip"><span class="fname">{feat_labels[k]}</span><span class="fval">{v:.4f}</span></div>'
        chips += "</div>"
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metrics-strip">
      <div class="metric-card">
        <span class="mc-label">Accuracy</span>
        <div class="mc-val">91.6%</div>
        <span class="mc-note">Test set &middot; 320 wines</span>
      </div>
      <div class="metric-card">
        <span class="mc-label">F1 Score (Premium)</span>
        <div class="mc-val">0.710</div>
        <span class="mc-note">Harmonic mean P&amp;R</span>
      </div>
      <div class="metric-card">
        <span class="mc-label">AUC-ROC</span>
        <div class="mc-val">0.951</div>
        <span class="mc-note">Discriminatory power</span>
      </div>
      <div class="metric-card secondary">
        <span class="mc-label">Baseline Accuracy</span>
        <div class="mc-val secondary">86.4%</div>
        <span class="mc-note">Majority-class dummy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ins_left, ins_right = st.columns([3, 2], gap="large")

    with ins_left:
        # ── Global feature importance chart ───────────────────────────────────
        st.markdown('<p class="sec-title">Global Feature Importances</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Mean decrease in Gini impurity across all 100+ decision trees. Higher = more influential globally across all 1,599 wines.</p>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        max_imp = imp_df["Importance"].max()
        bar_colors = []
        for v in imp_df["Importance"]:
            t = 0.3 + 0.7 * (v / max_imp)
            bar_colors.append((0.545*t + (1-t), 0.102*t + (1-t), 0.184*t + (1-t)))

        fig2, ax2 = plt.subplots(figsize=(6, 2.8))
        fig2.patch.set_facecolor("#faf7f2")
        ax2.set_facecolor("#faf7f2")
        ax2.barh(imp_df["Feature"], imp_df["Importance"],
                 color=bar_colors, height=0.5, edgecolor="none")
        for i, (_, row) in enumerate(imp_df.iterrows()):
            ax2.text(row["Importance"] + 0.003, i,
                     f'{row["Importance"]:.3f}', va="center", color="#5c4a3a", fontsize=8.5)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.tick_params(colors="#5c4a3a", labelsize=8.5)
        ax2.set_xlabel("Mean Decrease in Gini Impurity", fontsize=8, color="#9b8c84")
        ax2.set_xlim(0, max_imp + 0.07)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # ── Model comparison table ─────────────────────────────────────────────
        st.markdown('<p class="sec-title" style="margin-top:0.9rem;">Model Comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Exact results from the held-out test set (320 wines, stratified 80/20 split).</p>', unsafe_allow_html=True)
        st.markdown("""
        <table class="cmp-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>F1 (Premium)</th>
              <th>AUC-ROC</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Baseline (majority class)</td>
              <td>86.4%</td>
              <td>0.000</td>
              <td>0.500</td>
            </tr>
            <tr>
              <td>5 Raw features (RF)</td>
              <td>90.0%</td>
              <td>0.686</td>
              <td>0.927</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>5 Engineered features (RF)</strong></td>
              <td class="best">91.6%</td>
              <td class="best">0.710</td>
              <td class="best">0.951</td>
            </tr>
          </tbody>
        </table>
        <div style="font-size:0.68rem;color:#9b8c84;margin-top:0.5rem;">
          Note: accuracy alone is misleading at 14% class balance &mdash;
          F1 and AUC-ROC are the meaningful measures.
        </div>
        """, unsafe_allow_html=True)

    with ins_right:
        # ── Image strip ────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="img-strip">
            <div class="img-card">
                <img src="{IMG['vineyard']}" alt="Vineyard">
                <div class="img-label">1,599 Red Wines &middot; UCI Dataset</div>
            </div>
            <div class="img-card">
                <img src="{IMG['lab']}" alt="Laboratory">
                <div class="img-label">Random Forest &middot; 5-Fold CV</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Transparency box ────────────────────────────────────────────────────
        st.markdown('<p class="sec-title">Model Transparency</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="transp-box">
          <h4>Dataset</h4>
          UCI Red Wine Quality &mdash; 1,599 Portuguese Vinho Verde red wines.
          80/20 stratified train/test split (1,279 train, 320 test).

          <h4>Algorithm</h4>
          Random Forest with <code>class_weight='balanced'</code>. SMOTE applied
          inside each fold via <code>imblearn.Pipeline</code> to prevent leakage.
          Tuned with 5-fold GridSearchCV on F1 score.

          <h4>Class Imbalance</h4>
          14.0% premium wines. Baseline accuracy of 86.4% is achieved by predicting
          Non-Premium every time &mdash; F1 = 0.00. This model achieves F1 = 0.710.

          <h4>Threshold</h4>
          Default: 0.50. Raising it (e.g. 0.65) increases precision and reduces
          false Premium labels &mdash; useful when mislabelling has a cost.

          <h4>Scope</h4>
          Trained on red wine only. White wines or other regions are out of scope.

          <h4>Contribution Chart</h4>
          Uses importance &times; scaled value as a proxy for per-prediction
          explanations. SHAP values would give more rigorous local attribution.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="meth-hero">
        <img src="{IMG['cellar']}" alt="Wine cellar">
        <div class="meth-hero-overlay">
            <div class="meth-hero-text">
                <h2>Scientific Methodology</h2>
                <p>Five chemically meaningful ratios engineered from raw lab measurements &mdash;
                each encoding a winemaking principle used to assess quality.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="meth-grid">
      <div class="meth-card">
        <span class="mc-tag">alcohol_density_ratio</span>
        <div class="mc-formula">alcohol &divide; density</div>
        <div class="mc-title">Body &amp; Mouthfeel</div>
        <div class="mc-body">Higher alcohol relative to density indicates greater extract
        and fuller palate weight. Density decreases as fermentation converts sugar to alcohol.
        Correlation with quality: <strong>+0.48</strong>.</div>
        <div class="dir-good">Higher = fuller body</div>
      </div>
      <div class="meth-card">
        <span class="mc-tag">flavor_intensity</span>
        <div class="mc-formula">sulphates &times; alcohol</div>
        <div class="mc-title">Aroma Complexity</div>
        <div class="mc-body">Sulphates protect volatile aromatic compounds; alcohol extracts
        them from grape skins. Their product captures both preservation and extraction capacity.
        Correlation: <strong>+0.41</strong>.</div>
        <div class="dir-good">Higher = more aromatic richness</div>
      </div>
      <div class="meth-card">
        <span class="mc-tag">acidity_quality</span>
        <div class="mc-formula">pH &times; volatile acidity</div>
        <div class="mc-title">Fault Detection</div>
        <div class="mc-body">Volatile acidity above ~0.6 g/dm&sup3; is detectable as vinegar.
        Multiplying by pH amplifies the penalty when both are elevated, flagging
        poor microbial stability. Correlation: <strong>&minus;0.38</strong>.</div>
        <div class="dir-bad">Lower = cleaner, better balanced</div>
      </div>
      <div class="meth-card">
        <span class="mc-tag">sugar_acid_balance</span>
        <div class="mc-formula">residual sugar &divide; fixed acidity</div>
        <div class="mc-title">Sweetness Perception</div>
        <div class="mc-body">Residual sugar interacts with fixed acidity to shape perceived
        roundness. A high ratio indicates excess sweetness relative to the acid backbone
        &mdash; atypical for quality dry reds. Correlation: <strong>&minus;0.03</strong>
        (retained for domain interaction effects).</div>
        <div class="dir-bad">Lower = drier, more structured</div>
      </div>
      <div class="meth-card">
        <span class="mc-tag">so2_efficiency</span>
        <div class="mc-formula">free SO&sub2; &divide; alcohol</div>
        <div class="mc-title">Preservation Efficiency</div>
        <div class="mc-body">Normalising free SO&sub2; by alcohol yields a preservation
        efficiency score. Too low: oxidation risk. Too high: sulfurous off-aromas
        detectable above ~50 mg/L. Correlation: <strong>&minus;0.12</strong>.</div>
        <div class="dir-range">Optimal: 1.5 &ndash; 3.5</div>
      </div>
    </div>

    <div style="font-size:0.69rem;color:#9b8c84;line-height:1.6;margin-top:0.9rem;">
      <strong>References:</strong>
      Peynaud, E. (1987). <em>Knowing and Making Wine.</em> Wiley. &middot;
      Cortez et al. (2009). Modeling wine preferences by data mining from physicochemical properties.
      <em>Decision Support Systems, 47(4).</em> &middot;
      OIV (2023). International Code of Oenological Practices.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Wine Quality Predictor &nbsp;&middot;&nbsp; Kavinda Pushpa Kumara
    &nbsp;&middot;&nbsp; Food Science &amp; Data Science Portfolio
    &nbsp;&middot;&nbsp; UCI Red Wine Dataset &middot; 1,599 Wines
</div>
""", unsafe_allow_html=True)
