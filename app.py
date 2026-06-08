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

/* FIXED: Changed global padding from 0 to professional centered dimensions to create a sidebar gutter */
.block-container { 
    padding: 1.5rem 1rem !important; 
    max-width: 1200px !important; 
}

/* Safe area for notched phones */
.stApp > div:first-child { padding-left: env(safe-area-inset-left); padding-right: env(safe-area-inset-right); }

/* ── Hero ── */
.hero-banner {
    position: relative; width: 100%; height: 160px;
    overflow: hidden; background: #1a0a0e;
    border-radius: 12px; /* Adds a modern touch on curved glass displays */
}
.hero-banner img {
    width: 100%; height: 100%; object-fit: cover;
    object-position: center 60%; opacity: 0.5; display: block;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(15,4,6,0.88) 0%, rgba(15,4,6,0.5) 55%, rgba(15,4,6,0.1) 100%);
    display: flex; align-items: center;
    padding: 0 2rem; gap: 2rem;
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
    padding: 0 1.5rem; gap: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    border-radius: 8px;
    margin-top: 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Lato', sans-serif;
    font-size: 0.79rem; font-weight: 500;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: #9b8c84; padding: 0.85rem 1.2rem;
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
    padding: 1.2rem 0.5rem 1.5rem;
    max-width: 1320px; margin: 0 auto; width: 100%;
    box-sizing: border-box;
}

/* ── Input panel ── */
.input-panel {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 12px;
    padding: 1.1rem 1.2rem 0.6rem;
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
    border-radius: 8px;
    margin-top: 1rem;
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
    /* FIXED: Prevent container bleed, adding clear side breathing room frames */
    .main .block-container { 
        padding-left: 14px !important; 
        padding-right: 14px !important; 
    }
    
    .tab-content {
        background: #ffffff !important;
        border-radius: 14px !important;
        padding: 12px !important;
        margin: 8px 0 16px 0 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04) !important;
        box-sizing: border-box;
    }
    
    .input-panel {
        background: #ffffff !important;
        border: 1px solid #e8ddd5 !important;
        border-radius: 12px !important;
        padding: 14px 12px 10px !important;
        margin: 0 0 12px 0 !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
        box-sizing: border-box;
        width: 100% !important;
    }
    
    .panel-header img { height: 44px !important; }
    
    /* Ensure slider labels stay crisp and perfectly visible */
    [data-testid="stSlider"] label, .stSlider label {
        color: #5c4a3a !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        white-space: normal !important;
    }
    
    /* Dynamic internal structure paddings */
    [data-testid="column"] { 
        padding-left: 4px !important; 
        padding-right: 4px !important; 
    }

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

/* Fix white-on-light text */
.stSuccess, .stInfo, .stWarning, .stError {
    color: #2c2118 !important;
}
.stSuccess div, .stInfo div, .stWarning div, .stError div {
    color: #2c2118 !important;
}
/* Ensure light background panels protect dark text metrics */
[style*="#faf7f2"], [style*="#fffdf6"] {
    color: #5c4a3a !important;
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
    elif aq >
