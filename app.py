"""
Wine Quality Prediction — Streamlit Application
Author : Kavinda Pushpa Kumara
Role   : Food Science Student | IBM Certified Data Scientist
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — Sleek Professional Theme ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8F9FA;
    color: #212529;
}
.stApp {
    background-color: #F8F9FA;
}

/* Hide Default Chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container {
    padding: 1rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto;
}

/* Typography */
h1, h2, h3, .serif-font {
    font-family: 'Playfair Display', serif;
    color: #1A1A1A;
}

/* Header Banner */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid #DEE2E6;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #4A001F;
    margin: 0;
}
.header-meta {
    font-size: 0.8rem;
    color: #6C757D;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: right;
}

/* Cards & Panels */
.panel {
    background: #FFFFFF;
    border: 1px solid #E9ECEF;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    height: 100%;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 1px solid #F8F9FA;
    padding-bottom: 0.5rem;
}

/* Condense Sliders to Prevent Scroll */
.stSlider {
    padding-bottom: 0 !important;
    margin-bottom: -10px !important;
}
.stSlider > label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #495057 !important;
}
[data-testid="stThumbValue"] {
    background: #4A001F !important;
    color: #FFF !important;
    font-size: 0.7rem !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #4A001F !important;
}

/* Buttons */
.stButton > button {
    background-color: #4A001F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.5rem 1rem !important;
    width: 100%;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: #2D0013 !important;
    box-shadow: 0 4px 6px rgba(74, 0, 31, 0.2);
}

/* Result Outputs */
.result-box {
    text-align: center;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.result-premium {
    background-color: #FDFBFC;
    border: 1px solid #4A001F;
}
.result-standard {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
}
.result-status {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.text-premium { color: #4A001F; }
.text-standard { color: #495057; }

/* Engineered Features Grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.feature-item {
    background: #F8F9FA;
    padding: 0.75rem;
    border-radius: 6px;
    border: 1px solid #E9ECEF;
    text-align: center;
}
.feature-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: #6C757D;
    display: block;
    margin-bottom: 0.25rem;
}
.feature-value {
    font-family: monospace;
    font-weight: 600;
    font-size: 0.9rem;
    color: #212529;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .header-container { flex-direction: column; align-items: flex-start; }
    .header-meta { text-align: left; margin-top: 0.5rem; }
    .block-container { padding: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}\nPlease ensure wine_model.pkl and scaler.pkl are in the directory.")
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
    df = pd.DataFrame([feat_dict])
    scaled = scaler.transform(df)
    prob   = model.predict_proba(scaled)[0, 1]
    label  = 1 if prob >= 0.5 else 0
    return label, prob, scaled[0]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">Predictive Oenology Analysis</div>
    <div class="header-meta">
        Kavinda Pushpa Kumara<br>
        Food Science & Data Science Portfolio
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
# Using a 2-column layout to fit everything on one screen
col1, col2 = st.columns([1.2, 1], gap="medium")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Chemical Profile Parameters</div>', unsafe_allow_html=True)
    
    # 4x2 tight grid for sliders to prevent scrolling
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1)
        sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01)
        pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01)
        residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1)
    
    with r1c2:
        density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f")
        volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01)
        fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1)
        free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Quality Assessment")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Aesthetically pleasing placeholder image for the right side
    st.image("https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&q=80&w=1200&h=400", use_column_width=True)
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_so2)

    if predict_btn:
        label, prob, scaled_vals = make_prediction(feats)

        if label == 1:
            st.markdown(f"""
            <div class="result-box result-premium">
                <div class="result-status text-premium">Premium Grade</div>
                <div style="color: #6C757D; font-size: 0.9rem;">
                    Confidence Score: <strong>{prob*100:.1f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-standard">
                <div class="result-status text-standard">Standard Grade</div>
                <div style="color: #6C757D; font-size: 0.9rem;">
                    Confidence Score: <strong>{(1-prob)*100:.1f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        # Detailed Analysis Expander replaces the immediate chart
        with st.expander("View Detailed Feature Analysis"):
            st.markdown("<span style='font-size:0.8rem; color:#6C757D;'>Feature influence on this specific prediction.</span>", unsafe_allow_html=True)
            contributions = importances * scaled_vals
            contrib_df = pd.DataFrame({
                "Feature": ["Alc/Den Ratio", "Flavor Int.", "Acidity Qual.", "Sugar/Acid Bal.", "SO2 Effic."],
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            colors = ["#4A001F" if v >= 0 else "#ADB5BD" for v in contrib_df["Contribution"]]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors, height=0.6)
            ax.axvline(0, color="#DEE2E6", linewidth=1)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#495057", labelsize=8)
            
            pos_p = mpatches.Patch(color="#4A001F", label="Favors Premium")
            neg_p = mpatches.Patch(color="#ADB5BD", label="Favors Standard")
            ax.legend(handles=[pos_p, neg_p], frameon=False, fontsize=7, loc="lower right")
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #ADB5BD;">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Awaiting Input Parameters</p>
            <p style="font-size: 0.85rem;">Adjust the chemical properties on the left and run the assessment to view predicted quality.</p>
        </div>
        """, unsafe_allow_html=True)

    # Engineered Features Matrix
    st.markdown('<div class="panel-title" style="margin-top: 1rem; font-size: 0.9rem;">Live Engineered Metrics</div>', unsafe_allow_html=True)
    
    labels_map = {
        "alcohol_density_ratio": "Alc / Den",
        "flavor_intensity":      "Flavor Int",
        "acidity_quality":       "Acidity Qual",
        "sugar_acid_balance":    "Sugar / Acid",
        "so2_efficiency":        "SO2 Effic",
    }
    
    html_grid = '<div class="feature-grid">'
    for k, v in feats.items():
        html_grid += f'''
        <div class="feature-item">
            <span class="feature-label">{labels_map[k]}</span>
            <span class="feature-value">{v:.3f}</span>
        </div>'''
    html_grid += '</div>'
    st.markdown(html_grid, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
