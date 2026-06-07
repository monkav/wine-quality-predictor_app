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

# ── CSS — Modern Professional Dashboard Theme ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Merriweather:wght@300;400;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
    color: #1a1c20;
}
.stApp {
    background: #f8f9fa;
}

/* ── Hide default Streamlit chrome & Optimize space ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    overflow: hidden; /* Helps prevent scrolling on optimized screens */
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(rgba(26, 28, 32, 0.85), rgba(26, 28, 32, 0.95)), 
                url('https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=2000&auto=format&fit=crop') center/cover;
    padding: 1.5rem 3rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 3px solid #722F37;
}
.hero-title {
    font-family: 'Merriweather', serif;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}
.hero-meta {
    font-size: 0.8rem;
    color: #adb5bd;
    text-align: right;
    line-height: 1.4;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Layout Containers ── */
.main-wrapper {
    padding: 1rem 3rem;
    max-width: 1600px;
    margin: 0 auto;
}

/* ── Inputs Section ── */
.input-section {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.section-title {
    font-family: 'Merriweather', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #212529;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid #e9ecef;
    padding-bottom: 0.5rem;
}

/* ── Sliders Customization ── */
.stSlider > label {
    font-size: 0.75rem !important;
    color: #495057 !important;
    font-weight: 600 !important;
}
[data-testid="stTickBar"] { display: none; }
[data-testid="stSlider"] > div > div > div > div {
    background: #722F37 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #722F37 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 0.5rem 1rem !important;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #5a252c !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* ── Result Cards ── */
.result-card {
    border-radius: 6px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.result-premium {
    background: #fdfaf6;
    border: 1px solid #722F37;
    border-top: 4px solid #722F37;
}
.result-standard {
    background: white;
    border: 1px solid #ced4da;
    border-top: 4px solid #6c757d;
}
.result-label {
    font-family: 'Merriweather', serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-prob {
    font-size: 0.9rem;
    color: #6c757d;
}

/* ── Engineered Feature Grid ── */
.feature-metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
}
.metric-box {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 0.75rem;
    text-align: center;
}
.metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: #6c757d;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'Merriweather', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #212529;
}

/* ── Expander overrides ── */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #495057;
    background-color: #f8f9fa;
    border-radius: 4px;
}

/* ── Mobile & Responsive ── */
@media (max-width: 1024px) {
    .feature-metrics { grid-template-columns: repeat(3, 1fr); }
    .hero-header { flex-direction: column; text-align: center; padding: 1rem; }
    .hero-meta { text-align: center; margin-top: 0.5rem; }
    .main-wrapper { padding: 1rem; overflow-y: auto; } /* Allow scrolling on smaller devices */
}
@media (max-width: 768px) {
    .feature-metrics { grid-template-columns: repeat(2, 1fr); }
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
        st.error(f"Model file not found: {e}\n\nPlace wine_model.pkl and scaler.pkl in the same directory.")
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
<div class="hero-header">
    <div class="hero-title">Predictive Wine Analytics</div>
    <div class="hero-meta">
        <strong>Kavinda Pushpa Kumara</strong><br>
        Food Science & Data Science Portfolio
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Container for Inputs
st.markdown('<div class="input-section"><div class="section-title">Chemical Profile Inputs</div>', unsafe_allow_html=True)

# Compact 4-column grid for 8 inputs to save vertical space
col1, col2, col3, col4 = st.columns(4)

with col1:
    alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1)
    sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01)
with col2:
    pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1)
with col3:
    density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f")
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01)
with col4:
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1)
    free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# Process Features Live
feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_so2)

# Action & Results Area
res_col1, res_col2 = st.columns([1, 3])

with res_col1:
    st.markdown('<br>', unsafe_allow_html=True)
    predict_btn = st.button("Generate Prediction", use_container_width=True)

with res_col2:
    if predict_btn:
        label, prob, scaled_vals = make_prediction(feats)
        
        if label == 1:
            st.markdown(f"""
            <div class="result-card result-premium">
                <div class="result-label" style="color:#722F37;">Premium Classification</div>
                <div class="result-prob">Confidence Level: <strong>{prob*100:.1f}%</strong> | Predicted Quality ≥ 7</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-standard">
                <div class="result-label" style="color:#495057;">Standard Classification</div>
                <div class="result-prob">Confidence Level: <strong>{(1-prob)*100:.1f}%</strong> | Predicted Quality &lt; 7</div>
            </div>""", unsafe_allow_html=True)
            
        # Detailed Analysis hidden behind an expander instead of cluttering the main view
        with st.expander("View Detailed Analysis"):
            st.markdown("### Feature Contributions")
            st.markdown("Displays how each engineered feature influenced the current prediction threshold.")
            contributions = importances * scaled_vals
            contrib_df = pd.DataFrame({
                "Feature": ["Alcohol/Density", "Flavor Intensity", "Acidity Quality", "Sugar/Acid Bal.", "SO2 Efficiency"],
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            colors = ["#722F37" if v >= 0 else "#6c757d" for v in contrib_df["Contribution"]]

            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors, height=0.4)
            ax.axvline(0, color="#dee2e6", linewidth=1.5)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#495057", labelsize=9)
            ax.set_xlabel("Contribution Weight", fontsize=9, color="#6c757d")
            pos_p = mpatches.Patch(color="#722F37", label="Pushes towards Premium")
            neg_p = mpatches.Patch(color="#6c757d", label="Pushes towards Standard")
            ax.legend(handles=[pos_p, neg_p], framealpha=0, fontsize=8, loc="lower right")
            plt.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()
    else:
        st.markdown("""
        <div style="background:white; border:1px dashed #ced4da; border-radius:6px; padding:2rem; text-align:center; color:#6c757d;">
            Adjust the chemical parameters and initialize the model to view predictions.
        </div>""", unsafe_allow_html=True)

# ── Engineered Features Bar ───────────────────────────────────────────────────
feat_labels = {
    "alcohol_density_ratio": "Alcohol / Density",
    "flavor_intensity":      "Flavor Intensity",
    "acidity_quality":       "Acidity Quality",
    "sugar_acid_balance":    "Sugar/Acid Ratio",
    "so2_efficiency":        "SO₂ Efficiency",
}

chips_html = '<div class="feature-metrics">'
for k, v in feats.items():
    chips_html += f"""
    <div class="metric-box">
        <div class="metric-label">{feat_labels[k]}</div>
        <div class="metric-value">{v:.4f}</div>
    </div>"""
chips_html += '</div></div>'

st.markdown(chips_html, unsafe_allow_html=True)
