"""
Wine Quality Prediction — Streamlit Application
Author : Kavinda Pushpa Kumara
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional CSS Theme with improved margins and compact inputs
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F4F6F8;
    color: #1A1A1A;
}

.stApp {
    background-color: #F4F6F8;
}

/* Improved Margins: Center the app and prevent cornering */
.block-container {
    max-width: 1000px !important;
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
    margin: 0 auto;
}

#MainMenu, footer, header { visibility: hidden; }

/* Typography */
.title-text {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #2C3E50;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Card Containers */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Compact Sliders to prevent scrolling */
.stSlider {
    padding-bottom: 0 !important;
    margin-bottom: -15px !important;
}
.stSlider > label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #4A5568 !important;
}

/* Buttons */
.stButton > button {
    background-color: #2C3E50 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    padding: 0.5rem 1rem !important;
    width: 100%;
    transition: background-color 0.2s ease;
}
.stButton > button:hover {
    background-color: #1A252F !important;
}

/* Results */
.result-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for UI toggles
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files missing: {e}")
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
    "alcohol": 10.42, "density": 0.9967, "sulphates": 0.658, "pH": 3.311,
    "volatile_acidity": 0.528, "residual_sugar": 2.539, "fixed_acidity": 8.32,
    "free_sulfur_dioxide": 15.87,
}

def engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_sulfur_dioxide):
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

# Header
st.markdown('<div class="title-text">Predictive Oenology Analysis</div>', unsafe_allow_html=True)

# Main Grid
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="font-weight: 600; margin-bottom: 1rem;">Chemical Parameters</p>', unsafe_allow_html=True)
    
    r1, r2 = st.columns(2)
    with r1:
        alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1)
        sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01)
        pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01)
        residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1)
    
    with r2:
        density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f")
        volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01)
        fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1)
        free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Quality Assessment"):
        st.session_state.analyzed = True
        st.session_state.show_analysis = False # Reset analysis view on new run
        feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_so2)
        label, prob, scaled_vals = make_prediction(feats)
        st.session_state.prediction_data = {"label": label, "prob": prob, "scaled": scaled_vals}
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
    
    # Static Model Info (To be updated with notebook data)
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
        <p style="font-size: 0.8rem; font-weight: 600; color: #4A5568; margin-bottom: 0.2rem;">Model Performance Metrics</p>
        <p style="font-size: 0.75rem; color: #718096; margin: 0;">
            <strong>Accuracy:</strong> [Insert Accuracy]%<br>
            <strong>F1 Score:</strong> [Insert F1]<br>
            <em>Note: Metrics derived via Random Forest Classifier evaluated on the standard test holdout.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.analyzed:
        data = st.session_state.prediction_data
        
        # Display Prediction
        if data["label"] == 1:
            st.markdown(f'<div class="result-text" style="color: #2C3E50;">Premium Grade</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-text" style="color: #718096;">Standard Grade</div>', unsafe_allow_html=True)
            
        st.markdown(f'<p style="text-align: center; color: #4A5568; font-size: 0.9rem;">Confidence: <strong>{data["prob"]*100:.1f}%</strong></p>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)
        
        # Conditional Analysis Button
        if not st.session_state.show_analysis:
            if st.button("View Detailed Feature Analysis"):
                st.session_state.show_analysis = True
                st.rerun()
                
        if st.session_state.show_analysis:
            contributions = importances * data["scaled"]
            contrib_df = pd.DataFrame({
                "Feature": ["Alc/Den", "Flavor", "Acidity", "Sugar/Acid", "SO2"],
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            colors = ["#2C3E50" if v >= 0 else "#A0AEC0" for v in contrib_df["Contribution"]]

            fig, ax = plt.subplots(figsize=(5, 2))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors, height=0.5)
            ax.axvline(0, color="#E2E8F0", linewidth=1)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#4A5568", labelsize=7)
            
            pos_p = mpatches.Patch(color="#2C3E50", label="Favors Premium")
            neg_p = mpatches.Patch(color="#A0AEC0", label="Favors Standard")
            ax.legend(handles=[pos_p, neg_p], frameon=False, fontsize=6, loc="lower right")
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #A0AEC0;">
            <p style="font-size: 0.9rem;">Awaiting Input Parameters...</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
