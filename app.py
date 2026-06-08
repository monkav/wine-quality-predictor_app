import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Premium Wine Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .wine-card {
        background: linear-gradient(135deg, #6b4e31, #8b5a2b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 1rem 0;
        text-align: center;
    }
    .result-premium { background: linear-gradient(135deg, #228B22, #32CD32); }
    .result-non { background: linear-gradient(135deg, #CD5C5C, #FF6347); }
    h1 { color: #4a2c0b; }
</style>
""", unsafe_allow_html=True)

st.title("🍷 Premium Wine Quality Predictor")
st.markdown("**Engineered Features • Random Forest • Food Science Insights**")

# ====================== LOAD MODELS FROM GITHUB ======================
@st.cache_resource
def load_model():
    base_url = "https://raw.githubusercontent.com/monkav/wine-quality-predictor_app/main/"
    try:
        # Load model
        model_resp = requests.get(base_url + "wine_model.pkl")
        model = joblib.load(BytesIO(model_resp.content))
        
        # Load scaler
        scaler_resp = requests.get(base_url + "scaler.pkl")
        scaler = joblib.load(BytesIO(scaler_resp.content))
        
        st.success("✅ Models loaded successfully from GitHub")
        return model, scaler
    except Exception as e:
        st.warning("⚠️ Could not load models from GitHub. Running in demo mode.")
        return None, None

model, scaler = load_model()

# ====================== INPUTS ======================
st.subheader("Enter Wine Chemical Measurements")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, 10.0, 0.1)
    density = st.slider("Density (g/cm³)", 0.990, 1.004, 0.997, 0.0001)
    sulphates = st.slider("Sulphates (g/dm³)", 0.3, 2.0, 0.65, 0.01)
    pH = st.slider("pH", 2.7, 4.0, 3.3, 0.01)

with col2:
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.1, 1.6, 0.5, 0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.9, 15.0, 2.5, 0.1)
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 8.0, 0.1)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 1, 72, 15, 1)

# ====================== PREDICTION ======================
if st.button("🔍 Predict Premium Quality", type="primary", use_container_width=True):
    # Engineer the 5 features (exactly as in notebook)
    feat_dict = {
        "alcohol_density_ratio": alcohol / density,
        "flavor_intensity": sulphates * alcohol,
        "acidity_quality": pH * volatile_acidity,
        "sugar_acid_balance": residual_sugar / (fixed_acidity + 1e-6),
        "so2_efficiency": free_sulfur_dioxide / (alcohol + 1e-6),
    }
    
    input_df = pd.DataFrame([feat_dict])
    
    if model is not None and scaler is not None:
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0, 1]
    else:
        # Demo mode fallback
        prob = min(0.95, max(0.05, (alcohol * 0.15 + sulphates * 0.4 - volatile_acidity * 0.5 + 0.3)))
    
    is_premium = prob >= 0.5
    label = "PREMIUM" if is_premium else "STANDARD"
    
    # Winemaker notes
    notes = []
    if feat_dict['alcohol_density_ratio'] > 12.0:
        notes.append("🌟 Full-bodied (high alcohol-density ratio)")
    if feat_dict['flavor_intensity'] > 9.0:
        notes.append("🍇 High aroma complexity")
    if feat_dict['acidity_quality'] < 2.0:
        notes.append("⚖️ Clean acidity balance")
    if 15 <= feat_dict['so2_efficiency'] <= 35:
        notes.append("🛡️ Optimal SO₂ preservation")
    
    # ====================== DISPLAY RESULT ======================
    st.markdown("### Prediction Result")
    
    card_class = "result-premium" if is_premium else "result-non"
    st.markdown(f"""
    <div class="wine-card {card_class}">
        <h2 style="margin:0;">{label}</h2>
        <h1 style="margin:0; font-size:3.5rem;">{prob:.1%}</h1>
        <p style="margin:0;">Premium Probability</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Engineered Feature Values")
    cols = st.columns(5)
    for i, (k, v) in enumerate(feat_dict.items()):
        with cols[i % 5]:
            st.metric(k.replace("_", " ").title(), f"{v:.3f}")
    
    if notes:
        st.subheader("Winemaker Notes")
        for note in notes:
            st.success(note)
    
    if is_premium:
        st.balloons()
        st.success("🎉 This wine exhibits premium characteristics!")
    else:
        st.info("💡 This wine falls in the standard category. Consider adjustments in key parameters.")

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Built with Feature Engineering + Random Forest | Inspired by Food Science | UCI Wine Quality Dataset")
