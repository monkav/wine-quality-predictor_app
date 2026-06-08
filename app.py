import streamlit as st
import pandas as pd
import numpy as np
import os

# 1. Page Configuration for crisp Mobile layout
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Predictor")
st.write("Adjust the wine characteristics below to predict its quality.")

# 2. Arrange inputs in columns (displays side-by-side on PC, automatically stacks vertically on Mobile)
col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.5, step=0.1)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65, step=0.01)

with col2:
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.52, step=0.01)
    ph = st.slider("pH Level", 2.7, 4.1, 3.3, step=0.01)

st.markdown("---")

# 3. Safe Prediction Logic
if st.button("🔍 Evaluate Wine Quality", use_container_width=True):
    
    # Check if your saved model file actually exists on GitHub yet
    if os.path.exists('wine_model.pkl'):
        try:
            import joblib
            model = joblib.load('wine_model.pkl')
            
            # Formulate the feature dict matching your data science notebook
            input_data = pd.DataFrame({
                'alcohol_density_ratio': [alcohol / 0.9967], # default average density proxy
                'flavor_intensity':      [sulphates * alcohol],
                'acidity_quality':       [ph * volatile_acidity],
                'sugar_acid_balance':    [2.5 / 8.3],        # default averages
                'so2_efficiency':        [15.0 / alcohol]
            })
            
            prob = model.predict_proba(input_data)[0, 1]
            if prob >= 0.5:
                st.success(f"🏆 PREMIUM QUALITY (Confidence: {prob*100:.1f}%)")
            else:
                st.info(f"🍴 STANDARD TABLE WINE (Confidence: {(1-prob)*100:.1f}%)")
                
        except Exception as e:
            st.error(f"Error reading model file: {e}")
            
    else:
        # ⚠️ SAFE FALLBACK MODE: Runs seamlessly even if your model files aren't uploaded yet!
        st.warning("⚡ Running in Preview Mode (Model file 'wine_model.pkl' not found on GitHub yet)")
        
        # Simple food science fallback calculation to show how the app works on your phone
        mock_score = (alcohol * 0.5) + (sulphates * 0.3) - (volatile_acidity * 0.8)
        
        if mock_score > 4.8:
            st.success("🏆 PREDICTED RESULT: Premium Quality Wine (Score ≥ 7)")
        else:
            st.info("🍴 PREDICTED RESULT: Ordinary Table Wine")

# Author Footer
st.markdown("""
    <br><hr><center style='color: gray; font-size: 0.85rem;'>
        <b>Author:</b> Kavinda Pushpa Kumara<br>
        🔬 Food Science Student | IBM Certified Data Scientist
    </center>
""", unsafe_allowed_html=True)
