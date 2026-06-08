import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Settings optimized for Mobile Device Layouts
st.set_page_config(
    page_title="Premium Red Wine Evaluator",
    page_icon="🍷",
    layout="centered",  # Keeps layout tight and readable on smaller devices
    initial_sidebar_state="collapsed"
)

# 2. Injecting custom mobile responsive CSS tweaks
st.markdown("""
    <style>
    .main-title {
        color: #722F37;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-align: center;
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 2.2rem;
    }
    .subtitle {
        color: #555555;
        text-align: center;
        font-size: 1.0rem;
        margin-bottom: 25px;
    }
    .footer {
        text-align: center;
        color: #666666;
        font-size: 0.85rem;
        margin-top: 50px;
        border-top: 1px solid #e0e0e0;
        padding-top: 20px;
    }
    /* Make metric cards pop beautifully on mobile viewports */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #722F37;
    }
    </style>
""", unsafe_allowed_html=True)

# Application Header
st.markdown("<h1 class='main-title'>🍷 Premium Red Wine Evaluator</h1>", unsafe_allowed_html=True)
st.markdown("<p class='subtitle'>Bridging Food Science & Machine Learning to Identify Exceptional Red Wines</p>", unsafe_allowed_html=True)

# Navigational Tabs for a clean mobile user experience
tab1, tab2 = st.tabs(["🔮 Prediction Engine", "📊 Domain Feature Insights"])

with tab1:
    st.markdown("### 🧪 Input Lab Measurements")
    st.write("Adjust the sliders below to replicate the wine's chemical profile.")

    # Expanders cluster parameters cleanly so mobile screens don't look cluttered
    with st.expander("🍇 Body & Alcohol Characteristics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            alcohol = st.slider("Alcohol (% vol)", min_value=8.0, max_value=15.0, value=10.5, step=0.1,
                                help="Ethanol content by volume.")
        with col2:
            density = st.slider("Density (g/cm³)", min_value=0.9900, max_value=1.0040, value=0.9967, step=0.0001, format="%.4f",
                                help="Mass density of the fluid.")

    with st.expander("🍋 Acidity Structure", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            ph = st.slider("pH Level", min_value=2.70, max_value=4.10, value=3.30, step=0.01,
                           help="Logarithmic scale of wine acidity.")
        with col4:
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", min_value=0.10, max_value=1.60, value=0.52, step=0.01,
                                         help="Acetic acid concentration (high levels introduce vinegar traits).")
        
        col5, _ = st.columns(2)
        with col5:
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", min_value=4.5, max_value=16.0, value=8.3, step=0.1,
                                       help="Primary non-volatile organic acids (tartaric/malic).")

    with st.expander("🛡️ Preservation & Sweetness Balance", expanded=True):
        col6, col7 = st.columns(2)
        with col6:
            sulphates = st.slider("Sulphates (g/dm³)", min_value=0.30, max_value=2.00, value=0.65, step=0.01,
                                  help="Potassium sulphate additives protecting wine against microbial spoiling.")
        with col7:
            residual_sugar = st.slider("Residual Sugar (g/dm³)", min_value=0.9, max_value=15.5, value=2.2, step=0.1,
                                       help="Unfermented sugar concentration.")
        
        col8, _ = st.columns(2)
        with col8:
            free_sulfur_dioxide = st.slider("Free SO₂ (mg/dm³)", min_value=1.0, max_value=72.0, value=15.0, step=1.0,
                                             help="Active sulfur dioxide available for oxidation defense.")

    # Advanced Threshold Controls
    st.markdown("### 🎛️ Settings")
    threshold = st.slider("Classification Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Tuning this upward ensures only top-tier profiles receive the Premium classification.")

    # 4. On-the-fly Feature Engineering matching Kavinda's Notebook
    feat_vals = {
        'alcohol_density_ratio': alcohol / density,
        'flavor_intensity':      sulphates * alcohol,
        'acidity_quality':       ph * volatile_acidity,
        'sugar_acid_balance':    residual_sugar / (fixed_acidity + 1e-6),
        'so2_efficiency':        free_sulfur_dioxide / (alcohol + 1e-6),
    }

    # Extract food science notes based on your notebook thresholds
    notes = []
    if feat_vals['alcohol_density_ratio'] > 12.5:
        notes.append("🍷 **Mouthfeel:** Full-bodied structure (high alcohol-to-density ratio).")
    elif feat_vals['alcohol_density_ratio'] < 11.5:
        notes.append("🍇 **Mouthfeel:** Light-bodied structure (low alcohol-to-density ratio).")

    if feat_vals['flavor_intensity'] > 9:
        notes.append("✨ **Complexity:** High aroma complexity (optimal sulphates × alcohol extraction).")
    elif feat_vals['flavor_intensity'] < 6:
        notes.append("📉 **Complexity:** Low flavor intensity — might lack body depth.")

    if feat_vals['acidity_quality'] < 2.0:
        notes.append("🍏 **Acidity:** Clean acidity balance with no aggressive sour signatures.")
    elif feat_vals['acidity_quality'] > 3.0:
        notes.append("⚠️ **Acidity:** Elevated volatile acidity index; high risk of vinegar off-flavors.")

    if 15 <= feat_vals['so2_efficiency'] <= 35:
        notes.append("🛡️ **Preservation:** $SO_2$ efficiency sits inside the optimal preservation range.")
    elif feat_vals['so2_efficiency'] < 15:
        notes.append("💨 **Preservation:** Insufficient $SO_2$ protection; highly vulnerable to premature oxidation.")
    else:
        notes.append("🛑 **Preservation:** Excessive $SO_2$ treatment; risk of aromatic masking.")

    if not notes:
        notes.append("⚖️ **Profile:** Structurally balanced profile without extreme deviations.")

    st.markdown("---")
    
    # Large mobile-friendly target action button
    if st.button("🔍 Evaluate Wine Profile", use_container_width=True):
        # Package data into correct array format
        input_df = pd.DataFrame([feat_vals])
        # Force strict feature ordering corresponding to notebook compilation
        feature_order = ['alcohol_density_ratio', 'flavor_intensity', 'acidity_quality', 'sugar_acid_balance', 'so2_efficiency']
        input_df = input_df[feature_order]

        # Look for pickle artifacts
        if os.path.exists('wine_model.pkl') and os.path.exists('scaler.pkl'):
            try:
                scaler = joblib.load('scaler.pkl')
                model = joblib.load('wine_model.pkl')
                
                # Preprocess and execute prediction
                scaled_data = scaler.transform(input_df)
                prob = model.predict_proba(scaled_data)[0, 1]
                is_premium = prob >= threshold
            except Exception as e:
                st.error(f"Error reading model artifacts: {e}")
                prob = None
        else:
            # Intuitive fallback logic so app layout can be validated without files
            st.warning("ℹ️ **Model files (`wine_model.pkl` / `scaler.pkl`) not detected yet.** Running in UI Demonstration Mode:")
            # Simple fallback formulation matching direction of correlations
            prob = min(max((alcohol * 0.12 + sulphates * 0.25 - volatile_acidity * 0.45) / 2.0 + 0.3, 0.0), 1.0)
            is_premium = prob >= threshold

        if prob is not None:
            st.markdown("### 📊 Chemical Evaluation Report")
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                if is_premium:
                    st.success("🏆 **PREMIUM SELECTION**")
                else:
                    st.info("🍴 **ORDINARY TABLE WINE**")
            
            with c_res2:
                st.metric(label="Premium Grade Confidence", value=f"{prob*100:.1f}%")
            
            # Display localized winemaking notes
            st.markdown("#### 🔬 Food Science Commentary")
            for note in notes:
                st.write(note)

with tab2:
    st.markdown("### 📊 Engineering Foundations")
    st.write("Raw data point checks often fail to describe integrated tasting dynamics. This application leverages specialized **engineered interaction combinations** grounded in enology principles:")
    
    # Mathematical representations via LaTeX formatting
    st.markdown(r"""
    * **Alcohol-Density Ratio:** $\frac{\text{Alcohol}}{\text{Density}}$ — Proxies physical extract, viscosity, and mouthfeel weight.
    * **Flavor Intensity:** $\text{Sulphates} \times \text{Alcohol}$ — Captures structural preservation versus flavor concentration.
    * **Acidity Quality:** $\text{pH} \times \text{Volatile Acidity}$ — Evaluates vinegar defects relative to active matrix acidity.
    * **Sugar-Acid Balance:** $\frac{\text{Residual Sugar}}{\text{Fixed Acidity}}$ — Maps tartness mitigation on the palate.
    * **$SO_2$ Preservation Efficiency:** $\frac{\text{Free } SO_2}{\text{Alcohol}}$ — Identifies active chemical stability relative to alcohol volatility.
    """)
    
    st.success("📈 **Model Accuracy Achieved:** Your custom pipeline achieves a **91.56% Accuracy** and **0.951 AUC-ROC** on test datasets using these exact 5 parameters!")

# Footer element preserving author identity 
st.markdown("""
    <div class='footer'>
        <p><b>Author:</b> Kavinda Pushpa Kumara</p>
        <p>🔬 Food Science Student | IBM Certified Data Scientist</p>
    </div>
""", unsafe_allowed_html=True)
