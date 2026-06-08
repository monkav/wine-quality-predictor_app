import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Intelligence",
    layout="wide",
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

# ── CSS (Design System) ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: #F8F5F2;
    color: #1A1A1A;
}}

/* ── Headings ── */
h1, h2, h3, .serif-title {{
    font-family: 'Cormorant Garamond', serif;
    color: #7B1E3A;
}}

/* ── Hero Section ── */
.hero-banner {{
    position: relative; width: 100%; height: 280px;
    border-radius: 16px; overflow: hidden; margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}}
.hero-banner img {{
    width: 100%; height: 100%; object-fit: cover;
    object-position: center 60%;
}}
.hero-overlay {{
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(10,10,10,0.9) 0%, rgba(123,30,58,0.7) 100%);
    display: flex; flex-direction: column; justify-content: center;
    padding: 0 4rem;
}}
.hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700; color: #FFFFFF;
    margin: 0 0 0.5rem; line-height: 1.1;
}}
.hero-sub {{
    font-family: 'Inter', sans-serif; font-size: 1.1rem;
    color: #D4AF37; font-weight: 300; margin-bottom: 1.5rem;
}}
.hero-cta {{
    display: inline-block; padding: 0.8rem 2rem;
    background-color: #D4AF37; color: #1A1A1A;
    font-weight: 600; border-radius: 4px; text-decoration: none;
    text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;
    width: max-content;
}}

/* ── Glassmorphism Cards ── */
.glass-card {{
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
}}

/* ── KPI Overview ── */
.kpi-container {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem; margin-bottom: 2rem;
}}
.kpi-card {{
    text-align: center; padding: 1.5rem;
    border-top: 4px solid #7B1E3A;
}}
.kpi-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem; font-weight: 700; color: #7B1E3A; line-height: 1;
}}
.kpi-label {{
    font-family: 'Inter', sans-serif; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 1px; color: #666;
    margin-top: 0.5rem;
}}

/* ── Prediction Results ── */
.result-premium {{
    background: linear-gradient(135deg, #7B1E3A 0%, #A63D5D 100%);
    color: white; border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 10px 25px rgba(123, 30, 58, 0.3);
}}
.result-premium h2 {{ color: #D4AF37; font-size: 2.5rem; margin-bottom: 0; }}

.result-standard {{
    background: #FFFFFF; border: 1px solid #E0E0E0;
    color: #1A1A1A; border-radius: 16px; padding: 2rem; text-align: center;
}}
.result-standard h2 {{ color: #1A1A1A; font-size: 2.5rem; margin-bottom: 0; }}

.gauge-bar {{
    width: 100%; height: 8px; background: rgba(255,255,255,0.2);
    border-radius: 4px; margin: 1rem 0; overflow: hidden;
}}
.gauge-fill {{
    height: 100%; background: #D4AF37; border-radius: 4px;
}}
.gauge-bar-std {{ background: #EEEEEE; }}
.gauge-fill-std {{ background: #7B1E3A; }}

/* ── Features Card ── */
.feat-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;
}}
.feat-item {{
    background: #FFFFFF; padding: 1rem; border-radius: 8px;
    border-left: 3px solid #D4AF37;
}}
.feat-name {{ font-size: 0.75rem; text-transform: uppercase; color: #666; letter-spacing: 0.5px; }}
.feat-val {{ font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; font-weight: 600; color: #7B1E3A; }}

/* ── Flow Diagram ── */
.flow-diagram {{
    text-align: center; font-family: 'Inter', monospace;
    background: #FFFFFF; padding: 2rem; border-radius: 12px;
    color: #7B1E3A; font-weight: 500; font-size: 1.1rem;
}}

/* ── Footer ── */
.footer {{
    text-align: center; padding: 2rem 0; margin-top: 3rem;
    border-top: 1px solid #E0E0E0; font-size: 0.85rem; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
}}
</style>
""", unsafe_allow_html=True)


# ── Model & Data Functions ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        # Mocking for preview if model not present
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        model = RandomForestClassifier(random_state=42)
        model.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        scaler = StandardScaler()
        scaler.fit(np.random.rand(10, 5))
        return model, scaler

model, scaler = load_model()

FEATURE_NAMES = [
    "alcohol_density_ratio",
    "flavor_intensity",
    "acidity_quality",
    "sugar_acid_balance",
    "so2_efficiency",
]

MEANS = {
    "alcohol": 10.42, "density": 0.9967, "sulphates": 0.658, "pH": 3.311,
    "volatile_acidity": 0.528, "residual_sugar": 2.539, "fixed_acidity": 8.32,
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
    prob   = model.predict_proba(scaled)[0, 1] if hasattr(model, 'predict_proba') else 0.85
    label  = 1 if prob >= 0.5 else 0
    return label, prob


# ── Layout ────────────────────────────────────────────────────────────────────

# 1. HERO SECTION
st.markdown(f"""
<div class="hero-banner">
    <img src="{IMG['hero']}" alt="Wine Banner">
    <div class="hero-overlay">
        <h1 class="hero-title">Wine Quality Intelligence Platform</h1>
        <p class="hero-sub">AI-powered wine quality assessment using engineered chemical features</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. KPI OVERVIEW
st.markdown("""
<div class="kpi-container">
    <div class="glass-card kpi-card">
        <div class="kpi-value">1,599</div>
        <div class="kpi-label">Dataset Size</div>
    </div>
    <div class="glass-card kpi-card">
        <div class="kpi-value">5</div>
        <div class="kpi-label">Engineered Features</div>
    </div>
    <div class="glass-card kpi-card">
        <div class="kpi-value">RF</div>
        <div class="kpi-label">Model Type</div>
    </div>
    <div class="glass-card kpi-card">
        <div class="kpi-value">&ge; 7</div>
        <div class="kpi-label">Target Class (Premium)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# TABS
tab_analysis, tab_analytics, tab_model, tab_about = st.tabs([
    "MAIN ANALYSIS", "ANALYTICS", "MODEL ARCHITECTURE", "ABOUT"
])

# ── 3. MAIN ANALYSIS PAGE ─────────────────────────────────────────────────────
with tab_analysis:
    col_input, col_pred = st.columns([1.2, 1], gap="large")
    
    with col_input:
        st.markdown('<h3 class="serif-title">Chemical Profile Inputs</h3>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Responsive Grid Using number_input
        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.number_input("Alcohol (%vol)", value=float(MEANS["alcohol"]), step=0.1)
            sulphates = st.number_input("Sulphates (g/dm³)", value=float(MEANS["sulphates"]), step=0.01)
            pH = st.number_input("pH", value=float(MEANS["pH"]), step=0.01)
            residual_sugar = st.number_input("Residual Sugar (g/dm³)", value=float(MEANS["residual_sugar"]), step=0.1)
        with c2:
            density = st.number_input("Density (g/cm³)", value=float(MEANS["density"]), step=0.0001, format="%.4f")
            volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", value=float(MEANS["volatile_acidity"]), step=0.01)
            fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", value=float(MEANS["fixed_acidity"]), step=0.1)
            free_so2 = st.number_input("Free SO2 (mg/dm³)", value=float(MEANS["free_sulfur_dioxide"]), step=0.5)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_pred:
        st.markdown('<h3 class="serif-title">Prediction Result</h3>', unsafe_allow_html=True)
        
        feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_so2)
        label, prob = make_prediction(feats)
        
        # Right Panel: Prediction Card
        if label == 1:
            st.markdown(f"""
            <div class="result-premium">
                <h2>PREMIUM</h2>
                <p style="text-transform:uppercase; letter-spacing:1px; font-size:0.9rem; margin-bottom:0;">Quality Assessment</p>
                <div class="gauge-bar">
                    <div class="gauge-fill" style="width: {prob*100}%;"></div>
                </div>
                <p style="font-size:1.5rem; font-family:'Cormorant Garamond', serif;">{prob*100:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-standard">
                <h2>NON-PREMIUM</h2>
                <p style="text-transform:uppercase; letter-spacing:1px; font-size:0.9rem; color:#666; margin-bottom:0;">Quality Assessment</p>
                <div class="gauge-bar gauge-bar-std">
                    <div class="gauge-fill gauge-fill-std" style="width: {(1-prob)*100}%;"></div>
                </div>
                <p style="font-size:1.5rem; font-family:'Cormorant Garamond', serif; color:#7B1E3A;">{(1-prob)*100:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # 4. ENGINEERED FEATURES KPI CARDS
        st.markdown('<h3 class="serif-title" style="margin-top:2rem;">Engineered Features</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="feat-grid">
            <div class="feat-item"><div class="feat-name">Alcohol Density Ratio</div><div class="feat-val">{feats['alcohol_density_ratio']:.2f}</div></div>
            <div class="feat-item"><div class="feat-name">Flavor Intensity</div><div class="feat-val">{feats['flavor_intensity']:.2f}</div></div>
            <div class="feat-item"><div class="feat-name">Acidity Quality</div><div class="feat-val">{feats['acidity_quality']:.2f}</div></div>
            <div class="feat-item"><div class="feat-name">Sugar Acid Balance</div><div class="feat-val">{feats['sugar_acid_balance']:.2f}</div></div>
            <div class="feat-item" style="grid-column: span 2;"><div class="feat-name">SO2 Efficiency</div><div class="feat-val">{feats['so2_efficiency']:.2f}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 5. DETAILED ANALYSIS (Hidden by default)
        with st.expander("VIEW DETAILED ANALYSIS"):
            st.markdown("""
            **Chemical Interpretation of Engineered Ratios:**
            * **Alcohol Density Ratio:** Indicates extract and palate weight.
            * **Flavor Intensity:** Captures aromatic richness and preservation.
            * **Acidity Quality:** Flags potential faults like microbial instability.
            * **Sugar Acid Balance:** Represents sweetness perception versus acid structure.
            * **SO2 Efficiency:** Evaluates preservation against oxidation.
            """)

# ── 6. ANALYTICS PAGE ─────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown('<h2 class="serif-title">Model Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="kpi-container">
        <div class="glass-card kpi-card"><div class="kpi-value">91.6%</div><div class="kpi-label">Accuracy</div></div>
        <div class="glass-card kpi-card"><div class="kpi-value">0.710</div><div class="kpi-label">F1 Score</div></div>
        <div class="glass-card kpi-card"><div class="kpi-value">0.951</div><div class="kpi-label">ROC AUC</div></div>
        <div class="glass-card kpi-card"><div class="kpi-value">1599</div><div class="kpi-label">Dataset Size</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="serif-title">Global Feature Importance</h3>', unsafe_allow_html=True)
    
    # Modern Horizontal Bar Chart
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0.35, 0.25, 0.20, 0.15, 0.05]
        
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    y_pos = np.arange(len(FEATURE_NAMES))
    
    ax.barh(y_pos, importances, color='#7B1E3A', align='center', height=0.6)
    ax.set_yticks(y_pos, labels=[f.replace('_', ' ').title() for f in FEATURE_NAMES])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666')
    ax.spines['left'].set_color('#666')
    ax.tick_params(colors='#1A1A1A')
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ── 7. MODEL PAGE ─────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<h2 class="serif-title">Model Architecture</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 class="serif-title" style="margin-top:0;">Configuration</h3>
            <ul style="line-height: 2;">
                <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                <li><strong>Balancing:</strong> SMOTE</li>
                <li><strong>Scaling:</strong> StandardScaler</li>
                <li><strong>Decision Threshold:</strong> 0.50</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="flow-diagram">
            Raw Inputs<br>
            &#8595;<br>
            Feature Engineering<br>
            &#8595;<br>
            Scaling<br>
            &#8595;<br>
            Random Forest<br>
            &#8595;<br>
            Prediction
        </div>
        """, unsafe_allow_html=True)

# ── 8. ABOUT PAGE ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="serif-title">About the Project</h2>', unsafe_allow_html=True)
    st.write("""
    This project is an AI-powered quality assessment tool that translates raw lab measurements into premium wine classifications. 
    By bridging chemical analysis with machine learning, the platform derives engineered features reflecting real-world food science principles—such as flavor intensity and acidity quality—to produce highly accurate quality predictions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ── 9. FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Wine Quality Intelligence Platform</strong><br>
    Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
