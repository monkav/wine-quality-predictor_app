"""
Wine Quality Prediction — Professional Edition
Author: Kavinda Pushpa Kumara
Role: Food Science Student | IBM Certified Data Scientist
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page configuration (no emojis) ───────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor | Professional Edition",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS: Modern, clean, professional design ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=Playfair+Display:wght@400;500;600;700&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fc;
    color: #1e293b;
}

.stApp {
    background: linear-gradient(135deg, #f8f9fc 0%, #f0f4f8 100%);
}

/* Remove default Streamlit elements */
#MainMenu, footer, header {
    visibility: hidden;
}
[data-testid="stSidebar"] {
    display: none;
}
.block-container {
    padding: 0 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Main container */
.main-container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 2rem 2rem 1rem 2rem;
}

/* Header professional */
.pro-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    flex-wrap: wrap;
    gap: 1rem;
}
.brand {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.brand-icon {
    width: 48px;
    height: 48px;
    background: #7a2e3b;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 600;
    color: white;
}
.brand-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #2c1810;
    margin: 0;
    letter-spacing: -0.02em;
}
.brand-text p {
    margin: 0;
    font-size: 0.8rem;
    color: #6b7280;
    letter-spacing: 0.3px;
}
.author-info {
    text-align: right;
}
.author-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #2c1810;
}
.author-title {
    font-size: 0.7rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background: transparent;
    border-bottom: 1px solid #e2e8f0;
    padding: 0;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    color: #5b677b;
    padding: 0.7rem 0;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #7a2e3b;
    border-bottom: 2px solid #7a2e3b;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0;
}

/* Cards and containers */
.card {
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.03);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #edf2f7;
    transition: box-shadow 0.2s ease;
}
.card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2c1810;
    margin-bottom: 0.25rem;
}
.card-subtitle {
    font-size: 0.7rem;
    color: #8a99b0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #f0f2f5;
    padding-bottom: 0.5rem;
}

/* Two column layout for inputs */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}
.input-group {
    margin-bottom: 0.8rem;
}

/* Slider styling */
.stSlider > label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #3c4a5f !important;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
[data-testid="stSlider"] > div > div > div > div {
    background-color: #7a2e3b !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background-color: #7a2e3b !important;
}

/* Button primary */
.stButton > button {
    background: #2c1810 !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: #7a2e3b !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(122, 46, 59, 0.2);
}

/* Secondary button */
.secondary-btn > button {
    background: transparent !important;
    border: 1px solid #cbd5e1 !important;
    color: #3c4a5f !important;
    box-shadow: none !important;
}
.secondary-btn > button:hover {
    background: #f8fafc !important;
    border-color: #7a2e3b !important;
    color: #7a2e3b !important;
}

/* Result cards */
.result-premium {
    background: linear-gradient(135deg, #fff7f5 0%, #ffffff 100%);
    border-left: 4px solid #7a2e3b;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
    margin-bottom: 1.5rem;
}
.result-standard {
    background: #ffffff;
    border-left: 4px solid #94a3b8;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.result-prob {
    font-size: 0.85rem;
    color: #5b677b;
}
.result-prob strong {
    color: #2c1810;
    font-weight: 700;
}

/* Engineered feature chips */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.feature-chip {
    background: #f8fafc;
    border: 1px solid #eef2f6;
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
    transition: all 0.1s ease;
}
.feature-chip .fname {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #7e8b9e;
    display: block;
    margin-bottom: 0.2rem;
}
.feature-chip .fval {
    font-family: 'Inter', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    color: #1e293b;
}

/* Methodology grid */
.meth-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.meth-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem;
    border-top: 3px solid #7a2e3b;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: transform 0.1s ease;
}
.meth-card .mc-tag {
    font-family: monospace;
    font-size: 0.7rem;
    background: #f9efef;
    color: #7a2e3b;
    padding: 0.2rem 0.5rem;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 0.6rem;
}
.meth-card .mc-formula {
    font-family: monospace;
    font-size: 0.75rem;
    background: #f1f5f9;
    padding: 0.2rem 0.4rem;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 0.6rem;
}
.meth-card .mc-title {
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
    color: #2c1810;
}
.meth-card .mc-body {
    font-size: 0.75rem;
    color: #43506c;
    line-height: 1.5;
}
.direction-good { color: #2e7d64; font-size: 0.7rem; margin-top: 0.5rem; font-weight: 500; }
.direction-bad { color: #b91c1c; font-size: 0.7rem; margin-top: 0.5rem; font-weight: 500; }
.direction-range { color: #b45309; font-size: 0.7rem; margin-top: 0.5rem; font-weight: 500; }

/* Insights and metrics */
.metric-strip {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    flex: 1;
    text-align: center;
}
.metric-box .mb-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: #7e8b9e;
}
.metric-box .mb-val {
    font-weight: 700;
    font-size: 1rem;
    color: #2c1810;
}

/* Image banner */
.wine-banner {
    width: 100%;
    height: 140px;
    object-fit: cover;
    border-radius: 20px;
    margin-bottom: 1.5rem;
}

/* Divider */
.section-divider {
    margin: 1.5rem 0;
    border-top: 1px solid #edf2f7;
}

/* Footer */
.pro-footer {
    text-align: center;
    padding: 1.5rem 0;
    font-size: 0.7rem;
    color: #8ca0b3;
    border-top: 1px solid #edf2f7;
    margin-top: 2rem;
}

/* Responsive */
@media (max-width: 768px) {
    .main-container { padding: 1rem; }
    .pro-header { flex-direction: column; align-items: start; }
    .author-info { text-align: left; }
    .input-grid { grid-template-columns: 1fr; }
    .feature-grid { grid-template-columns: 1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}\n\nPlease ensure wine_model.pkl and scaler.pkl are in the same directory.")
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

# Dataset means for default values
MEANS = {
    "alcohol": 10.42,
    "density": 0.9967,
    "sulphates": 0.658,
    "pH": 3.311,
    "volatile_acidity": 0.528,
    "residual_sugar": 2.539,
    "fixed_acidity": 8.32,
    "free_sulfur_dioxide": 15.87,
}


def engineer_features(alcohol, density, sulphates, pH, volatile_acidity,
                      residual_sugar, fixed_acidity, free_sulfur_dioxide):
    return {
        "alcohol_density_ratio": alcohol / density,
        "flavor_intensity": sulphates * alcohol,
        "acidity_quality": pH * volatile_acidity,
        "sugar_acid_balance": residual_sugar / (fixed_acidity + 1e-6),
        "so2_efficiency": free_sulfur_dioxide / (alcohol + 1e-6),
    }


def make_prediction(feat_dict):
    df = pd.DataFrame([feat_dict])
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0, 1]
    label = 1 if prob >= 0.5 else 0
    return label, prob, scaled[0]


# ── Initialize session state ──────────────────────────────────────────────────
if "pred_available" not in st.session_state:
    st.session_state.pred_available = False
    st.session_state.pred_label = None
    st.session_state.pred_prob = None
    st.session_state.pred_scaled = None
    st.session_state.pred_feats = None
    st.session_state.pred_contributions = None
    st.session_state.show_detail = False


# ── Header with brand and optional image ──────────────────────────────────────
col_logo, col_img = st.columns([2, 1])
with col_logo:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">W</div>
        <div class="brand-text">
            <h1>Wine Quality Predictor</h1>
            <p>Analytical decision support for premium red wine classification</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_img:
    st.image(
        "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?w=400&auto=format",
        use_container_width=True,
        caption="Chemical analysis meets oenology"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_method, tab_insights = st.tabs([
    "Predict Wine Quality",
    "Methodology",
    "Model Insights"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    # Two column layout: inputs | result + features
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Raw chemical profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Enter laboratory measurements</div>', unsafe_allow_html=True)

            # Input grid
            col1, col2 = st.columns(2)
            with col1:
                alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1,
                                    help="Percentage alcohol by volume")
                sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01,
                                      help="Potassium sulphate — preservative & aroma")
                pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01,
                               help="Acidity level — lower = more acidic")
                residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1,
                                           help="Unfermented sugar remaining")
            with col2:
                density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f",
                                    help="Wine density — related to alcohol & sugar")
                volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01,
                                             help="Acetic acid — high levels = vinegar taste")
                fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1,
                                          help="Tartaric acid — backbone of the wine")
                free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5,
                                     help="Free sulfur dioxide — prevents oxidation")

            st.markdown('</div>', unsafe_allow_html=True)

            # Predict button
            predict_clicked = st.button("Analyse Wine Quality", use_container_width=True)

    with col_result:
        # Live engineered features
        feats = engineer_features(alcohol, density, sulphates, pH,
                                  volatile_acidity, residual_sugar,
                                  fixed_acidity, free_so2)

        if predict_clicked:
            label, prob, scaled_vals = make_prediction(feats)
            contributions = importances * scaled_vals

            # Store in session state
            st.session_state.pred_available = True
            st.session_state.pred_label = label
            st.session_state.pred_prob = prob
            st.session_state.pred_scaled = scaled_vals
            st.session_state.pred_feats = feats
            st.session_state.pred_contributions = contributions
            st.session_state.show_detail = False  # reset detail view

            # Display result
            if label == 1:
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label">Premium</div>
                    <div class="result-prob">
                        Confidence {prob*100:.1f}% · Predicted quality score ≥ 7
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("This wine meets the premium chemical profile. Balanced structure and aromatic complexity.")
            else:
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label">Non-Premium</div>
                    <div class="result-prob">
                        Confidence {(1-prob)*100:.1f}% · Predicted quality score < 7
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("Profile below premium threshold. Adjusting volatile acidity and flavor intensity may improve quality.")
        else:
            st.markdown("""
            <div style="background: #fafcff; border: 1px solid #eef2f6; border-radius: 20px; padding: 2rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 0.9rem; color: #6a7a99;">Enter measurements and click<br><strong>Analyse Wine Quality</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # Engineered features always visible
        st.markdown('<div class="card-title">Engineered Features</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Chemically meaningful ratios</div>', unsafe_allow_html=True)
        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity": "Flavour Intensity",
            "acidity_quality": "Acidity Quality",
            "sugar_acid_balance": "Sugar / Acid Balance",
            "so2_efficiency": "SO₂ Efficiency",
        }
        chips_html = '<div class="feature-grid">'
        for k, v in feats.items():
            chips_html += f"""
            <div class="feature-chip">
                <span class="fname">{feat_labels[k]}</span>
                <span class="fval">{v:.4f}</span>
            </div>"""
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        # Detailed analysis button + toggle section
        if st.session_state.pred_available:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            # Button toggles detail view
            if not st.session_state.show_detail:
                if st.button("Show Detailed Analysis", key="detail_btn", use_container_width=True):
                    st.session_state.show_detail = True
                    st.rerun()
            else:
                if st.button("Hide Detailed Analysis", key="hide_detail_btn", use_container_width=True):
                    st.session_state.show_detail = False
                    st.rerun()

            # Show detailed analysis if flag is true
            if st.session_state.show_detail:
                with st.container():
                    st.markdown('<div class="card-title">Why this decision?</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-subtitle">Feature contribution analysis</div>', unsafe_allow_html=True)

                    contributions = st.session_state.pred_contributions
                    contrib_df = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Contribution": contributions,
                    }).sort_values("Contribution", key=abs, ascending=True)

                    colors = ["#7a2e3b" if v >= 0 else "#a0abbc" for v in contrib_df["Contribution"]]

                    fig, ax = plt.subplots(figsize=(5, 2.8))
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#ffffff")
                    ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                            color=colors, height=0.5, edgecolor="none")
                    ax.axvline(0, color="#e2e8f0", linewidth=1)
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    ax.tick_params(colors="#43506c", labelsize=8)
                    ax.set_xlabel("Contribution (importance × scaled value)", fontsize=8, color="#7e8b9e")
                    pos_patch = mpatches.Patch(color="#7a2e3b", label="Contributes to Premium")
                    neg_patch = mpatches.Patch(color="#a0abbc", label="Contributes to Non-Premium")
                    ax.legend(handles=[pos_patch, neg_patch], framealpha=0, labelcolor="#3c4a5f", fontsize=7.5, loc="lower right")
                    plt.tight_layout(pad=0.4)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.info("Positive contribution (burgundy) indicates the feature pushes toward Premium classification. Negative contribution (grey) pushes toward Non-Premium. Values are derived from model feature importance multiplied by scaled input value.")
        else:
            st.info("After analysing a wine, you can view detailed feature contributions here.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown("""
    <div class="card">
        <div class="card-title">Scientific methodology</div>
        <div class="card-subtitle">Engineered features based on enological principles</div>
        <div class="meth-grid">
            <div class="meth-card">
                <span class="mc-tag">alcohol_density_ratio</span>
                <div class="mc-formula">alcohol ÷ density</div>
                <div class="mc-title">Body & mouthfeel</div>
                <div class="mc-body">Higher alcohol relative to density indicates greater extract and fuller palate weight — a key attribute of premium red wines.</div>
                <div class="direction-good">Higher = fuller body</div>
            </div>
            <div class="meth-card">
                <span class="mc-tag">flavor_intensity</span>
                <div class="mc-formula">sulphates × alcohol</div>
                <div class="mc-title">Aroma complexity</div>
                <div class="mc-body">Sulphates protect aromatic compounds; alcohol extracts them. Product captures preservation capacity and extraction power.</div>
                <div class="direction-good">Higher = aromatic richness</div>
            </div>
            <div class="meth-card">
                <span class="mc-tag">acidity_quality</span>
                <div class="mc-formula">pH × volatile acidity</div>
                <div class="mc-title">Fault detection</div>
                <div class="mc-body">Volatile acidity >0.6 g/dm³ is perceptible as vinegar. Amplifies penalty for wines with high VA and elevated pH.</div>
                <div class="direction-bad">Lower = cleaner balance</div>
            </div>
            <div class="meth-card">
                <span class="mc-tag">sugar_acid_balance</span>
                <div class="mc-formula">residual sugar ÷ fixed acidity</div>
                <div class="mc-title">Sweetness perception</div>
                <div class="mc-body">Even in dry reds, residual sugar interacts with fixed acidity to shape roundness. High ratio indicates excess sweetness.</div>
                <div class="direction-bad">Lower = drier, structured</div>
            </div>
            <div class="meth-card">
                <span class="mc-tag">so2_efficiency</span>
                <div class="mc-formula">free SO₂ ÷ alcohol</div>
                <div class="mc-title">Preservation efficiency</div>
                <div class="mc-body">Free SO₂ prevents oxidation. Normalising by alcohol yields preservation efficiency. Too low → oxidation risk; too high → off-aromas.</div>
                <div class="direction-range">Optimal range: 1.5 – 3.5</div>
            </div>
        </div>
        <div style="font-size:0.7rem; color:#8a99b0; margin-top: 1rem; border-top: 1px solid #edf2f7; padding-top: 1rem;">
            <strong>References:</strong> Peynaud, E. (1987). Knowing and Making Wine. · Cortez et al. (2009). Modeling wine preferences. Decision Support Systems. · OIV (2023).
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
with tab_insights:
    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Global feature importances</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Average reduction in Gini impurity (Random Forest)</div>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        fig2.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")
        bars = ax2.barh(imp_df["Feature"], imp_df["Importance"], color="#7a2e3b", height=0.5, alpha=0.8)
        for i, (_, row) in enumerate(imp_df.iterrows()):
            ax2.text(row["Importance"] + 0.003, i, f"{row['Importance']:.3f}", va="center", color="#3c4a5f", fontsize=8.5)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.tick_params(colors="#43506c", labelsize=8.5)
        ax2.set_xlabel("Mean Decrease in Gini Impurity", fontsize=8, color="#7e8b9e")
        ax2.set_xlim(0, imp_df["Importance"].max() + 0.05)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="card">
            <div class="card-title">Model transparency</div>
            <div class="card-subtitle">Dataset & algorithm details</div>
            <div style="font-size: 0.8rem; line-height: 1.5;">
                <strong>Dataset</strong><br>
                UCI Red Wine Quality — 1,599 Portuguese Vinho Verde wines.<br><br>
                <strong>Algorithm</strong><br>
                Random Forest Classifier with <code>class_weight='balanced'</code> and SMOTE inside cross-validation pipeline.<br><br>
                <strong>Class imbalance</strong><br>
                Only ~14% of wines score ≥7. Model evaluated on F1 score and AUC-ROC.<br><br>
                <strong>Classification threshold</strong><br>
                Default 0.50. Adjust upward for higher precision.<br><br>
                <strong>Scope</strong><br>
                Trained on red wine only. Results for white wines or other regions unreliable.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pro-footer">
    Wine Quality Predictor — Kavinda Pushpa Kumara · Food Science & Data Science Portfolio
</div>
""", unsafe_allow_html=True)
