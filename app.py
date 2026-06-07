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
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — warm ivory editorial light theme ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
}
.stApp {
    background: #faf7f2;
    max-width: 100%;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── App shell ── */
.app-shell {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ── Top header bar ── */
.top-bar {
    background: #fff;
    border-bottom: 2px solid #8b1a2f;
    padding: 0.7rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.top-bar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #8b1a2f;
    letter-spacing: 0.01em;
}
.top-bar-meta {
    font-size: 0.72rem;
    color: #9b8c84;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    text-align: right;
    line-height: 1.5;
}

/* ── Tab nav ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-bottom: 1px solid #e8ddd5;
    padding: 0 2rem;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Lato', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9b8c84;
    padding: 0.85rem 1.5rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: #8b1a2f !important;
    border-bottom: 2px solid #8b1a2f !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Tab content wrapper ── */
.tab-content {
    padding: 1.4rem 2rem;
    max-width: 1280px;
    margin: 0 auto;
    width: 100%;
}

/* ── Input panel ── */
.input-panel {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.input-panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #2c2118;
    margin-bottom: 0.1rem;
}
.input-panel-sub {
    font-size: 0.73rem;
    color: #9b8c84;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border-bottom: 1px solid #f0ebe4;
    padding-bottom: 0.6rem;
}

/* ── Sliders ── */
.stSlider > label {
    font-size: 0.78rem !important;
    color: #5c4a3a !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
}
.stSlider [data-testid="stThumbValue"] {
    background: #8b1a2f !important;
    color: #fff !important;
    font-size: 0.72rem !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #8b1a2f !important;
}

/* ── Predict button ── */
.stButton > button {
    background: #8b1a2f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.65rem 0 !important;
    width: 100%;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #a02038 !important;
}

/* ── Result card ── */
.result-premium {
    background: linear-gradient(135deg, #fff8f8 0%, #fff 100%);
    border: 2px solid #8b1a2f;
    border-radius: 12px;
    padding: 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-standard {
    background: #fff;
    border: 2px solid #d0c8c0;
    border-radius: 12px;
    padding: 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.3rem;
}
.result-prob {
    font-size: 0.85rem;
    color: #9b8c84;
}
.result-prob strong { color: #2c2118; }

/* ── Engineered feature tags ── */
.feat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.feat-chip {
    background: #faf7f2;
    border: 1px solid #e8ddd5;
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    font-size: 0.75rem;
}
.feat-chip .fname {
    color: #9b8c84;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
    margin-bottom: 0.1rem;
}
.feat-chip .fval {
    color: #2c2118;
    font-weight: 600;
    font-family: monospace;
    font-size: 0.82rem;
}

/* ── Section titles ── */
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2c2118;
    margin: 0 0 0.3rem 0;
}
.sec-sub {
    font-size: 0.75rem;
    color: #9b8c84;
    margin-bottom: 0.9rem;
    line-height: 1.5;
}

/* ── Methodology cards ── */
.meth-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.9rem;
    margin-top: 0.5rem;
}
.meth-card {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    border-top: 3px solid #8b1a2f;
}
.meth-card .mc-tag {
    font-family: monospace;
    font-size: 0.72rem;
    background: #fdf0f2;
    color: #8b1a2f;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.4rem;
}
.meth-card .mc-formula {
    font-size: 0.78rem;
    color: #5c4a3a;
    font-family: monospace;
    margin-bottom: 0.4rem;
}
.meth-card .mc-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #2c2118;
    margin-bottom: 0.3rem;
}
.meth-card .mc-body {
    font-size: 0.79rem;
    color: #5c4a3a;
    line-height: 1.55;
}
.direction-good  { color: #4a7c59; font-size: 0.72rem; font-weight: 500; margin-top: 0.35rem; }
.direction-bad   { color: #8b1a2f; font-size: 0.72rem; font-weight: 500; margin-top: 0.35rem; }
.direction-range { color: #7a6a30; font-size: 0.72rem; font-weight: 500; margin-top: 0.35rem; }

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.metric-box {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-box .mb-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9b8c84;
    display: block;
    margin-bottom: 0.15rem;
}
.metric-box .mb-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c2118;
}

/* ── Insights panel ── */
.insights-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

/* ── Transparency box ── */
.transp-box {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #5c4a3a;
    line-height: 1.65;
}
.transp-box h4 {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: #2c2118;
    margin: 0 0 0.5rem 0;
}

/* ── Footer ── */
.app-footer {
    background: #fff;
    border-top: 1px solid #e8ddd5;
    padding: 0.7rem 2rem;
    text-align: center;
    font-size: 0.71rem;
    color: #c8bdb8;
    letter-spacing: 0.05em;
    margin-top: auto;
}

/* ── Mobile responsive ── */
@media (max-width: 640px) {
    .top-bar { padding: 0.6rem 1rem; }
    .top-bar-title { font-size: 1.1rem; }
    .top-bar-meta { display: none; }
    .tab-content { padding: 1rem; }
    .feat-grid { grid-template-columns: 1fr 1fr; }
    .insights-grid { grid-template-columns: 1fr; }
    .meth-grid { grid-template-columns: 1fr; }
    .stTabs [data-baseweb="tab"] { padding: 0.75rem 0.9rem; font-size: 0.75rem; }
}
@media (max-width: 400px) {
    .feat-grid { grid-template-columns: 1fr; }
    .metric-strip { gap: 0.5rem; }
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

# Dataset means for default slider values (UCI Red Wine)
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
    """Convert raw wine measurements into 5 engineered features."""
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


# ── Top header bar ────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-title">🍷 Wine Quality Predictor</div>
    <div class="top-bar-meta">
        Kavinda Pushpa Kumara<br>
        Food Science Student · IBM Certified Data Scientist
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_method, tab_insights = st.tabs([
    "  🔬  Predict  ",
    "  📖  Methodology  ",
    "  📊  Model Insights  ",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    col_inputs, col_result = st.columns([1, 1], gap="large")

    # ── Left: raw inputs ──────────────────────────────────────────────
    with col_inputs:
        st.markdown("""
        <div class="input-panel">
            <div class="input-panel-title">Raw Chemical Measurements</div>
            <div class="input-panel-sub">Enter lab values — features are engineered automatically</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0,
                                float(MEANS["alcohol"]), 0.1,
                                help="Percentage alcohol by volume")
            sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00,
                                  float(MEANS["sulphates"]), 0.01,
                                  help="Potassium sulphate — preservative & aroma")
            pH = st.slider("pH", 2.70, 4.50,
                           float(MEANS["pH"]), 0.01,
                           help="Acidity level — lower = more acidic")
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0,
                                       float(MEANS["residual_sugar"]), 0.1,
                                       help="Unfermented sugar remaining")

        with c2:
            density = st.slider("Density (g/cm³)", 0.990, 1.004,
                                float(MEANS["density"]), 0.0001,
                                format="%.4f",
                                help="Wine density — related to alcohol & sugar")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60,
                                         float(MEANS["volatile_acidity"]), 0.01,
                                         help="Acetic acid — high levels = vinegar taste")
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0,
                                      float(MEANS["fixed_acidity"]), 0.1,
                                      help="Tartaric acid — backbone of the wine")
            free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0,
                                 float(MEANS["free_sulfur_dioxide"]), 0.5,
                                 help="Free sulfur dioxide — prevents oxidation")

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyse Wine", use_container_width=True)

    # ── Right: result ─────────────────────────────────────────────────
    with col_result:

        # always engineer features so chips update live
        feats = engineer_features(alcohol, density, sulphates, pH,
                                   volatile_acidity, residual_sugar,
                                   fixed_acidity, free_so2)

        if predict_btn:
            label, prob, scaled_vals = make_prediction(feats)

            if label == 1:
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label" style="color:#8b1a2f;">✦ Premium</div>
                    <div class="result-prob">
                        Confidence <strong>{prob*100:.1f}%</strong> · Quality predicted ≥ 7
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("Chemical profile meets premium criteria. Strong body, aromatic complexity, and controlled acidity contribute positively.")
            else:
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label" style="color:#5c4a3a;">◇ Non-Premium</div>
                    <div class="result-prob">
                        Confidence <strong>{(1-prob)*100:.1f}%</strong> · Quality predicted &lt; 7
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("Profile does not meet premium threshold. Review volatile acidity and flavour intensity — key levers for quality improvement.")

            # ── Per-prediction contribution chart ──
            st.markdown('<p class="sec-title" style="margin-top:1rem;">Why this decision?</p>', unsafe_allow_html=True)
            st.markdown('<p class="sec-sub">Feature importance × scaled value. Burgundy = pushes Premium; grey = pushes Non-Premium.</p>', unsafe_allow_html=True)

            contributions = importances * scaled_vals
            contrib_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in contrib_df["Contribution"]]

            fig, ax = plt.subplots(figsize=(5, 2.6))
            fig.patch.set_facecolor("#faf7f2")
            ax.set_facecolor("#faf7f2")
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                    color=colors, height=0.5, edgecolor="none")
            ax.axvline(0, color="#e8ddd5", linewidth=1)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#5c4a3a", labelsize=8)
            ax.xaxis.label.set_color("#9b8c84")
            ax.set_xlabel("Contribution", fontsize=8, color="#9b8c84")
            pos_p = mpatches.Patch(color="#8b1a2f", label="→ Premium")
            neg_p = mpatches.Patch(color="#b0a8a0", label="→ Non-Premium")
            ax.legend(handles=[pos_p, neg_p], framealpha=0,
                      labelcolor="#5c4a3a", fontsize=7.5, loc="lower right")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            st.markdown("""
            <div style="background:#fff;border:1px dashed #d0c8c0;border-radius:12px;
                        padding:2.5rem 1.5rem;text-align:center;color:#c8bdb8;font-size:0.88rem;margin-bottom:1rem;">
                Set measurements on the left<br>and press <strong style="color:#8b1a2f;">Analyse Wine</strong>
            </div>""", unsafe_allow_html=True)

        # ── Engineered feature chips — always visible ──────────────────
        st.markdown('<p class="sec-title">Engineered Features</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Calculated live from your inputs above.</p>', unsafe_allow_html=True)

        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity":      "Flavour Intensity",
            "acidity_quality":       "Acidity Quality",
            "sugar_acid_balance":    "Sugar / Acid Balance",
            "so2_efficiency":        "SO₂ Efficiency",
        }
        chips_html = '<div class="feat-grid">'
        for k, v in feats.items():
            chips_html += f"""
            <div class="feat-chip">
                <span class="fname">{feat_labels[k]}</span>
                <span class="fval">{v:.4f}</span>
            </div>"""
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    st.markdown('<p class="sec-title">Scientific Methodology</p>', unsafe_allow_html=True)
    st.markdown("""<p class="sec-sub">
        This model does not feed raw laboratory measurements directly into the classifier.
        Five chemically meaningful ratios are engineered from the raw data — each encoding a
        principle that winemakers and oenologists use when assessing quality.
    </p>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="meth-grid">

      <div class="meth-card">
        <span class="mc-tag">alcohol_density_ratio</span>
        <div class="mc-formula">alcohol ÷ density</div>
        <div class="mc-title">Body &amp; Mouthfeel</div>
        <div class="mc-body">Higher alcohol relative to density indicates greater extract and
        fuller palate weight — a key attribute of premium red wines. Density falls as
        fermentation converts sugar to alcohol.</div>
        <div class="direction-good">↑ Higher = fuller body</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">flavor_intensity</span>
        <div class="mc-formula">sulphates × alcohol</div>
        <div class="mc-title">Aroma Complexity</div>
        <div class="mc-body">Sulphates (potassium sulphate) protect volatile aromatic compounds
        from oxidation; alcohol acts as the solvent that extracts them from grape skins.
        Their product captures both preservation capacity and extraction power.</div>
        <div class="direction-good">↑ Higher = more aromatic richness</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">acidity_quality</span>
        <div class="mc-formula">pH × volatile acidity</div>
        <div class="mc-title">Fault Detection</div>
        <div class="mc-body">Volatile acidity (mainly acetic acid) above ~0.6 g/dm³ is
        perceptible as vinegar — a classic wine fault. Multiplying by pH amplifies the
        penalty for wines that are simultaneously high in volatile acidity and have
        elevated pH (poor microbial stability).</div>
        <div class="direction-bad">↓ Lower = cleaner, better-balanced</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">sugar_acid_balance</span>
        <div class="mc-formula">residual sugar ÷ fixed acidity</div>
        <div class="mc-title">Sweetness Perception</div>
        <div class="mc-body">Even in dry reds, residual sugar interacts with fixed acidity
        (tartaric, malic acids) to shape perceived roundness and length. A high ratio
        indicates excess sweetness relative to the acid backbone — atypical for quality
        dry reds. Marginal correlation ≈ −0.03, but retained for interaction effects.</div>
        <div class="direction-bad">↓ Lower = drier, more structured (dry reds)</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">so2_efficiency</span>
        <div class="mc-formula">free SO₂ ÷ alcohol</div>
        <div class="mc-title">Preservation Efficiency</div>
        <div class="mc-body">Free SO₂ prevents oxidation and microbial spoilage.
        Normalising by alcohol yields a preservation efficiency score.
        Too low → oxidation risk; too high → sulfurous off-aromas
        detectable by consumers above ~50 mg/L.</div>
        <div class="direction-range">⇔ Optimal range: 1.5 – 3.5</div>
      </div>

    </div>

    <br>
    <div style="font-size:0.73rem;color:#9b8c84;line-height:1.6;">
      <strong>References:</strong>
      Peynaud, E. (1987). <em>Knowing and Making Wine.</em> Wiley. ·
      Cortez et al. (2009). Modeling wine preferences by data mining from physicochemical properties.
      <em>Decision Support Systems, 47(4).</em> ·
      OIV (2023). International Code of Oenological Practices.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    ins_left, ins_right = st.columns([3, 2], gap="large")

    with ins_left:
        st.markdown('<p class="sec-title">Global Feature Importances</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Average reduction in Gini impurity across all trees. Higher = more influential across all 1,599 training wines.</p>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        bar_colors = []
        max_imp = imp_df["Importance"].max()
        for v in imp_df["Importance"]:
            alpha = 0.35 + 0.65 * (v / max_imp)
            bar_colors.append((0.545 * alpha + (1 - alpha),
                                0.102 * alpha + (1 - alpha),
                                0.184 * alpha + (1 - alpha)))

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor("#faf7f2")
        ax2.set_facecolor("#faf7f2")
        bars = ax2.barh(imp_df["Feature"], imp_df["Importance"],
                        color=bar_colors, height=0.5, edgecolor="none")
        for i, (_, row) in enumerate(imp_df.iterrows()):
            ax2.text(row["Importance"] + 0.003, i,
                     f'{row["Importance"]:.3f}',
                     va="center", color="#5c4a3a", fontsize=8.5)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.tick_params(colors="#5c4a3a", labelsize=8.5)
        ax2.set_xlabel("Mean Decrease in Gini Impurity", fontsize=8, color="#9b8c84")
        ax2.set_xlim(0, imp_df["Importance"].max() + 0.07)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with ins_right:
        st.markdown('<p class="sec-title">Model Transparency</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="transp-box">
          <h4>Dataset</h4>
          UCI Red Wine Quality — 1,599 Portuguese Vinho Verde wines.<br><br>

          <h4>Algorithm</h4>
          Random Forest Classifier with <code>class_weight='balanced'</code>
          and SMOTE applied inside each CV fold via <code>imblearn.Pipeline</code>
          to prevent data leakage.<br><br>

          <h4>Class Imbalance</h4>
          Only ~14% of wines score ≥ 7. A majority-class baseline achieves ~86% accuracy —
          this model is evaluated on <strong>F1 score and AUC-ROC</strong>, not raw accuracy.<br><br>

          <h4>Threshold</h4>
          Default classification threshold: 0.50. Raising it (e.g. 0.65) increases precision
          — useful when falsely labelling a wine as Premium is costly.<br><br>

          <h4>Scope</h4>
          Trained on red wine only. Results for white wines or other regions are unreliable.<br><br>

          <h4>Contribution chart</h4>
          The per-prediction chart uses importance × scaled value as a proxy.
          For rigorous local explanations, SHAP values should be used.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Wine Quality Predictor · Kavinda Pushpa Kumara · Food Science | Data Science Portfolio
</div>
""", unsafe_allow_html=True)
