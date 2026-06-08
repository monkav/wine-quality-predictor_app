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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=32&h=32&fit=crop",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Curated Unsplash image URLs ───────────────────────────────────────────────
# All free to use under Unsplash licence
IMG = {
    # Wide vineyard at golden hour — header hero banner
    "hero": "https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=1600&q=80&fit=crop&crop=center",
    # Close-up red wine being poured — predict tab accent
    "pour": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80&fit=crop&crop=center",
    # Wine bottles in cellar — methodology tab
    "cellar": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80&fit=crop&crop=center",
    # Laboratory glassware — insights tab
    "lab": "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=800&q=80&fit=crop&crop=center",
    # Vineyard rows aerial — methodology background
    "vineyard": "https://images.unsplash.com/photo-1464207687429-7505649dae38?w=800&q=80&fit=crop&crop=center",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
}}
.stApp {{ background: #faf7f2; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none; }}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ── Hero banner ── */
.hero-banner {{
    position: relative;
    width: 100%;
    height: 180px;
    overflow: hidden;
    background: #1a0a0e;
}}
.hero-banner img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 60%;
    opacity: 0.55;
    display: block;
}}
.hero-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        rgba(20,5,8,0.82) 0%,
        rgba(20,5,8,0.45) 60%,
        rgba(20,5,8,0.15) 100%
    );
    display: flex;
    align-items: center;
    padding: 0 2.5rem;
    gap: 1.5rem;
}}
.hero-text {{ flex: 1; }}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.4rem, 3vw, 2.1rem);
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
    line-height: 1.2;
    margin: 0 0 0.3rem 0;
}}
.hero-rule {{
    width: 48px;
    height: 2px;
    background: #8b1a2f;
    margin: 0.4rem 0;
}}
.hero-author {{
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.hero-badge {{
    background: rgba(139,26,47,0.85);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    color: #fff;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
}}
.hero-badge strong {{
    display: block;
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: none;
    margin-bottom: 0.1rem;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #fff;
    border-bottom: 1px solid #e8ddd5;
    padding: 0 2rem;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 99;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Lato', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #9b8c84;
    padding: 0.9rem 1.6rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}}
.stTabs [aria-selected="true"] {{
    color: #8b1a2f !important;
    border-bottom: 2px solid #8b1a2f !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding: 0 !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}

/* ── Tab content ── */
.tab-content {{
    padding: 1.4rem 2rem 2rem;
    max-width: 1280px;
    margin: 0 auto;
    width: 100%;
}}

/* ── Input panel ── */
.input-panel {{
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 12px;
    padding: 1.2rem 1.4rem 0.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}}
.panel-header {{
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #f0ebe4;
}}
.panel-header img {{
    width: 54px;
    height: 54px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
}}
.panel-header-text .ph-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #2c2118;
}}
.panel-header-text .ph-sub {{
    font-size: 0.71rem;
    color: #9b8c84;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}}

/* ── Sliders ── */
.stSlider > label {{
    font-size: 0.77rem !important;
    color: #5c4a3a !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
}}
[data-testid="stSlider"] > div > div > div > div {{
    background: #8b1a2f !important;
}}

/* ── Button ── */
.stButton > button {{
    background: #8b1a2f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.65rem 0 !important;
    width: 100%;
    transition: background 0.2s !important;
    margin-top: 0.5rem;
}}
.stButton > button:hover {{ background: #a02038 !important; }}

/* ── Result cards ── */
.result-premium {{
    background: linear-gradient(135deg, #fff5f6 0%, #fff 100%);
    border: 2px solid #8b1a2f;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 1.2rem;
    text-align: center;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
}}
.result-premium::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b1a2f, #c9406a, #8b1a2f);
}}
.result-standard {{
    background: #fff;
    border: 2px solid #d0c8c0;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 1.2rem;
    text-align: center;
    margin-bottom: 0.9rem;
}}
.result-label {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}}
.result-prob {{
    font-size: 0.83rem;
    color: #9b8c84;
}}
.result-prob strong {{ color: #2c2118; }}

/* ── Probability gauge bar ── */
.gauge-wrap {{
    margin: 0.7rem 0;
}}
.gauge-label {{
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #9b8c84;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}}
.gauge-track {{
    background: #f0ebe4;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}}
.gauge-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.4s ease;
}}

/* ── Feature chips ── */
.feat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
    margin-top: 0.5rem;
}}
.feat-chip {{
    background: #faf7f2;
    border: 1px solid #e8ddd5;
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
}}
.feat-chip .fname {{
    color: #9b8c84;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
    margin-bottom: 0.08rem;
}}
.feat-chip .fval {{
    color: #2c2118;
    font-weight: 600;
    font-family: monospace;
    font-size: 0.8rem;
}}

/* ── Section labels ── */
.sec-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #2c2118;
    margin: 0 0 0.2rem 0;
}}
.sec-sub {{
    font-size: 0.73rem;
    color: #9b8c84;
    margin-bottom: 0.75rem;
    line-height: 1.5;
}}

/* ── Methodology ── */
.meth-hero {{
    position: relative;
    width: 100%;
    height: 130px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}}
.meth-hero img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 40%;
    opacity: 0.65;
}}
.meth-hero-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(20,5,8,0.75) 0%, rgba(20,5,8,0.2) 100%);
    display: flex;
    align-items: center;
    padding: 0 1.8rem;
}}
.meth-hero-text {{
    color: #fff;
}}
.meth-hero-text h2 {{
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}}
.meth-hero-text p {{
    font-size: 0.77rem;
    opacity: 0.8;
    margin: 0;
    max-width: 520px;
    line-height: 1.45;
}}

.meth-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.85rem;
}}
.meth-card {{
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    border-top: 3px solid #8b1a2f;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}
.mc-tag {{
    font-family: monospace;
    font-size: 0.7rem;
    background: #fdf0f2;
    color: #8b1a2f;
    padding: 0.12rem 0.42rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.35rem;
}}
.mc-formula {{
    font-size: 0.75rem;
    color: #9b8c84;
    font-family: monospace;
    margin-bottom: 0.35rem;
}}
.mc-title {{
    font-family: 'Playfair Display', serif;
    font-size: 0.93rem;
    font-weight: 600;
    color: #2c2118;
    margin-bottom: 0.28rem;
}}
.mc-body {{
    font-size: 0.77rem;
    color: #5c4a3a;
    line-height: 1.55;
}}
.dir-good  {{ color: #4a7c59; font-size: 0.71rem; font-weight: 500; margin-top: 0.3rem; }}
.dir-bad   {{ color: #8b1a2f; font-size: 0.71rem; font-weight: 500; margin-top: 0.3rem; }}
.dir-range {{ color: #7a6a30; font-size: 0.71rem; font-weight: 500; margin-top: 0.3rem; }}

/* ── Insights ── */
.insights-image-strip {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}}
.insight-img-card {{
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    height: 90px;
}}
.insight-img-card img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.7;
}}
.insight-img-label {{
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(0deg, rgba(20,5,8,0.8) 0%, transparent 100%);
    padding: 0.4rem 0.7rem;
    font-size: 0.7rem;
    color: #fff;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 500;
}}

/* ── Transparency box ── */
.transp-box {{
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.79rem;
    color: #5c4a3a;
    line-height: 1.65;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}
.transp-box h4 {{
    font-family: 'Playfair Display', serif;
    font-size: 0.92rem;
    color: #2c2118;
    margin: 0.7rem 0 0.2rem 0;
}}
.transp-box h4:first-child {{ margin-top: 0; }}

/* ── Footer ── */
.app-footer {{
    background: #fff;
    border-top: 1px solid #e8ddd5;
    padding: 0.75rem 2rem;
    text-align: center;
    font-size: 0.7rem;
    color: #c8bdb8;
    letter-spacing: 0.06em;
}}

/* ── Mobile ── */
@media (max-width: 640px) {{
    .hero-banner {{ height: 140px; }}
    .hero-badge {{ display: none; }}
    .tab-content {{ padding: 1rem; }}
    .feat-grid {{ grid-template-columns: 1fr 1fr; }}
    .insights-image-strip {{ grid-template-columns: 1fr; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.75rem 0.85rem; font-size: 0.73rem; }}
}}
@media (max-width: 400px) {{
    .feat-grid {{ grid-template-columns: 1fr; }}
    .meth-grid {{ grid-template-columns: 1fr; }}
    .hero-title {{ font-size: 1.15rem; }}
}}
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


# ══════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <img src="{IMG['hero']}" alt="Vineyard at golden hour">
    <div class="hero-overlay">
        <div class="hero-text">
            <div class="hero-title">Wine Quality Predictor</div>
            <div class="hero-rule"></div>
            <div class="hero-author">
                Kavinda Pushpa Kumara &nbsp;&middot;&nbsp;
                Food Science Student &nbsp;&middot;&nbsp;
                IBM Certified Data Scientist
            </div>
        </div>
        <div class="hero-badge">
            <strong>UCI</strong>
            Red Wine Dataset<br>1,599 Wines
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_predict, tab_method, tab_insights = st.tabs([
    "  Predict  ",
    "  Methodology  ",
    "  Model Insights  ",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    col_inputs, col_result = st.columns([1, 1], gap="large")

    # ── Left: inputs ──────────────────────────────────────────────────
    with col_inputs:
        st.markdown(f"""
        <div class="input-panel">
            <div class="panel-header">
                <img src="{IMG['pour']}" alt="Wine being poured">
                <div class="panel-header-text">
                    <div class="ph-title">Raw Chemical Measurements</div>
                    <div class="ph-sub">Enter lab values — features are engineered automatically</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0,
                                float(MEANS["alcohol"]), 0.1,
                                help="Percentage alcohol by volume")
            sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00,
                                  float(MEANS["sulphates"]), 0.01,
                                  help="Potassium sulphate — preservative & aroma enhancer")
            pH = st.slider("pH", 2.70, 4.50,
                           float(MEANS["pH"]), 0.01,
                           help="Acidity level — lower = more acidic")
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0,
                                       float(MEANS["residual_sugar"]), 0.1,
                                       help="Unfermented sugar remaining after fermentation")
        with c2:
            density = st.slider("Density (g/cm³)", 0.990, 1.004,
                                float(MEANS["density"]), 0.0001,
                                format="%.4f",
                                help="Wine density — related to alcohol and sugar content")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60,
                                         float(MEANS["volatile_acidity"]), 0.01,
                                         help="Acetic acid — high levels produce a vinegar taste")
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0,
                                      float(MEANS["fixed_acidity"]), 0.1,
                                      help="Tartaric acid — forms the structural backbone")
            free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0,
                                 float(MEANS["free_sulfur_dioxide"]), 0.5,
                                 help="Free sulfur dioxide — prevents oxidation and microbial growth")

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyse Wine", use_container_width=True)

    # ── Right: result ──────────────────────────────────────────────────
    with col_result:
        feats = engineer_features(alcohol, density, sulphates, pH,
                                   volatile_acidity, residual_sugar,
                                   fixed_acidity, free_so2)

        if predict_btn:
            label, prob, scaled_vals = make_prediction(feats)

            # ── Result card ──
            if label == 1:
                gauge_color = "#8b1a2f"
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label" style="color:#8b1a2f;">Premium</div>
                    <div class="result-prob">
                        Quality predicted &ge; 7
                    </div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence: <strong style="color:#2c2118">{prob*100:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{prob*100:.1f}%;background:{gauge_color};"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("Chemical profile meets premium criteria. Strong body, aromatic complexity, and controlled acidity all contribute positively.")
            else:
                gauge_color = "#b0a8a0"
                conf = (1 - prob) * 100
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label" style="color:#5c4a3a;">Non-Premium</div>
                    <div class="result-prob">
                        Quality predicted &lt; 7
                    </div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence: <strong style="color:#2c2118">{conf:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{prob*100:.1f}%;background:#b0a8a0;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("Profile does not meet the premium threshold. Volatile acidity and flavour intensity are the primary levers for quality improvement.")

            # ── Contribution chart ──
            st.markdown('<p class="sec-title" style="margin-top:0.8rem;">Why this decision?</p>', unsafe_allow_html=True)
            st.markdown('<p class="sec-sub">Each feature\'s contribution = importance &times; scaled value. Burgundy pushes toward Premium; grey pushes toward Non-Premium.</p>', unsafe_allow_html=True)

            contributions = importances * scaled_vals
            contrib_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in contrib_df["Contribution"]]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                    color=colors, height=0.5, edgecolor="none")
            ax.axvline(0, color="#e8ddd5", linewidth=1)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#5c4a3a", labelsize=8)
            ax.set_xlabel("Contribution to decision", fontsize=8, color="#9b8c84")
            pos_p = mpatches.Patch(color="#8b1a2f", label="Toward Premium")
            neg_p = mpatches.Patch(color="#b0a8a0", label="Toward Non-Premium")
            ax.legend(handles=[pos_p, neg_p], framealpha=0,
                      labelcolor="#5c4a3a", fontsize=7.5, loc="lower right")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            # Placeholder before first prediction
            st.markdown(f"""
            <div style="position:relative;border-radius:12px;overflow:hidden;height:160px;margin-bottom:0.9rem;">
                <img src="{IMG['pour']}" style="width:100%;height:100%;object-fit:cover;opacity:0.35;" alt="Wine">
                <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                            flex-direction:column;gap:0.4rem;background:rgba(250,247,242,0.6);">
                    <div style="font-family:'Playfair Display',serif;font-size:1rem;color:#5c4a3a;font-weight:600;">
                        Awaiting Analysis
                    </div>
                    <div style="font-size:0.75rem;color:#9b8c84;">
                        Set measurements on the left and press <strong style="color:#8b1a2f;">Analyse Wine</strong>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Engineered feature chips — always visible ──
        st.markdown('<p class="sec-title">Engineered Features</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Calculated live from your inputs. These are what the model actually sees.</p>', unsafe_allow_html=True)

        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity":      "Flavour Intensity",
            "acidity_quality":       "Acidity Quality",
            "sugar_acid_balance":    "Sugar / Acid Balance",
            "so2_efficiency":        "SO2 Efficiency",
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

    # Hero image strip
    st.markdown(f"""
    <div class="meth-hero">
        <img src="{IMG['cellar']}" alt="Wine cellar">
        <div class="meth-hero-overlay">
            <div class="meth-hero-text">
                <h2>Scientific Methodology</h2>
                <p>Five chemically meaningful ratios engineered from raw lab data —
                each encoding a principle winemakers use to assess quality.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="meth-grid">

      <div class="meth-card">
        <span class="mc-tag">alcohol_density_ratio</span>
        <div class="mc-formula">alcohol &divide; density</div>
        <div class="mc-title">Body &amp; Mouthfeel</div>
        <div class="mc-body">Higher alcohol relative to density indicates greater extract and
        fuller palate weight — a key attribute of premium red wines. Density decreases
        as fermentation converts sugar to alcohol.</div>
        <div class="dir-good">Higher = fuller body</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">flavor_intensity</span>
        <div class="mc-formula">sulphates &times; alcohol</div>
        <div class="mc-title">Aroma Complexity</div>
        <div class="mc-body">Sulphates protect volatile aromatic compounds from oxidation;
        alcohol acts as the solvent that extracts them from grape skins.
        Their product captures both preservation capacity and extraction power.</div>
        <div class="dir-good">Higher = more aromatic richness</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">acidity_quality</span>
        <div class="mc-formula">pH &times; volatile acidity</div>
        <div class="mc-title">Fault Detection</div>
        <div class="mc-body">Volatile acidity above ~0.6 g/dm&sup3; is perceptible as vinegar —
        a classic fault. Multiplying by pH amplifies the penalty for wines
        simultaneously high in volatile acidity and elevated pH.</div>
        <div class="dir-bad">Lower = cleaner, better-balanced</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">sugar_acid_balance</span>
        <div class="mc-formula">residual sugar &divide; fixed acidity</div>
        <div class="mc-title">Sweetness Perception</div>
        <div class="mc-body">Even in dry reds, residual sugar interacts with fixed acidity
        to shape perceived roundness. A high ratio indicates excess sweetness relative
        to the acid backbone — atypical for quality dry reds.</div>
        <div class="dir-bad">Lower = drier, more structured</div>
      </div>

      <div class="meth-card">
        <span class="mc-tag">so2_efficiency</span>
        <div class="mc-formula">free SO&sub2; &divide; alcohol</div>
        <div class="mc-title">Preservation Efficiency</div>
        <div class="mc-body">Free SO&sub2; prevents oxidation and microbial spoilage.
        Normalising by alcohol yields a preservation efficiency score —
        too low risks oxidation; too high produces sulfurous off-aromas.</div>
        <div class="dir-range">Optimal range: 1.5 &ndash; 3.5</div>
      </div>

    </div>

    <br>
    <div style="font-size:0.71rem;color:#9b8c84;line-height:1.6;">
      <strong>References:</strong>
      Peynaud, E. (1987). <em>Knowing and Making Wine.</em> Wiley. &middot;
      Cortez et al. (2009). Modeling wine preferences by data mining from physicochemical properties.
      <em>Decision Support Systems, 47(4).</em> &middot;
      OIV (2023). International Code of Oenological Practices.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Image strip at top
    st.markdown(f"""
    <div class="insights-image-strip">
        <div class="insight-img-card">
            <img src="{IMG['vineyard']}" alt="Vineyard rows">
            <div class="insight-img-label">1,599 Red Wines &middot; UCI Dataset</div>
        </div>
        <div class="insight-img-card">
            <img src="{IMG['lab']}" alt="Laboratory analysis">
            <div class="insight-img-label">Random Forest &middot; 5-Fold CV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ins_left, ins_right = st.columns([3, 2], gap="large")

    with ins_left:
        st.markdown('<p class="sec-title">Global Feature Importances</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Mean decrease in Gini impurity across all decision trees. Higher = more influential to the model\'s decisions globally.</p>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        max_imp = imp_df["Importance"].max()
        bar_colors = []
        for v in imp_df["Importance"]:
            t = 0.35 + 0.65 * (v / max_imp)
            bar_colors.append((
                0.545 * t + 1 * (1 - t),
                0.102 * t + 1 * (1 - t),
                0.184 * t + 1 * (1 - t),
            ))

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor("#faf7f2")
        ax2.set_facecolor("#faf7f2")
        ax2.barh(imp_df["Feature"], imp_df["Importance"],
                 color=bar_colors, height=0.5, edgecolor="none")
        for i, (_, row) in enumerate(imp_df.iterrows()):
            ax2.text(row["Importance"] + 0.003, i,
                     f'{row["Importance"]:.3f}',
                     va="center", color="#5c4a3a", fontsize=8.5)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.tick_params(colors="#5c4a3a", labelsize=8.5)
        ax2.set_xlabel("Mean Decrease in Gini Impurity", fontsize=8, color="#9b8c84")
        ax2.set_xlim(0, max_imp + 0.07)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with ins_right:
        st.markdown('<p class="sec-title">Model Transparency</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="transp-box">
          <h4>Dataset</h4>
          UCI Red Wine Quality — 1,599 Portuguese Vinho Verde wines.

          <h4>Algorithm</h4>
          Random Forest with <code>class_weight='balanced'</code> and SMOTE
          applied inside each CV fold via <code>imblearn.Pipeline</code> to prevent data leakage.

          <h4>Class Imbalance</h4>
          Only ~14% of wines score &ge; 7. A majority-class baseline achieves ~86% accuracy —
          this model is evaluated on <strong>F1 score and AUC-ROC</strong>.

          <h4>Threshold</h4>
          Default threshold: 0.50. Raising it (e.g. 0.65) increases precision —
          important when falsely labelling a wine as Premium is costly.

          <h4>Scope</h4>
          Trained on red wine only. White wines or other regions are out of scope.

          <h4>Contribution Chart</h4>
          Uses importance &times; scaled value as a proxy.
          For rigorous local explanations, SHAP values should be used.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Wine Quality Predictor &nbsp;&middot;&nbsp; Kavinda Pushpa Kumara
    &nbsp;&middot;&nbsp; Food Science &amp; Data Science Portfolio
</div>
""", unsafe_allow_html=True)
