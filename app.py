"""
Wine Quality Prediction — Streamlit Application (Redesigned)
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
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Image URLs ────────────────────────────────────────────────────────────────
IMG = {
    "hero":     "https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=1600&q=80&fit=crop&crop=center",
    "pour":     "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80&fit=crop&crop=center",
    "cellar":   "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80&fit=crop&crop=center",
    "lab":      "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=800&q=80&fit=crop&crop=center",
    "vineyard": "https://images.unsplash.com/photo-1464207687429-7505649dae38?w=800&q=80&fit=crop&crop=center",
}

# ── Enhanced CSS with better mobile support ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
}
.stApp { background: #faf7f2; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Hero Banner - Improved mobile */
.hero-banner {
    position: relative; width: 100%; height: 180px;
    overflow: hidden; background: #1a0a0e;
}
.hero-banner img {
    width: 100%; height: 100%; object-fit: cover;
    object-position: center 60%; opacity: 0.55; display: block;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(15,4,6,0.9) 0%, rgba(15,4,6,0.6) 50%, rgba(15,4,6,0.2) 100%);
    display: flex; align-items: center;
    padding: 0 1.5rem; gap: 1.5rem;
}
.hero-text { flex: 1; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.4rem, 5vw, 2.4rem);
    font-weight: 700; color: #fff;
    letter-spacing: 0.01em; line-height: 1.15; margin: 0 0 0.4rem;
}
.hero-rule { width: 50px; height: 3px; background: #8b1a2f; margin: 0.4rem 0; }
.hero-author {
    font-size: 0.8rem; color: rgba(255,255,255,0.75);
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* Tabs - Better mobile */
.stTabs [data-baseweb="tab-list"] {
    background: #fff; border-bottom: 1px solid #e8ddd5;
    padding: 0 1rem; gap: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem; font-weight: 500; padding: 1rem 1.2rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] { padding: 0.85rem 0.9rem; font-size: 0.78rem; }
}

/* Tab content */
.tab-content {
    padding: 1.5rem 2rem 2rem;
    max-width: 1280px; margin: 0 auto;
}
@media (max-width: 768px) {
    .tab-content { padding: 1rem 1.1rem; }
}

/* Input panel */
.input-panel {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.panel-header {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0ebe4;
}
.panel-header img {
    width: 55px; height: 55px; border-radius: 8px; object-fit: cover;
}

/* Sliders & Button */
.stSlider > label { font-size: 0.82rem !important; font-weight: 500; }
.stButton > button {
    background: #8b1a2f !important; color: #fff !important;
    border-radius: 8px !important; font-weight: 600;
    padding: 0.75rem !important; width: 100%; margin-top: 1rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.stButton > button:hover { background: #a02038 !important; }

/* Result cards */
.result-premium, .result-standard {
    border-radius: 12px; padding: 1.4rem; text-align: center;
    margin-bottom: 1rem;
}
.result-premium {
    background: linear-gradient(135deg, #fff5f6 0%, #fff 100%);
    border: 2px solid #8b1a2f;
}
.result-standard {
    background: #fff; border: 2px solid #d0c8c0;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem; font-weight: 700; margin-bottom: 0.3rem;
}

/* Gauge */
.gauge-track {
    background: #f0ebe4; border-radius: 99px; height: 8px;
}
.gauge-fill { height: 100%; border-radius: 99px; }

/* Feature chips */
.feat-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;
    margin-top: 0.8rem;
}
@media (max-width: 640px) {
    .feat-grid { grid-template-columns: 1fr; }
}
.feat-chip {
    background: #faf7f2; border: 1px solid #e8ddd5;
    border-radius: 8px; padding: 0.5rem 0.8rem;
}

/* Methodology cards */
.meth-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
}
.meth-card {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 10px; padding: 1.1rem;
    border-top: 4px solid #8b1a2f;
}

/* Footer */
.app-footer {
    background: #fff; border-top: 1px solid #e8ddd5;
    padding: 1rem 2rem; text-align: center;
    font-size: 0.75rem; color: #9b8c84;
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
    except FileNotFoundError:
        st.error("Model files (wine_model.pkl and scaler.pkl) not found.")
        st.stop()

model, scaler = load_model()

FEATURE_NAMES = [
    "alcohol_density_ratio", "flavor_intensity", "acidity_quality",
    "sugar_acid_balance", "so2_efficiency"
]

# Dataset means
MEANS = {
    "alcohol": 10.42, "density": 0.9967, "sulphates": 0.658,
    "pH": 3.311, "volatile_acidity": 0.528, "residual_sugar": 2.539,
    "fixed_acidity": 8.32, "free_sulfur_dioxide": 15.87,
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

def get_food_science_notes(feats):
    notes = []
    adr = feats["alcohol_density_ratio"]
    fi = feats["flavor_intensity"]
    aq = feats["acidity_quality"]
    so2 = feats["so2_efficiency"]
    sab = feats["sugar_acid_balance"]

    # Body
    if adr > 12.5:
        notes.append(("Body", "Full-bodied with excellent structure and weight.", "good"))
    elif adr < 11.0:
        notes.append(("Body", "Light-bodied — may feel thin on the palate.", "warn"))
    else:
        notes.append(("Body", f"Balanced medium body (ratio {adr:.2f}).", "neutral"))

    # Aroma
    if fi > 9.0:
        notes.append(("Aroma", "Rich and complex aromatics expected.", "good"))
    elif fi < 6.0:
        notes.append(("Aroma", "Subtle aroma profile — could benefit from more intensity.", "warn"))
    else:
        notes.append(("Aroma", f"Moderate aromatic intensity ({fi:.2f}).", "neutral"))

    # Acidity
    if aq < 2.0:
        notes.append(("Acidity", "Clean and well-balanced acidity.", "good"))
    elif aq > 3.0:
        notes.append(("Acidity", "Elevated volatile acidity risk — possible off-notes.", "warn"))
    else:
        notes.append(("Acidity", f"Acceptable acidity balance ({aq:.2f}).", "neutral"))

    # Preservation
    if 1.5 <= so2 <= 3.5:
        notes.append(("Preservation", "Excellent SO₂ efficiency for stability.", "good"))
    elif so2 < 1.5:
        notes.append(("Preservation", "Higher risk of oxidation or spoilage.", "warn"))
    else:
        notes.append(("Preservation", "Very high SO₂ levels — may show sulfur notes.", "warn"))

    # Balance
    if sab < 0.3:
        notes.append(("Balance", "Beautifully dry with great structure.", "good"))
    elif sab > 0.8:
        notes.append(("Balance", "Higher residual sugar relative to acidity.", "warn"))
    else:
        notes.append(("Balance", f"Good sugar-acid harmony ({sab:.3f}).", "neutral"))

    return notes

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <img src="{IMG['hero']}" alt="Vineyard">
    <div class="hero-overlay">
        <div class="hero-text">
            <div class="hero-title">Wine Quality Predictor</div>
            <div class="hero-rule"></div>
            <div class="hero-author">
                Kavinda Pushpa Kumara • Food Science Student &amp; IBM Certified Data Scientist
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS (Only Predict + Methodology)
# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_method = st.tabs(["  Predict Quality  ", "  Methodology  "])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: PREDICT
# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f"""
        <div class="input-panel">
            <div class="panel-header">
                <img src="{IMG['pour']}" alt="Wine pour">
                <div>
                    <h3 style="margin:0; font-family:'Playfair Display',serif;">Raw Chemical Measurements</h3>
                    <p style="margin:0; color:#9b8c84; font-size:0.85rem;">Enter your lab values below</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1)
            sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01)
            pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01)
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1)

        with c2:
            density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01)
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1)
            free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5)

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Analyse This Wine", use_container_width=True, type="primary")

    with col2:
        feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity,
                                  residual_sugar, fixed_acidity, free_so2)

        if predict_btn:
            label, prob, scaled_vals = make_prediction(feats)
            st.session_state["last_label"] = label
            st.session_state["last_prob"] = prob
            st.session_state["last_feats"] = feats

        if "last_prob" in st.session_state:
            label = st.session_state["last_label"]
            prob = st.session_state["last_prob"]
            feats_saved = st.session_state["last_feats"]

            if label == 1:
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label" style="color:#8b1a2f;">Premium Quality</div>
                    <div style="font-size:0.95rem;color:#555;">Predicted quality ≥ 7</div>
                    <div class="gauge-wrap" style="margin:1rem 0;">
                        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#777;margin-bottom:4px;">
                            <span>Standard</span>
                            <span><strong>{prob*100:.1f}%</strong> Confidence</span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track"><div class="gauge-fill" style="width:{prob*100:.1f}%; background:#8b1a2f;"></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("This wine shows the chemical signature of a premium bottle.")
            else:
                conf = (1 - prob) * 100
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label" style="color:#5c4a3a;">Standard Quality</div>
                    <div style="font-size:0.95rem;color:#555;">Predicted quality &lt; 7</div>
                    <div class="gauge-wrap" style="margin:1rem 0;">
                        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#777;margin-bottom:4px;">
                            <span>Standard</span>
                            <span><strong>{conf:.1f}%</strong> Confidence</span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track"><div class="gauge-fill" style="width:{prob*100:.1f}%; background:#b0a8a0;"></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("This wine falls into the standard quality category.")

            # Explanation
            if st.button("Explain This Prediction", use_container_width=True):
                st.session_state["show_exp"] = True

            if st.session_state.get("show_exp", False):
                st.subheader("Why this prediction?")
                contributions = model.feature_importances_ * scaled_vals
                contrib_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Contribution": contributions
                }).sort_values("Contribution", key=abs, ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in contrib_df["Contribution"]]
                ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors)
                ax.axvline(0, color="#ddd", lw=1)
                ax.set_xlabel("Contribution to Prediction")
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Food science notes
                st.subheader("Winemaker's Perspective")
                notes = get_food_science_notes(feats_saved)
                for aspect, text, tone in notes:
                    color = "#e8f5ee" if tone == "good" else "#fdf0f2" if tone == "warn" else "#f9f6f0"
                    st.markdown(f"""
                    <div style="background:{color}; padding:0.9rem; border-radius:8px; margin:0.5rem 0; border-left:4px solid {'#4a7c59' if tone=='good' else '#8b1a2f'}">
                        <strong>{aspect}:</strong> {text}
                    </div>""", unsafe_allow_html=True)

        else:
            st.info("Adjust the values on the left and click **Analyse This Wine** to get a prediction.")

        # Engineered Features
        st.subheader("Engineered Features")
        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity": "Flavour Intensity",
            "acidity_quality": "Acidity Quality",
            "sugar_acid_balance": "Sugar-Acid Balance",
            "so2_efficiency": "SO₂ Efficiency"
        }
        chips = '<div class="feat-grid">'
        for k, v in feats.items():
            chips += f'<div class="feat-chip"><strong>{feat_labels[k]}</strong><br><span style="font-family:monospace;">{v:.4f}</span></div>'
        chips += '</div>'
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: METHODOLOGY (Humanized & Professional)
# ──────────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="position:relative; height:140px; border-radius:12px; overflow:hidden; margin-bottom:1.5rem;">
        <img src="{IMG['cellar']}" style="width:100%; height:100%; object-fit:cover; opacity:0.7;">
        <div style="position:absolute; inset:0; background:linear-gradient(transparent, rgba(26,10,14,0.85)); display:flex; align-items:flex-end; padding:1.5rem;">
            <div>
                <h2 style="color:white; margin:0; font-family:'Playfair Display',serif;">The Science Behind the Prediction</h2>
                <p style="color:#ddd; margin:0.4rem 0 0;">Turning chemistry into winemaking wisdom</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:1.05rem; line-height:1.7; color:#444;">
        Great wine is more than just numbers — it's the careful balance of chemistry, craftsmanship, and nature.
        This predictor uses five thoughtfully engineered features that winemakers and food scientists use every day to evaluate quality.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="meth-grid">
      <div class="meth-card">
        <strong style="color:#8b1a2f; font-size:1.1rem;">Alcohol / Density Ratio</strong>
        <p><em>Body &amp; Mouthfeel</em></p>
        <p>This ratio tells us how full and structured the wine feels. Higher alcohol relative to density usually means richer extract and a more satisfying weight on the palate — a hallmark of quality reds.</p>
      </div>
      <div class="meth-card">
        <strong style="color:#8b1a2f; font-size:1.1rem;">Flavour Intensity</strong>
        <p><em>Aroma &amp; Complexity</em></p>
        <p>Calculated as sulphates × alcohol. Sulphates help preserve delicate aromas while alcohol helps extract them. This feature captures the aromatic potential of the wine.</p>
      </div>
      <div class="meth-card">
        <strong style="color:#8b1a2f; font-size:1.1rem;">Acidity Quality</strong>
        <p><em>Fault Prevention</em></p>
        <p>Volatile acidity (the vinegar note) multiplied by pH. This helps flag wines that might have off-flavours. Lower values mean cleaner, more elegant acidity.</p>
      </div>
      <div class="meth-card">
        <strong style="color:#8b1a2f; font-size:1.1rem;">Sugar / Acid Balance</strong>
        <p><em>Harmony on the Palate</em></p>
        <p>Residual sugar relative to fixed acidity. Quality dry reds usually show a low ratio — creating that refreshing, structured finish we all love.</p>
      </div>
      <div class="meth-card">
        <strong style="color:#8b1a2f; font-size:1.1rem;">SO₂ Efficiency</strong>
        <p><em>Preservation &amp; Stability</em></p>
        <p>Free sulfur dioxide normalized by alcohol. The right balance protects the wine from oxidation and spoilage without leaving unpleasant sulfur notes.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="margin-top:2rem; font-size:0.9rem; color:#777; line-height:1.6;">
        <strong>Trained on 1,599 real Portuguese red wines.</strong> The model achieved 91.6% accuracy and strong performance on identifying premium bottles (quality 7+).
        This tool is designed to support winemakers, sommeliers, and wine enthusiasts in understanding what makes a wine exceptional.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    Wine Quality Predictor • Kavinda Pushpa Kumara • Built with ❤️ for wine lovers
</div>
""", unsafe_allow_html=True)
""
