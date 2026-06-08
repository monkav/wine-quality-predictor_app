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
    page_icon="https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=32&h=32&fit=crop",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Image URLs ───────────────────────────────────────────────────────────────
IMG = {
    "hero":     "https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=1600&q=80&fit=crop&crop=center",
    "pour":     "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80&fit=crop&crop=center",
    "cellar":   "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80&fit=crop&crop=center",
    "lab":      "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=800&q=80&fit=crop&crop=center",
    "vineyard": "https://images.unsplash.com/photo-1464207687429-7505649dae38?w=800&q=80&fit=crop&crop=center",
}

# ── CSS with enhanced mobile responsiveness ───────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
}}
.stApp {{ background: #faf7f2; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

/* Hero */
.hero-banner {{
    position: relative; width: 100%; height: 180px;
    overflow: hidden; background: #1a0a0e;
}}
.hero-banner img {{
    width: 100%; height: 100%; object-fit: cover;
    object-position: center 60%; opacity: 0.5;
}}
.hero-overlay {{
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(15,4,6,0.88) 0%, rgba(15,4,6,0.5) 55%, rgba(15,4,6,0.1) 100%);
    display: flex; align-items: center;
    padding: 0 1.5rem; gap: 1.5rem;
}}
.hero-text {{ flex: 1; }}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.4rem, 5vw, 2.2rem);
    font-weight: 700; color: #fff;
    letter-spacing: 0.01em; line-height: 1.1;
}}
.hero-rule {{ width: 44px; height: 2px; background: #8b1a2f; margin: 0.4rem 0; }}
.hero-author {{
    font-size: 0.78rem; color: rgba(255,255,255,0.7);
    letter-spacing: 0.1em; text-transform: uppercase;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: #fff; border-bottom: 1px solid #e8ddd5;
    padding: 0 1rem;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.85rem; font-weight: 500; padding: 0.9rem 1.2rem;
}}
@media (max-width: 768px) {{
    .stTabs [data-baseweb="tab"] {{ font-size: 0.78rem; padding: 0.75rem 0.9rem; }}
}}

/* Input & Result */
.input-panel, .result-premium, .result-standard, .explanation-panel {{
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.tab-content {{
    padding: 1.5rem 2rem;
    max-width: 1280px; margin: 0 auto;
}}
@media (max-width: 768px) {{
    .tab-content {{ padding: 1rem 1rem; }}
}}

/* Buttons */
.stButton > button {{
    background: #8b1a2f !important; color: #fff !important;
    border-radius: 8px !important; font-weight: 600;
    padding: 0.75rem !important; width: 100%;
}}
.stButton > button:hover {{ background: #a02038 !important; }}

/* Gauge & Features */
.gauge-track {{ height: 8px; }}
.feat-grid {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.6rem; }}

/* Methodology Cards */
.meth-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
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
    except FileNotFoundError:
        st.error("Model files (wine_model.pkl and scaler.pkl) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler = load_model()

FEATURE_NAMES = [
    "alcohol_density_ratio", "flavor_intensity", "acidity_quality",
    "sugar_acid_balance", "so2_efficiency"
]

# Try to get feature importances safely
try:
    importances = model.feature_importances_
except AttributeError:
    # Fallback if model doesn't support it (e.g. other classifier)
    importances = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

# ── Dataset means ─────────────────────────────────────────────────────────────
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
        notes.append(("Body", "Full-bodied with excellent structure and palate weight.", "good"))
    elif adr < 11.0:
        notes.append(("Body", "Light-bodied — may feel thin on the palate.", "warn"))
    else:
        notes.append(("Body", f"Balanced medium body (ratio {adr:.2f}).", "neutral"))

    # Aroma
    if fi > 9.0:
        notes.append(("Aroma", "Rich aromatic intensity and complexity.", "good"))
    elif fi < 6.0:
        notes.append(("Aroma", "Subtle aroma — could benefit from more intensity.", "warn"))
    else:
        notes.append(("Aroma", f"Moderate flavour intensity ({fi:.2f}).", "neutral"))

    # Acidity
    if aq < 2.0:
        notes.append(("Acidity", "Clean, fresh acidity with good balance.", "good"))
    elif aq > 3.0:
        notes.append(("Acidity", "Elevated volatile acidity — risk of off-flavours.", "warn"))
    else:
        notes.append(("Acidity", f"Acceptable acidity balance ({aq:.2f}).", "neutral"))

    # Preservation
    if 1.5 <= so2 <= 3.5:
        notes.append(("Preservation", "Excellent SO₂ efficiency for stability.", "good"))
    elif so2 < 1.5:
        notes.append(("Preservation", "Low preservation capacity — oxidation risk.", "warn"))
    else:
        notes.append(("Preservation", "High SO₂ levels — may show sulfur notes.", "warn"))

    # Balance
    if sab < 0.3:
        notes.append(("Balance", "Well-balanced dry style with great structure.", "good"))
    elif sab > 0.8:
        notes.append(("Balance", "Higher residual sugar relative to acidity.", "warn"))
    else:
        notes.append(("Balance", f"Good sugar-acid harmony ({sab:.3f}).", "neutral"))

    return notes

# HERO
st.markdown(f"""
<div class="hero-banner">
    <img src="{IMG['hero']}" alt="Vineyard">
    <div class="hero-overlay">
        <div class="hero-text">
            <div class="hero-title">Wine Quality Predictor</div>
            <div class="hero-rule"></div>
            <div class="hero-author">Kavinda Pushpa Kumara • Food Scientist & Data Scientist</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS - Only Predict and Methodology
tab_predict, tab_method = st.tabs(["  Predict  ", "  Methodology  "])

# TAB 1 — PREDICT
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown(f"""
        <div class="input-panel" style="background:#fff; padding:1.2rem; border:1px solid #e8ddd5;">
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem;">
                <img src="{IMG['pour']}" style="width:48px;height:48px;border-radius:8px;" alt="Pour">
                <div>
                    <strong style="font-size:1.05rem;">Raw Chemical Measurements</strong><br>
                    <small style="color:#9b8c84;">Enter lab values — features engineered automatically</small>
                </div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.slider("Alcohol (%vol)", 8.0, 15.0, float(MEANS["alcohol"]), 0.1)
            sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, float(MEANS["sulphates"]), 0.01)
            pH = st.slider("pH", 2.70, 4.50, float(MEANS["pH"]), 0.01)
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 1.0, 16.0, float(MEANS["residual_sugar"]), 0.1)
        with c2:
            density = st.slider("Density (g/cm³)", 0.990, 1.004, float(MEANS["density"]), 0.0001, format="%.4f")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, float(MEANS["volatile_acidity"]), 0.01)
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(MEANS["fixed_acidity"]), 0.1)
            free_so2 = st.slider("Free SO₂ (mg/dm³)", 1.0, 72.0, float(MEANS["free_sulfur_dioxide"]), 0.5)

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyse This Wine", use_container_width=True, type="primary")

    with col_result:
        feats = engineer_features(alcohol, density, sulphates, pH, volatile_acidity, residual_sugar, fixed_acidity, free_so2)

        if predict_btn:
            label, prob, scaled_vals = make_prediction(feats)
            st.session_state["last_label"] = label
            st.session_state["last_prob"] = prob
            st.session_state["last_scaled"] = scaled_vals
            st.session_state["last_feats"] = feats
            st.session_state["show_explanation"] = False

        if "last_prob" in st.session_state:
            label = st.session_state["last_label"]
            prob = st.session_state["last_prob"]
            scaled_vals = st.session_state["last_scaled"]
            feats_saved = st.session_state["last_feats"]

            if label == 1:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#fff5f6 0%,#fff 100%); border:2px solid #8b1a2f; border-radius:12px; padding:1.5rem; text-align:center;">
                    <div style="font-size:2.1rem; font-weight:700; color:#8b1a2f;">Premium Quality</div>
                    <div style="color:#8b1a2f; margin:0.4rem 0;">Quality ≥ 7 predicted</div>
                    <div style="margin:1rem 0;">
                        <div style="font-size:0.8rem; color:#666;">Confidence</div>
                        <div style="height:10px; background:#e8ddd5; border-radius:999px; overflow:hidden;">
                            <div style="width:{prob*100:.1f}%; height:100%; background:#8b1a2f;"></div>
                        </div>
                        <div style="text-align:right; font-weight:600; margin-top:4px;">{prob*100:.1f}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                conf = (1 - prob) * 100
                st.markdown(f"""
                <div style="background:#fff; border:2px solid #d0c8c0; border-radius:12px; padding:1.5rem; text-align:center;">
                    <div style="font-size:2.1rem; font-weight:700; color:#5c4a3a;">Standard Quality</div>
                    <div style="margin:0.4rem 0;">Quality < 7 predicted</div>
                    <div style="margin:1rem 0;">
                        <div style="font-size:0.8rem; color:#666;">Confidence</div>
                        <div style="height:10px; background:#e8ddd5; border-radius:999px; overflow:hidden;">
                            <div style="width:{prob*100:.1f}%; height:100%; background:#b0a8a0;"></div>
                        </div>
                        <div style="text-align:right; font-weight:600; margin-top:4px;">{conf:.1f}% Non-Premium</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            if st.button("Explain This Prediction", use_container_width=True):
                st.session_state["show_explanation"] = not st.session_state.get("show_explanation", False)

            if st.session_state.get("show_explanation", False):
                st.markdown('<div class="explanation-panel" style="background:#fff; padding:1.2rem; margin-top:1rem;">', unsafe_allow_html=True)

                st.subheader("Feature Contributions")
                contributions = importances * scaled_vals
                contrib_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Contribution": contributions,
                }).sort_values("Contribution", key=abs, ascending=True)

                colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in contrib_df["Contribution"]]

                fig, ax = plt.subplots(figsize=(6, 2.8))
                ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors)
                ax.axvline(0, color="#ddd", linewidth=1)
                ax.set_xlabel("Contribution to Prediction")
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.subheader("Food Science Interpretation")
                notes = get_food_science_notes(feats_saved)
                for aspect, text, tone in notes:
                    color = "#e8f5ee" if tone == "good" else "#fdf0f2" if tone == "warn" else "#faf7f2"
                    st.markdown(f"""
                    <div style="background:{color}; padding:0.9rem; border-radius:8px; margin-bottom:0.6rem; border-left:4px solid {'#4a7c59' if tone=='good' else '#8b1a2f'}">
                        <strong>{aspect}:</strong> {text}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("Adjust the sliders on the left and click **Analyse This Wine** to get a prediction.")

        # Engineered Features
        st.subheader("Engineered Features")
        feat_labels = {
            "alcohol_density_ratio": "Alcohol / Density",
            "flavor_intensity": "Flavour Intensity",
            "acidity_quality": "Acidity Quality",
            "sugar_acid_balance": "Sugar / Acid Balance",
            "so2_efficiency": "SO₂ Efficiency",
        }
        cols = st.columns(2)
        for i, (k, v) in enumerate(feats.items()):
            with cols[i % 2]:
                st.metric(feat_labels[k], f"{v:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)

# TAB 2 — METHODOLOGY (Humanized & Professional)
with tab_method:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="position:relative; height:140px; border-radius:12px; overflow:hidden; margin-bottom:1.5rem;">
        <img src="{IMG['cellar']}" style="width:100%;height:100%;object-fit:cover;opacity:0.75;">
        <div style="position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,4,6,0.75),rgba(15,4,6,0.2)); display:flex;align-items:center;padding:0 2rem;">
            <div style="color:white;">
                <h2 style="margin:0; font-size:1.8rem;">The Science Behind the Prediction</h2>
                <p style="margin:0.5rem 0 0; opacity:0.9; max-width:600px;">Five carefully chosen chemical relationships that winemakers have used for generations to judge quality.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); gap:1.2rem;">
    """, unsafe_allow_html=True)

    meth_cards = [
        ("Alcohol / Density", "alcohol ÷ density", "This ratio reflects how much extract and body the wine has. Higher alcohol relative to density usually means a fuller, more structured wine.", "Higher values = richer mouthfeel"),
        ("Flavour Intensity", "sulphates × alcohol", "Sulphates help preserve delicate aromas while alcohol extracts them from the grape skins. Their combination tells us about aromatic potential.", "Higher = more aromatic complexity"),
        ("Acidity Quality", "pH × volatile acidity", "Volatile acidity can turn into vinegar if too high. Multiplying by pH helps flag wines at risk of spoilage or sharpness.", "Lower = cleaner, fresher taste"),
        ("Sugar / Acid Balance", "residual sugar ÷ fixed acidity", "The classic balance between sweetness and structure. Quality dry reds usually sit in a tight, harmonious range.", "Lower = more structured & dry"),
        ("SO₂ Efficiency", "free SO₂ ÷ alcohol", "How effectively the wine is protected against oxidation and bacteria. Too little risks spoilage; too much can leave unpleasant notes.", "Ideal range: 1.5 – 3.5")
    ]

    for title, formula, desc, direction in meth_cards:
        st.markdown(f"""
        <div style="background:white; border:1px solid #e8ddd5; border-radius:10px; padding:1.2rem;">
            <div style="font-family:monospace; background:#fdf0f2; color:#8b1a2f; padding:0.2rem 0.5rem; border-radius:4px; display:inline-block; font-size:0.8rem;">{formula}</div>
            <h3 style="margin:0.8rem 0 0.4rem; color:#2c2118;">{title}</h3>
            <p style="color:#5c4a3a; line-height:1.55;">{desc}</p>
            <small style="color:#8b1a2f; font-weight:500;">{direction}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.info("This model was trained on 1,599 real Portuguese red wines using scientifically meaningful feature engineering.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#9b8c84; font-size:0.8rem; border-top:1px solid #e8ddd5; margin-top:2rem;">
    Wine Quality Predictor — Kavinda Pushpa Kumara • Built with care for wine lovers and winemakers
</div>
""", unsafe_allow_html=True)
