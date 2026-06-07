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

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark refined wine-lab aesthetic ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0b0f;
    color: #e8ddd5;
}

/* ── Main container background ── */
.stApp {
    background: linear-gradient(160deg, #0e0b0f 0%, #150d11 60%, #0e0b0f 100%);
}

/* ── Header block ── */
.hero {
    border-left: 3px solid #8b1a2f;
    padding: 1.4rem 2rem;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, rgba(139,26,47,0.08) 0%, transparent 100%);
    border-radius: 0 8px 8px 0;
}
.hero h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #e8ddd5;
    margin: 0 0 0.25rem 0;
    letter-spacing: 0.02em;
}
.hero .sub {
    font-size: 0.85rem;
    color: #9b8c84;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero .author {
    font-size: 0.95rem;
    color: #c9a87a;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #c9a87a;
    border-bottom: 1px solid #2e1a20;
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem 0;
    letter-spacing: 0.03em;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid #2e1a20;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9b8c84;
    margin-bottom: 0.2rem;
}
.metric-card .value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8ddd5;
}

/* ── Prediction box ── */
.pred-premium {
    background: linear-gradient(135deg, rgba(139,26,47,0.2) 0%, rgba(139,26,47,0.05) 100%);
    border: 1px solid #8b1a2f;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.pred-standard {
    background: linear-gradient(135deg, rgba(60,60,80,0.25) 0%, rgba(60,60,80,0.05) 100%);
    border: 1px solid #3c3c50;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.pred-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.pred-prob {
    font-size: 1rem;
    color: #9b8c84;
    margin-top: 0.4rem;
}

/* ── Feature science table ── */
.feature-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.feature-table th {
    background: rgba(139,26,47,0.15);
    color: #c9a87a;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 500;
    border-bottom: 1px solid #2e1a20;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.75rem;
}
.feature-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #1e1218;
    color: #c8bdb8;
    vertical-align: top;
    line-height: 1.5;
}
.feature-table tr:hover td {
    background: rgba(255,255,255,0.02);
}
.code-tag {
    font-family: monospace;
    background: rgba(139,26,47,0.18);
    color: #e8a0a8;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.82rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #100d12 !important;
    border-right: 1px solid #2e1a20;
}
[data-testid="stSidebar"] .stSlider > label {
    color: #c9a87a !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Buttons ── */
.stButton > button {
    background: #8b1a2f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.8rem !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background: #a02038 !important;
}

/* ── Caution badge ── */
.caution-tag {
    display: inline-block;
    background: rgba(201,168,122,0.15);
    border: 1px solid rgba(201,168,122,0.35);
    color: #c9a87a;
    font-size: 0.72rem;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* ── Matplotlib figure background ── */
.stImage { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(
            f"⚠️ Model file not found: {e}\n\n"
            "Place `wine_model.pkl` and `scaler.pkl` in the same folder as `app.py` and restart."
        )
        st.stop()

model, scaler = load_model()

# Pull feature importances from the loaded RF model
FEATURE_NAMES = [
    "alcohol_density_ratio",
    "flavor_intensity",
    "acidity_quality",
    "sugar_acid_balance",
    "so2_efficiency",
]
importances = model.feature_importances_


# ── Dataset means (from UCI Red Wine dataset, used as slider defaults) ───────
DEFAULTS = {
    "alcohol_density_ratio": 10.4 / 0.9967,   # ≈ 10.43
    "flavor_intensity":      0.658 * 10.4,     # ≈ 6.84
    "acidity_quality":       3.311 * 0.528,    # ≈ 1.75
    "sugar_acid_balance":    2.539 / 8.32,     # ≈ 0.305
    "so2_efficiency":        15.87 / 10.4,     # ≈ 1.53
}

SLIDER_CONFIG = {
    "alcohol_density_ratio": {
        "label": "Alcohol / Density Ratio",
        "min": 8.0, "max": 15.0, "step": 0.01,
        "help": "Body & mouthfeel. Higher → fuller-bodied wine.",
    },
    "flavor_intensity": {
        "label": "Flavor Intensity  (sulphates × alcohol)",
        "min": 2.0, "max": 20.0, "step": 0.05,
        "help": "Aroma richness. Sulphates preserve volatile compounds extracted by alcohol.",
    },
    "acidity_quality": {
        "label": "Acidity Quality  (pH × volatile acidity)",
        "min": 0.5, "max": 6.0, "step": 0.01,
        "help": "Lower is better. High volatile acidity → vinegar off-flavours.",
    },
    "sugar_acid_balance": {
        "label": "Sugar / Acid Balance",
        "min": 0.05, "max": 2.0, "step": 0.005,
        "help": "Sensory sweetness perception relative to fixed acidity.",
    },
    "so2_efficiency": {
        "label": "SO₂ Efficiency  (free SO₂ / alcohol)",
        "min": 0.1, "max": 8.0, "step": 0.05,
        "help": "Preservation efficiency. Optimal: 1.5–3.5. Outside → oxidation or over-sulfiting.",
    },
}


# ── SIDEBAR — inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Cormorant Garamond,serif; font-size:1.3rem;"
        "color:#c9a87a; border-bottom:1px solid #2e1a20; padding-bottom:0.4rem;"
        "margin-bottom:1.2rem;'>🍷 Wine Parameters</div>",
        unsafe_allow_html=True,
    )
    st.caption("Adjust the five engineered chemical features. Defaults are dataset means.")

    user_inputs = {}
    for feat, cfg in SLIDER_CONFIG.items():
        user_inputs[feat] = st.slider(
            label=cfg["label"],
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=float(round(DEFAULTS[feat], 3)),
            step=cfg["step"],
            help=cfg["help"],
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Wine", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#4a3a3f; text-align:center;'>"
        "Model: Random Forest · UCI Red Wine · n=1599</div>",
        unsafe_allow_html=True,
    )


# ── MAIN BODY ────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <div class="sub">Predictive Quality Analysis System</div>
    <h1>🍷 Wine Quality Predictor</h1>
    <div class="author">
        Kavinda Pushpa Kumara &nbsp;·&nbsp; Food Science Student &nbsp;·&nbsp; IBM Certified Data Scientist
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout: prediction (left) | engineered values (right) ───────────────────
col_pred, col_vals = st.columns([3, 2], gap="large")

with col_pred:
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build input dataframe → scale → predict
        input_df = pd.DataFrame([user_inputs])
        input_scaled = scaler.transform(input_df)

        pred_label = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0]
        premium_prob = pred_proba[1]
        standard_prob = pred_proba[0]

        if pred_label == 1:
            st.markdown(f"""
            <div class="pred-premium">
                <div class="pred-label" style="color:#e8a0a8;">✦ Premium</div>
                <div class="pred-prob">
                    Confidence: <strong style="color:#e8ddd5;">{premium_prob*100:.1f}%</strong>
                    &nbsp;·&nbsp; Quality score predicted ≥ 7
                </div>
            </div>""", unsafe_allow_html=True)
            st.success(
                "This wine's chemical profile meets the criteria for a **Premium** classification. "
                "High alcohol-density body, strong flavour compounds, and controlled acidity all "
                "contribute positively."
            )
        else:
            st.markdown(f"""
            <div class="pred-standard">
                <div class="pred-label" style="color:#8e8aa0;">◇ Non-Premium</div>
                <div class="pred-prob">
                    Confidence: <strong style="color:#e8ddd5;">{standard_prob*100:.1f}%</strong>
                    &nbsp;·&nbsp; Quality score predicted &lt; 7
                </div>
            </div>""", unsafe_allow_html=True)
            st.warning(
                "This wine's profile does not meet the **Premium** threshold. "
                "Review acidity quality and flavour intensity — these are the strongest levers "
                "for quality improvement."
            )

        # ── Per-prediction feature contribution chart ──────────────────────
        st.markdown('<div class="section-title">Why did the model decide this?</div>', unsafe_allow_html=True)
        st.caption(
            "Each bar shows how much each feature contributed to the model's decision, "
            "weighted by its importance in the Random Forest. "
            "A feature's contribution = its importance × its scaled value. "
            "Positive (burgundy) contributions push toward Premium; "
            "negative (slate) push toward Non-Premium."
        )

        # Scaled input values × importances = contribution proxy
        scaled_vals = input_scaled[0]
        contributions = importances * scaled_vals
        contrib_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Contribution": contributions,
        }).sort_values("Contribution", key=abs, ascending=True)

        colors = ["#8b1a2f" if v >= 0 else "#4a5568" for v in contrib_df["Contribution"]]

        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor("#0e0b0f")
        ax.set_facecolor("#0e0b0f")

        bars = ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                       color=colors, height=0.55, edgecolor="none")
        ax.axvline(0, color="#3c2a2f", linewidth=1)

        ax.set_xlabel("Contribution to decision", color="#9b8c84", fontsize=9)
        ax.tick_params(colors="#c8bdb8", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.label.set_color("#9b8c84")
        ax.tick_params(axis="x", colors="#9b8c84")
        ax.tick_params(axis="y", colors="#c8bdb8")

        pos_patch = mpatches.Patch(color="#8b1a2f", label="Pushes → Premium")
        neg_patch = mpatches.Patch(color="#4a5568", label="Pushes → Non-Premium")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
                  framealpha=0, labelcolor="#9b8c84", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        st.markdown(
            "<div style='color:#4a3a3f; padding:2.5rem; text-align:center; "
            "border:1px dashed #2e1a20; border-radius:10px; font-size:0.9rem;'>"
            "Adjust parameters in the sidebar<br>then press <strong>Analyse Wine</strong>"
            "</div>",
            unsafe_allow_html=True,
        )


with col_vals:
    st.markdown('<div class="section-title">Engineered Values</div>', unsafe_allow_html=True)
    st.caption("Current slider values mapped to the 5 model features.")

    labels = {
        "alcohol_density_ratio": "Alcohol / Density",
        "flavor_intensity":      "Flavour Intensity",
        "acidity_quality":       "Acidity Quality",
        "sugar_acid_balance":    "Sugar / Acid Balance",
        "so2_efficiency":        "SO₂ Efficiency",
    }
    for feat, val in user_inputs.items():
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{labels[feat]}</div>
            <div class="value">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)


# ── Global feature importance chart ─────────────────────────────────────────
st.markdown('<div class="section-title">Model Feature Importance</div>', unsafe_allow_html=True)
st.caption(
    "These are the overall importances learned by the Random Forest across all 1,599 wines. "
    "A feature's importance measures the average reduction in impurity (Gini) it provides across all decision trees. "
    "Higher = the model relies on it more when classifying wines globally."
)

imp_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(9, 3.2))
fig2.patch.set_facecolor("#0e0b0f")
ax2.set_facecolor("#0e0b0f")

bar_colors = [
    "#8b1a2f" if i == len(imp_df) - 1
    else "#5e1420" if i == len(imp_df) - 2
    else "#3a2838"
    for i in range(len(imp_df))
]

ax2.barh(imp_df["Feature"], imp_df["Importance"], color=bar_colors, height=0.55, edgecolor="none")

for i, (_, row) in enumerate(imp_df.iterrows()):
    ax2.text(row["Importance"] + 0.003, i, f'{row["Importance"]:.3f}',
             va="center", color="#9b8c84", fontsize=9)

ax2.set_xlabel("Mean Decrease in Gini Impurity", color="#9b8c84", fontsize=9)
ax2.tick_params(colors="#c8bdb8", labelsize=9)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.xaxis.label.set_color("#9b8c84")
ax2.tick_params(axis="x", colors="#9b8c84")
ax2.tick_params(axis="y", colors="#c8bdb8")
ax2.set_xlim(0, imp_df["Importance"].max() + 0.06)

plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()


# ── Scientific Methodology ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Scientific Methodology</div>', unsafe_allow_html=True)
st.markdown(
    "This model does not feed raw laboratory measurements directly into the classifier. "
    "Instead, five **chemically meaningful ratios** are engineered from the raw data — each one "
    "encoding a winemaking principle that trained sommeliers and oenologists use to assess quality.",
    unsafe_allow_html=False,
)

st.markdown("""
<table class="feature-table">
<thead>
  <tr>
    <th>Feature</th>
    <th>Formula</th>
    <th>Winemaking Principle</th>
    <th>Direction</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><span class="code-tag">alcohol_density_ratio</span></td>
    <td>alcohol ÷ density</td>
    <td><strong>Body &amp; mouthfeel.</strong> Higher alcohol relative to density indicates greater extract and fuller palate weight — a key premium red wine attribute.</td>
    <td style="color:#8b9a6a;">↑ Higher = better</td>
  </tr>
  <tr>
    <td><span class="code-tag">flavor_intensity</span></td>
    <td>sulphates × alcohol</td>
    <td><strong>Aroma complexity.</strong> Sulphates (potassium sulphate) protect volatile aromatic compounds; alcohol acts as the solvent that extracts them from grape skins. Their product captures both preservation and extraction capacity.</td>
    <td style="color:#8b9a6a;">↑ Higher = better</td>
  </tr>
  <tr>
    <td><span class="code-tag">acidity_quality</span></td>
    <td>pH × volatile acidity</td>
    <td><strong>Acidity balance &amp; fault detection.</strong> Volatile acidity (mainly acetic acid) above ~0.6 g/dm³ is perceptible as vinegar — a classic fault. Multiplying by pH amplifies the penalty for wines that are both high in volatile acidity and have an elevated pH (poor microbial stability).</td>
    <td style="color:#9a6a6a;">↓ Lower = better</td>
  </tr>
  <tr>
    <td><span class="code-tag">sugar_acid_balance</span></td>
    <td>residual sugar ÷ fixed acidity</td>
    <td><strong>Sensory sweetness perception.</strong> Even in dry red wines, residual sugar interacts with fixed acidity (tartaric, malic acids) to shape perceived roundness. A very high ratio indicates excess residual sweetness relative to the acid backbone — atypical for quality dry reds. Note: marginal correlation with quality is weak (≈ −0.03) but retained for its established role in winemaking sensory science.</td>
    <td style="color:#9a6a6a;">↓ Lower = better (dry reds)</td>
  </tr>
  <tr>
    <td><span class="code-tag">so2_efficiency</span></td>
    <td>free SO₂ ÷ alcohol</td>
    <td><strong>Preservation efficiency.</strong> Free SO₂ prevents oxidation and microbial spoilage; however, its antimicrobial efficacy depends on pH and the wine's alcohol content. Normalising by alcohol yields a preservation efficiency score. Too low → oxidation risk; too high → sulfurous off-aromas detectable by consumers.</td>
    <td style="color:#9b8c84;">⇔ Optimal range: 1.5–3.5</td>
  </tr>
</tbody>
</table>
<br>
""", unsafe_allow_html=True)

st.caption(
    "References: Peynaud, E. (1987). Knowing and Making Wine. Wiley. "
    "· Cortez et al. (2009). Modeling wine preferences by data mining from physicochemical properties. "
    "Decision Support Systems, 47(4). "
    "· OIV (2023). International Code of Oenological Practices."
)


# ── Model transparency note ──────────────────────────────────────────────────
with st.expander("ℹ️  Model transparency & limitations"):
    st.markdown("""
**Dataset:** UCI Machine Learning Repository — Red Wine Quality (n = 1,599 Portuguese Vinho Verde wines).  
**Algorithm:** Random Forest Classifier with `class_weight='balanced'` and SMOTE oversampling applied *inside* each cross-validation fold to prevent data leakage.  
**Class imbalance:** Only ~14% of wines score ≥ 7. A majority-class baseline achieves ~86% accuracy — this model is evaluated on F1 score and AUC-ROC, not raw accuracy.  
**Threshold:** Default classification threshold is 0.50. Depending on business context, a higher threshold (e.g. 0.60) may be preferred to minimise false Premium labels.  
**Scope:** This model was trained exclusively on *red* wine data. Results for white wines or wines from other regions are unreliable.  
**Feature importances** reflect the global behaviour of the Random Forest (mean Gini impurity reduction). The per-prediction contribution chart is a proxy (importance × scaled value) — for rigorous local explanations, SHAP values should be used.
    """)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#2e1a20; font-size:0.75rem; "
    "border-top:1px solid #1e1218; padding-top:1rem;'>"
    "Wine Quality Predictor · Kavinda Pushpa Kumara · Food Science | Data Science Portfolio"
    "</div>",
    unsafe_allow_html=True,
)
