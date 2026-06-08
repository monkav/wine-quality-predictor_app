"""
Wine Quality Prediction 
Author : Kavinda Pushpa Kumara
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=32&h=32&fit=crop",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IMG = {
    "hero":   "https://images.unsplash.com/photo-1506377872008-6645d9d29ef7?w=1200&q=80&fit=crop&crop=center",
    "pour":   "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=600&q=80&fit=crop&crop=center",
    "cellar": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80&fit=crop&crop=center",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #faf7f2;
    color: #2c2118;
}
.stApp { background: #faf7f2; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/* Remove Streamlit default padding — use our own */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── App wrapper — consistent side margins ── */
.app-wrap {
    max-width: 680px;
    margin: 0 auto;
    padding: 0 1.25rem;
    box-sizing: border-box;
}

/* ── Hero ── */
.hero {
    position: relative;
    width: calc(100% + 2.5rem);
    margin-left: -1.25rem;
    height: 200px;
    overflow: hidden;
    background: #1a0a0e;
}
.hero img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center 55%;
    opacity: 0.45; display: block;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(180deg,
        rgba(12,3,5,0.3) 0%,
        rgba(12,3,5,0.75) 70%,
        rgba(12,3,5,0.92) 100%);
    display: flex; flex-direction: column;
    justify-content: flex-end;
    padding: 1.2rem 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem; font-weight: 700;
    color: #fff; line-height: 1.2; margin: 0 0 0.2rem;
    letter-spacing: 0.01em;
}
.hero-rule { width: 38px; height: 2px; background: #8b1a2f; margin: 0.3rem 0; }
.hero-author {
    font-size: 0.7rem; color: rgba(255,255,255,0.6);
    letter-spacing: 0.09em; text-transform: uppercase;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-bottom: 1px solid #e8ddd5;
    padding: 0;
    gap: 0;
    width: calc(100% + 2.5rem);
    margin-left: -1.25rem;
    padding-left: 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Lato', sans-serif;
    font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #9b8c84;
    padding: 0.8rem 1.1rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    color: #8b1a2f !important;
    border-bottom: 2px solid #8b1a2f !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Section spacing ── */
.section { margin: 1.4rem 0; }
.section-sm { margin: 1rem 0; }

/* ── Section title ── */
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; font-weight: 600;
    color: #2c2118; margin: 0 0 0.15rem;
}
.sec-sub {
    font-size: 0.71rem; color: #9b8c84;
    margin-bottom: 0.65rem; line-height: 1.45;
}

/* ── Input card ── */
.input-card {
    background: #fff;
    border: 1px solid #e8ddd5;
    border-radius: 14px;
    padding: 1.1rem 1.25rem 0.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.input-card-header {
    display: flex; align-items: center; gap: 0.8rem;
    padding-bottom: 0.7rem; margin-bottom: 0.55rem;
    border-bottom: 1px solid #f0ebe4;
}
.input-card-header img {
    width: 42px; height: 42px;
    border-radius: 8px; object-fit: cover; flex-shrink: 0;
}
.icard-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.93rem; font-weight: 600; color: #2c2118;
}
.icard-sub {
    font-size: 0.66rem; color: #9b8c84;
    letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.06rem;
}

/* ── Sliders ── */
.stSlider > label {
    font-size: 0.75rem !important;
    color: #5c4a3a !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #8b1a2f !important;
}
/* Larger touch target for mobile slider thumb */
[data-testid="stSlider"] input[type="range"] {
    height: 28px !important;
}

/* ── Primary button ── */
.stButton > button {
    background: #8b1a2f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.7rem 0 !important;
    width: 100%;
    min-height: 48px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #a02038 !important; }

/* ── Explain button (outline) ── */
.explain-btn > button {
    background: #fff !important;
    color: #8b1a2f !important;
    border: 1.5px solid #8b1a2f !important;
    margin-top: 0.5rem;
}
.explain-btn > button:hover { background: #fdf0f2 !important; }

/* ── Result card ── */
.result-premium {
    background: linear-gradient(135deg, #fff5f6 0%, #fff 100%);
    border: 2px solid #8b1a2f; border-radius: 14px;
    padding: 1.3rem 1.25rem 1.1rem;
    text-align: center; position: relative; overflow: hidden;
    margin-bottom: 0.75rem;
}
.result-premium::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #8b1a2f, #c9406a, #8b1a2f);
}
.result-standard {
    background: #fff; border: 2px solid #d0c8c0;
    border-radius: 14px; padding: 1.3rem 1.25rem 1.1rem;
    text-align: center; margin-bottom: 0.75rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700;
    letter-spacing: 0.03em; margin-bottom: 0.2rem; line-height: 1.1;
}
.result-sub { font-size: 0.78rem; color: #9b8c84; }
.gauge-wrap { margin: 0.7rem 0 0.2rem; }
.gauge-label {
    display: flex; justify-content: space-between;
    font-size: 0.62rem; color: #9b8c84;
    letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.gauge-track {
    background: #f0ebe4; border-radius: 99px; height: 8px; overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 99px; }

/* ── Feature chips ── */
.feat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.45rem; margin-top: 0.5rem;
}
.feat-chip {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 9px; padding: 0.42rem 0.65rem;
}
.feat-chip .fname {
    color: #9b8c84; font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.04em; display: block; margin-bottom: 0.04rem;
}
.feat-chip .fval {
    color: #2c2118; font-weight: 600;
    font-family: monospace; font-size: 0.8rem;
}

/* ── Explanation panel ── */
.expl-panel {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 12px; padding: 1rem 1.15rem;
    margin-top: 0.6rem;
}
.note-row {
    border-left: 3px solid #ccc; border-radius: 0 7px 7px 0;
    padding: 0.38rem 0.7rem; margin-bottom: 0.32rem;
    font-size: 0.75rem; line-height: 1.45;
}

/* ── Methodology cards ── */
.meth-hero {
    position: relative; width: 100%; height: 110px;
    border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}
.meth-hero img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center 40%; opacity: 0.55;
}
.meth-hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(12,3,5,0.85) 0%, rgba(12,3,5,0.15) 100%);
    display: flex; align-items: center; padding: 0 1.4rem;
}
.meth-hero-text h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; font-weight: 700; color: #fff; margin: 0 0 0.1rem;
}
.meth-hero-text p {
    font-size: 0.69rem; color: rgba(255,255,255,0.75);
    margin: 0; line-height: 1.4; max-width: 320px;
}

/* ── Method card — stacked single column on mobile ── */
.meth-card {
    background: #fff; border: 1px solid #e8ddd5;
    border-radius: 11px; padding: 0.9rem 1.05rem;
    border-top: 3px solid #8b1a2f; margin-bottom: 0.65rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.mc-tag {
    font-family: monospace; font-size: 0.67rem;
    background: #fdf0f2; color: #8b1a2f;
    padding: 0.1rem 0.4rem; border-radius: 4px;
    display: inline-block; margin-bottom: 0.3rem;
}
.mc-formula { font-size: 0.71rem; color: #9b8c84; font-family: monospace; margin-bottom: 0.3rem; }
.mc-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem; font-weight: 600;
    color: #2c2118; margin-bottom: 0.22rem;
}
.mc-body { font-size: 0.74rem; color: #5c4a3a; line-height: 1.55; }
.mc-meta { font-size: 0.63rem; color: #b0a8a0; margin-top: 0.2rem; }
.dir-good  { color: #4a7c59; font-size: 0.68rem; font-weight: 500; margin-top: 0.25rem; }
.dir-bad   { color: #8b1a2f; font-size: 0.68rem; font-weight: 500; margin-top: 0.25rem; }
.dir-range { color: #7a6a30; font-size: 0.68rem; font-weight: 500; margin-top: 0.25rem; }

/* ── Footer ── */
.app-footer {
    text-align: center; padding: 1.2rem 1.25rem 2rem;
    font-size: 0.68rem; color: #c8bdb8; letter-spacing: 0.05em;
    border-top: 1px solid #e8ddd5; margin-top: 0.5rem;
}

/* ── Awaiting placeholder ── */
.awaiting {
    position: relative; border-radius: 12px; overflow: hidden;
    height: 130px; margin-bottom: 0.75rem;
}
.awaiting img { width:100%; height:100%; object-fit:cover; opacity:0.25; }
.awaiting-text {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 0.28rem;
    background: rgba(250,247,242,0.5);
}
.awaiting-text .at-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.92rem; color: #5c4a3a; font-weight: 600;
}
.awaiting-text .at-sub { font-size: 0.69rem; color: #9b8c84; }

/* ── Wider screens: side-by-side inputs ── */
@media (min-width: 600px) {
    .hero { height: 220px; }
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("wine_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

model, scaler = load_model()

FEATURE_NAMES = [
    "alcohol_density_ratio",
    "flavor_intensity",
    "acidity_quality",
    "so2_efficiency",
    "sugar_acid_balance",
]
IMPORTANCES = np.array([0.300665, 0.285175, 0.177338, 0.125637, 0.111185])

MEANS = {
    "alcohol":             10.42,
    "density":              0.9967,
    "sulphates":            0.658,
    "pH":                   3.311,
    "volatile_acidity":     0.528,
    "residual_sugar":       2.539,
    "fixed_acidity":        8.320,
    "free_sulfur_dioxide": 15.87,
}


def engineer(alcohol, density, sulphates, pH, va, rs, fa, fso2):
    return {
        "alcohol_density_ratio": alcohol / density,
        "flavor_intensity":      sulphates * alcohol,
        "acidity_quality":       pH * va,
        "so2_efficiency":        fso2 / (alcohol + 1e-6),
        "sugar_acid_balance":    rs   / (fa    + 1e-6),
    }


def predict(feats):
    df     = pd.DataFrame([feats])[FEATURE_NAMES]
    scaled = scaler.transform(df)
    prob   = model.predict_proba(scaled)[0, 1]
    return (1 if prob >= 0.5 else 0), prob, scaled[0]


def science_notes(feats):
    notes = []
    adr, fi, aq  = feats["alcohol_density_ratio"], feats["flavor_intensity"], feats["acidity_quality"]
    so2, sab     = feats["so2_efficiency"],         feats["sugar_acid_balance"]

    notes.append(("Body",         (f"Full-bodied ({adr:.2f}) — strong extract and palate weight.",      "good")
                                   if adr > 12.5 else
                                  (f"Light-bodied ({adr:.2f}) — may lack structure.",                   "warn")
                                   if adr < 11.0 else
                                  (f"Medium body ({adr:.2f}) — typical range for dry reds.",            "neutral")))

    notes.append(("Aroma",        (f"High complexity ({fi:.2f}) — good aromatic richness.",              "good")
                                   if fi > 9.0 else
                                  (f"Low intensity ({fi:.2f}) — may lack aromatic complexity.",          "warn")
                                   if fi < 6.0 else
                                  (f"Moderate intensity ({fi:.2f}) — adequate presence.",                "neutral")))

    notes.append(("Acidity",      (f"Clean ({aq:.2f}) — no vinegar off-flavours expected.",              "good")
                                   if aq < 2.0 else
                                  (f"High volatile acidity ({aq:.2f}) — vinegar notes may be detectable.","warn")
                                   if aq > 3.0 else
                                  (f"Borderline ({aq:.2f}) — monitor volatile acidity.",                 "neutral")))

    notes.append(("Preservation", (f"Optimal SO2 efficiency ({so2:.2f}) — good protection.",             "good")
                                   if 1.5 <= so2 <= 3.5 else
                                  (f"Low SO2 efficiency ({so2:.2f}) — oxidation risk.",                  "warn")
                                   if so2 < 1.5 else
                                  (f"High SO2 ({so2:.2f}) — possible sulfurous off-aromas.",             "warn")))

    notes.append(("Balance",      (f"Well-balanced ({sab:.3f}) — low residual sugar; typical of quality dry reds.", "good")
                                   if sab < 0.3 else
                                  (f"Excess sweetness ({sab:.3f}) — high residual sugar vs acid backbone.",          "warn")
                                   if sab > 0.8 else
                                  (f"Moderate balance ({sab:.3f}) — within expected range.",                          "neutral")))
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <img src="{IMG['hero']}" alt="Vineyard">
  <div class="hero-overlay">
    <div class="hero-title">Wine Quality Predictor</div>
    <div class="hero-rule"></div>
    <div class="hero-author">
      Kavinda Pushpa Kumara &nbsp;&middot;&nbsp;
      Food Science &amp; Data Science
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_method = st.tabs(["  Predict  ", "  Methodology  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)

    # ── Input card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section">
      <div class="input-card">
        <div class="input-card-header">
          <img src="{IMG['pour']}" alt="Wine">
          <div>
            <div class="icard-title">Chemical Measurements</div>
            <div class="icard-sub">Enter raw lab values</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sliders — two columns on wider screens, natural stack on narrow
    c1, c2 = st.columns(2)
    with c1:
        alcohol  = st.slider("Alcohol (%vol)",         8.4,  14.9, float(round(MEANS["alcohol"],  1)), 0.1)
        sulphates= st.slider("Sulphates (g/dm³)",      0.33,  2.0, float(round(MEANS["sulphates"],2)), 0.01)
        pH       = st.slider("pH",                     2.74,  4.01,float(round(MEANS["pH"],       2)), 0.01)
        rs       = st.slider("Residual Sugar (g/dm³)", 0.9,  15.5, float(round(MEANS["residual_sugar"],1)), 0.1)
    with c2:
        density  = st.slider("Density (g/cm³)",        0.9901,1.0037,float(round(MEANS["density"],4)), 0.0001, format="%.4f")
        va       = st.slider("Volatile Acidity (g/dm³)",0.12, 1.58, float(round(MEANS["volatile_acidity"],2)), 0.01)
        fa       = st.slider("Fixed Acidity (g/dm³)",  4.6,  15.9, float(round(MEANS["fixed_acidity"],1)), 0.1)
        fso2     = st.slider("Free SO₂ (mg/dm³)",      1.0,  72.0, float(round(MEANS["free_sulfur_dioxide"],0)), 0.5)

    predict_btn = st.button("Analyse Wine", use_container_width=True)

    # always compute features live for the chips
    feats = engineer(alcohol, density, sulphates, pH, va, rs, fa, fso2)

    if predict_btn:
        label, prob, scaled = predict(feats)
        st.session_state.update({
            "label": label, "prob": prob,
            "scaled": scaled, "feats": dict(feats),
            "show_expl": False,
        })

    # ── Result ───────────────────────────────────────────────────────────────
    if "prob" in st.session_state:
        label  = st.session_state["label"]
        prob   = st.session_state["prob"]
        scaled = st.session_state["scaled"]
        sf     = st.session_state["feats"]

        if label == 1:
            st.markdown(f"""
            <div class="result-premium">
              <div class="result-label" style="color:#8b1a2f;">Premium</div>
              <div class="result-sub">
                Quality predicted &ge; 7 &nbsp;&middot;&nbsp;
                Confidence <strong style="color:#2c2118">{prob*100:.1f}%</strong>
              </div>
              <div class="gauge-wrap">
                <div class="gauge-label"><span>Non-Premium</span><span>Premium</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{prob*100:.1f}%;background:#8b1a2f;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.success("Strong body, aromatic complexity and controlled acidity all contribute positively.")
        else:
            st.markdown(f"""
            <div class="result-standard">
              <div class="result-label" style="color:#5c4a3a;">Non-Premium</div>
              <div class="result-sub">
                Quality predicted &lt; 7 &nbsp;&middot;&nbsp;
                Confidence <strong style="color:#2c2118">{(1-prob)*100:.1f}%</strong>
              </div>
              <div class="gauge-wrap">
                <div class="gauge-label"><span>Non-Premium</span><span>Premium</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{prob*100:.1f}%;background:#b0a8a0;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.warning("Volatile acidity and flavour intensity are the primary levers for quality improvement.")

        # Explain button — only appears after a prediction
        st.markdown('<div class="explain-btn">', unsafe_allow_html=True)
        if st.button("Why did the model decide this?", use_container_width=True):
            st.session_state["show_expl"] = not st.session_state.get("show_expl", False)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("show_expl", False):
            st.markdown('<div class="expl-panel">', unsafe_allow_html=True)

            # Contribution chart
            st.markdown('<p class="sec-title">Feature Contributions</p>', unsafe_allow_html=True)
            st.markdown('<p class="sec-sub">How much each feature pushed the decision — burgundy toward Premium, grey toward Non-Premium.</p>', unsafe_allow_html=True)

            contribs = IMPORTANCES * scaled
            cdf = (pd.DataFrame({"Feature": FEATURE_NAMES, "v": contribs})
                     .sort_values("v", key=abs, ascending=True))
            colors = ["#8b1a2f" if v >= 0 else "#b0a8a0" for v in cdf["v"]]

            fig, ax = plt.subplots(figsize=(4.5, 2.2))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            ax.barh(cdf["Feature"], cdf["v"], color=colors, height=0.46, edgecolor="none")
            ax.axvline(0, color="#e8ddd5", linewidth=1)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors="#5c4a3a", labelsize=7.5)
            ax.set_xlabel("Contribution", fontsize=7.5, color="#9b8c84")
            pos_p = mpatches.Patch(color="#8b1a2f", label="Toward Premium")
            neg_p = mpatches.Patch(color="#b0a8a0", label="Toward Non-Premium")
            ax.legend(handles=[pos_p, neg_p], framealpha=0,
                      labelcolor="#5c4a3a", fontsize=7, loc="lower right")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Food science notes
            st.markdown('<p class="sec-title" style="margin-top:0.7rem;">Food Science Analysis</p>', unsafe_allow_html=True)
            bg   = {"good":"#eef6f1","warn":"#fdf0f2","neutral":"#faf7f2"}
            bdr  = {"good":"#4a7c59","warn":"#8b1a2f","neutral":"#d0c8c0"}
            txt  = {"good":"#2d5a3d","warn":"#8b1a2f","neutral":"#5c4a3a"}
            for aspect, (text, tone) in science_notes(sf):
                st.markdown(
                    f'<div class="note-row" style="background:{bg[tone]};'
                    f'border-left-color:{bdr[tone]};color:{txt[tone]};">'
                    f'<strong>{aspect}:</strong> {text}</div>',
                    unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="awaiting">
          <img src="{IMG['pour']}" alt="Wine">
          <div class="awaiting-text">
            <div class="at-title">Awaiting Analysis</div>
            <div class="at-sub">Press <strong style="color:#8b1a2f;">Analyse Wine</strong> above</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Engineered feature chips (always visible) ────────────────────────────
    st.markdown('<div class="section-sm">', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Engineered Features</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Calculated live from your inputs. These 5 values are what the model actually receives.</p>', unsafe_allow_html=True)

    feat_labels = {
        "alcohol_density_ratio": "Alcohol / Density",
        "flavor_intensity":      "Flavour Intensity",
        "acidity_quality":       "Acidity Quality",
        "so2_efficiency":        "SO2 Efficiency",
        "sugar_acid_balance":    "Sugar / Acid Balance",
    }
    chips = '<div class="feat-grid">'
    for k in FEATURE_NAMES:
        chips += (f'<div class="feat-chip">'
                  f'<span class="fname">{feat_labels[k]}</span>'
                  f'<span class="fval">{feats[k]:.4f}</span>'
                  f'</div>')
    chips += "</div>"
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close app-wrap


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section">
      <div class="meth-hero">
        <img src="{IMG['cellar']}" alt="Wine cellar">
        <div class="meth-hero-overlay">
          <div class="meth-hero-text">
            <h2>Scientific Methodology</h2>
            <p>Five chemically meaningful ratios — each encoding a winemaking principle.</p>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    meth_cards = [
        {
            "tag":     "alcohol_density_ratio",
            "formula": "alcohol ÷ density",
            "title":   "Body &amp; Mouthfeel",
            "body":    "Higher alcohol relative to density signals greater extract and palate weight — a key premium red wine attribute.",
            "meta":    "Importance: 0.301 · Correlation: +0.467",
            "dir":     "Higher = fuller body", "dir_class": "dir-good",
        },
        {
            "tag":     "flavor_intensity",
            "formula": "sulphates × alcohol",
            "title":   "Aroma Complexity",
            "body":    "Sulphates protect volatile aromatic compounds; alcohol extracts them from grape skins. Their product captures both preservation and extraction power.",
            "meta":    "Importance: 0.285 · Correlation: +0.391",
            "dir":     "Higher = more aromatic richness", "dir_class": "dir-good",
        },
        {
            "tag":     "acidity_quality",
            "formula": "pH × volatile acidity",
            "title":   "Fault Detection",
            "body":    "Volatile acidity above ~0.6 g/dm³ is detectable as vinegar. Multiplying by pH amplifies the penalty when both are elevated, flagging poor microbial stability.",
            "meta":    "Importance: 0.177 · Correlation: −0.373",
            "dir":     "Lower = cleaner, better balanced", "dir_class": "dir-bad",
        },
        {
            "tag":     "so2_efficiency",
            "formula": "free SO₂ ÷ alcohol",
            "title":   "Preservation Efficiency",
            "body":    "Normalising free SO₂ by alcohol gives a preservation efficiency score. Too low: oxidation risk. Too high: sulfurous off-aromas detectable above ~50 mg/L.",
            "meta":    "Importance: 0.126 · Correlation: −0.132",
            "dir":     "Optimal: 1.5 – 3.5", "dir_class": "dir-range",
        },
        {
            "tag":     "sugar_acid_balance",
            "formula": "residual sugar ÷ fixed acidity",
            "title":   "Sweetness Perception",
            "body":    "Residual sugar interacts with fixed acidity to shape perceived roundness. A high ratio indicates excess sweetness — atypical for quality dry reds. Retained for interaction effects.",
            "meta":    "Importance: 0.111 · Correlation: −0.014",
            "dir":     "Lower = drier, more structured", "dir_class": "dir-bad",
        },
    ]

    cards_html = ""
    for c in meth_cards:
        cards_html += f"""
        <div class="meth-card">
          <span class="mc-tag">{c['tag']}</span>
          <div class="mc-formula">{c['formula']}</div>
          <div class="mc-title">{c['title']}</div>
          <div class="mc-body">{c['body']}</div>
          <div class="mc-meta">{c['meta']}</div>
          <div class="{c['dir_class']}">{c['dir']}</div>
        </div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.67rem;color:#b0a8a0;line-height:1.6;margin:0.5rem 0 1rem;">
      Cortez et al. (2009). <em>Modeling wine preferences by data mining.</em> Decision Support Systems, 47(4). &middot;
      Peynaud, E. (1987). <em>Knowing and Making Wine.</em> Wiley.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close app-wrap


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Kavinda Pushpa Kumara
</div>
Keep working
