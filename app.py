# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT (CORRECTED)
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Initialise session state for prediction and detail toggle
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None   # stores (label, prob, scaled_vals, feats)
    if "show_detail" not in st.session_state:
        st.session_state.show_detail = False

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
        # Always engineer features from current sliders (live update)
        feats = engineer_features(alcohol, density, sulphates, pH,
                                   volatile_acidity, residual_sugar,
                                   fixed_acidity, free_so2)

        if predict_btn:
            label, prob, scaled_vals = make_prediction(feats)
            # Store prediction in session state
            st.session_state.last_prediction = (label, prob, scaled_vals, feats)
            st.session_state.show_detail = False   # reset detail view on new prediction

        # Display the last prediction if it exists
        if st.session_state.last_prediction is not None:
            label, prob, scaled_vals, feats_stored = st.session_state.last_prediction

            # ── Result card with corrected gauge ──
            if label == 1:
                gauge_color = "#8b1a2f"
                fill_width = prob * 100
                st.markdown(f"""
                <div class="result-premium">
                    <div class="result-label" style="color:#8b1a2f;">Premium</div>
                    <div class="result-prob">
                        Quality predicted ≥ 7
                    </div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence: <strong style="color:#2c2118">{prob*100:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{fill_width:.1f}%;background:{gauge_color};"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.success("Chemical profile meets premium criteria. Strong body, aromatic complexity, and controlled acidity all contribute positively.")
            else:
                gauge_color = "#b0a8a0"
                non_premium_conf = (1 - prob) * 100
                # For Non-Premium, the gauge fill shows confidence in Non-Premium (i.e., 1-prob)
                fill_width = non_premium_conf
                st.markdown(f"""
                <div class="result-standard">
                    <div class="result-label" style="color:#5c4a3a;">Non-Premium</div>
                    <div class="result-prob">
                        Quality predicted &lt; 7
                    </div>
                    <div class="gauge-wrap">
                        <div class="gauge-label">
                            <span>Non-Premium</span>
                            <span>Confidence: <strong style="color:#2c2118">{non_premium_conf:.1f}%</strong></span>
                            <span>Premium</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{fill_width:.1f}%;background:{gauge_color};"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.warning("Profile does not meet the premium threshold. Volatile acidity and flavour intensity are the primary levers for quality improvement.")

            # ── Button to toggle detailed analysis ──
            if not st.session_state.show_detail:
                if st.button("Show Detailed Analysis", key="show_detail_btn", use_container_width=True):
                    st.session_state.show_detail = True
                    st.rerun()
            else:
                if st.button("Hide Detailed Analysis", key="hide_detail_btn", use_container_width=True):
                    st.session_state.show_detail = False
                    st.rerun()

            # ── Detailed analysis (contribution chart) – only if toggled on ──
            if st.session_state.show_detail:
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
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.tick_params(colors="#5c4a3a", labelsize=8)
                ax.set_xlabel("Contribution to decision", fontsize=8, color="#9b8c84")
                pos_p = mpatches.Patch(color="#8b1a2f", label="Toward Premium")
                neg_p = mpatches.Patch(color="#b0a8a0", label="Toward Non-Premium")
                ax.legend(handles=[pos_p, neg_p], framealpha=0,
                          labelcolor="#5c4a3a", fontsize=7.5, loc="lower right")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

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
