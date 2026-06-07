# Wine Quality Predictor — Setup Guide

## Project structure

```
wine_app/
├── app.py              ← Streamlit application
├── requirements.txt    ← Pinned dependencies
├── wine_model.pkl      ← Your trained Random Forest  ← YOU PLACE THIS HERE
└── scaler.pkl          ← Your StandardScaler          ← YOU PLACE THIS HERE
```

---

## Step 1 — Place your model files

Copy `wine_model.pkl` and `scaler.pkl` (saved with `joblib.dump()` in your notebook)
into the same folder as `app.py`.

The app expects:
- `wine_model.pkl` — a fitted `RandomForestClassifier`
- `scaler.pkl`     — a fitted `StandardScaler` that was trained on the 5 engineered features
  in this order:
  1. alcohol_density_ratio
  2. flavor_intensity
  3. acidity_quality
  4. sugar_acid_balance
  5. so2_efficiency

---

## Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Launch the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Saving your model files from the notebook

If your notebook doesn't already save the pipeline separately, add this cell:

```python
import joblib

# If you used the ImbPipeline approach, the scaler is inside the pipeline.
# Extract it separately so the Streamlit app can use it independently:
scaler_standalone = best_pipe.named_steps['scaler']
rf_standalone     = best_pipe.named_steps['clf']

joblib.dump(rf_standalone,     'wine_model.pkl')
joblib.dump(scaler_standalone, 'scaler.pkl')
print('Saved wine_model.pkl and scaler.pkl')
```

---

## Deploying to Streamlit Community Cloud (free, public URL)

1. Push the `wine_app/` folder to a public GitHub repository.
2. Go to https://share.streamlit.io → New app → select your repo.
3. Set the main file path to `app.py`.
4. Add your `.pkl` files to the repo (or use `st.secrets` + cloud storage for large files).
5. Click Deploy — you'll get a shareable URL for your portfolio.
