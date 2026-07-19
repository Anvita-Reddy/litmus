import sys
import io
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from audit import feature_reliance, proxy_detection, worst_group, predictive_multiplicity

st.set_page_config(page_title="Litmus — audit your model", page_icon="◆", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
.block-container { max-width: 760px; padding-top: 3.5rem; }
h1 { font-family: 'Instrument Serif', serif !important; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Audit your own model")
st.markdown(
    "Upload a trained sklearn-compatible classifier (`.pkl` / `.joblib`) and the test CSV it "
    "should be evaluated on. Litmus runs its checks live. Nothing is stored."
)
st.caption(
    "Limits: sklearn-API models only · CSV up to 50k rows · features must match the model's "
    "training columns. Only upload files you trust — model files execute code when loaded."
)

model_file = st.file_uploader("model file", type=["pkl", "joblib"])
data_file = st.file_uploader("test data (csv)", type=["csv"])

if model_file and data_file:
    try:
        model = joblib.load(io.BytesIO(model_file.read()))
    except Exception as e:
        st.error(f"couldn't load model: {e}")
        st.stop()

    df = pd.read_csv(data_file)
    if len(df) > 50_000:
        df = df.sample(50_000, random_state=0)
        st.info("sampled 50k rows")

    target_col = st.selectbox("target column", df.columns, index=len(df.columns) - 1)
    protected_col = st.selectbox("protected attribute (optional, for proxy + subgroup checks)",
                                 ["— none —"] + [c for c in df.columns if c != target_col])
    run_mult = st.checkbox("run multiplicity check (retrains 5 quick models — slower)")

    if st.button("run audit", type="primary"):
        y = df[target_col]
        X = df.drop(columns=[target_col])

        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            st.error(f"non-numeric columns: {non_numeric}. Encode them the way the model expects first.")
            st.stop()

        try:
            base_acc = (model.predict(X) == y).mean()
        except Exception as e:
            st.error(f"model.predict failed on this data: {e}")
            st.stop()

        st.metric("accuracy on this data", f"{base_acc:.1%}")

        with st.spinner("feature reliance..."):
            rel = feature_reliance(model, X, y)
        st.subheader("what the model actually relies on")
        top = dict(list(rel.items())[:8])
        st.bar_chart(pd.Series(top))
        top_feat, top_drop = next(iter(rel.items())), None
        if top_feat[1] > 0.1:
            st.warning(
                f"`{top_feat[0]}` alone accounts for a {top_feat[1]:.1%} accuracy drop when "
                f"permuted — that's shortcut-level reliance. Verify this feature is legitimately "
                f"available at prediction time."
            )

        if protected_col != "— none —":
            with st.spinner("proxy detection..."):
                auc, proxies = proxy_detection(pd.concat([X, df[[protected_col]]], axis=1)
                                               if protected_col not in X.columns else X,
                                               protected_col)
            st.subheader(f"can `{protected_col}` be reconstructed without itself?")
            st.metric("probe AUC", f"{auc:.2f}",
                      "0.50 = unrecoverable · 1.00 = fully encoded", delta_color="off")
            if auc > 0.8:
                st.warning(f"strong proxies present: {', '.join(proxies)}. "
                           f"Removing `{protected_col}` from the features would be cosmetic.")

            groups = worst_group(model, X, y, proxies[0]) if proxies[0] in X.columns else {}
            if protected_col in X.columns:
                groups = worst_group(model, X, y, protected_col)
            if groups:
                st.subheader("accuracy by subgroup")
                st.bar_chart(pd.Series({str(k): v for k, v in groups.items()}))

        if run_mult:
            with st.spinner("multiplicity — training 5 models..."):
                r = predictive_multiplicity(X, y, n_models=5)
            frac = np.array(r["frac_pos"])
            pct = float(((frac >= 0.2) & (frac <= 0.8)).mean() * 100)
            st.subheader("predictive multiplicity")
            st.metric("contested individuals", f"{pct:.1f}%",
                      f"5 models at {r['acc_min']:.1%}–{r['acc_max']:.1%}", delta_color="off")
            if pct > 10:
                st.warning("a large share of individual predictions flip across equally-accurate "
                           "models - these decisions are seed-dependent, not data-driven.")