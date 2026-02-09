import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from model.train_models import train_all_models

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Classification Models Demo")

st.write("Dataset used: Breast Cancer Wisconsin (Diagnostic) (via scikit-learn).")

# Train once per session
@st.cache_data
def train_cached():
    models, metrics_df, test_pack = train_all_models()
    return models, metrics_df, test_pack

models, metrics_df, (X_test_default, y_test_default) = train_cached()

st.subheader("Model Comparison (Test Split Metrics)")
st.dataframe(metrics_df, use_container_width=True)

model_name = st.selectbox("Select a model", list(models.keys()))
model = models[model_name]

st.markdown("---")
st.subheader(f"Selected Model: {model_name}")

uploaded = st.file_uploader("Upload a CSV (must contain SAME feature columns; optional)", type=["csv"])

if uploaded:
    user_df = pd.read_csv(uploaded)
    if "target" in user_df.columns:
        X_user = user_df.drop(columns=["target"])
        y_user = user_df["target"]
        X_eval, y_eval = X_user, y_user
    else:
        X_eval, y_eval = user_df, None

    st.write("Using uploaded data for predictions.")
else:
    X_eval, y_eval = X_test_default, y_test_default
    st.write("No upload provided — using internal test split.")

y_pred = model.predict(X_eval)

col1, col2 = st.columns(2)

with col1:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred) if y_eval is not None else None
    if cm is not None:
        st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
    else:
        st.info("No target column found in upload, so confusion matrix is unavailable.")

with col2:
    st.write("### Classification Report")
    if y_eval is not None:
        report = classification_report(y_eval, y_pred, output_dict=False)
        st.text(report)
    else:
        st.info("No target column found in upload, so classification report is unavailable.")

st.write("### AUC (if targets available)")
if y_eval is not None and hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_eval)[:, 1]
    st.write(round(roc_auc_score(y_eval, y_prob), 4))
elif y_eval is not None:
    st.info("This model does not provide predict_proba; AUC may not be available.")
else:
    st.info("AUC requires target labels in the uploaded CSV.")
