import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier


def compute_metrics(model, X_test, y_test, is_prob=True):
    y_pred = model.predict(X_test)

    # AUC needs probabilities (or decision function)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score) if y_score is not None else None,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "target"}, inplace=True)
    return df


def train_all_models(random_state=42):
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Pipelines where scaling helps
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state
        ),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        m = compute_metrics(model, X_test, y_test)
        results.append({
            "ML Model Name": name,
            "Accuracy": round(m["Accuracy"], 4),
            "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "MCC": round(m["MCC"], 4),
        })

    results_df = pd.DataFrame(results)
    return fitted_models, results_df, (X_test, y_test)
