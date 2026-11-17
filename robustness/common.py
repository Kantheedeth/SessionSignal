"""Shared helpers for the robustness study (same data, features and models as train.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "robustness"
SEED = 42
DROP = ["Revenue", "PageValues"]  # PageValues is revenue-derived (leakage)
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
THRESHOLD = 0.446  # best-F1 cutoff reported by train.py


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Cleaned data exactly as train.py: duplicates removed, PageValues excluded."""
    df = pd.read_csv(ROOT / "online_shoppers_intention.csv").drop_duplicates().reset_index(drop=True)
    return df.drop(columns=DROP), df["Revenue"].astype(int)


def make_pipeline(model: str = "rf", seed: int = SEED, drop: list[str] | None = None,
                  columns: list[str] | None = None) -> Pipeline:
    """Preprocess + model. `columns` = the feature columns the pipeline will see."""
    cols = columns or load_data()[0].columns.tolist()
    cols = [c for c in cols if c not in (drop or [])]
    cat = [c for c in CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        sparse_threshold=0,
    )
    models = {
        "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed),  # tuned forest from train.py
        "gb": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed),
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    }
    return Pipeline([("preprocess", prep), ("model", models[model])])


def fit_score(model: str, X_tr, y_tr, X_te, y_te, seed: int = SEED, drop: list[str] | None = None) -> dict:
    """Fit on train, return the study's standard metrics on test."""
    pipe = make_pipeline(model, seed, drop, columns=X_tr.columns.tolist())
    pipe.fit(X_tr.drop(columns=drop or []), y_tr)
    proba = pipe.predict_proba(X_te.drop(columns=drop or []))[:, 1]
    return metrics(y_te, proba)


def metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return {k: float("nan") for k in ("ap", "auc", "recall", "f1", "top10_lift", "n", "pos_rate")}
    top = max(1, int(np.ceil(len(proba) * 0.10)))
    top_idx = np.argsort(proba)[-top:]
    label = proba >= threshold
    return {
        "ap": float(average_precision_score(y_true, proba)),
        "auc": float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, label, zero_division=0)),
        "f1": float(f1_score(y_true, label, zero_division=0)),
        "top10_lift": float(y_true[top_idx].mean() / y_true.mean()),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def save(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved {path}")
    return path
