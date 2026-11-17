"""Does LightGBM beat our chosen model? Optional follow-up experiment.

PyCaret's 14-model sweep puts LightGBM first by AUC, so this is the obvious
question to ask, and other people reproducing this project have landed on it.
This script puts LightGBM through the *same* protocol as train.py — the same
cleaned data, the same PageValues exclusion, the same stratified split with
seed 42, the same four-fold cross-validation, the same average-precision
metric, and the same untouched 2,441-row holdout — so the comparison is fair.

LightGBM is not in requirements.txt because it needs an OpenMP runtime that is
awkward on macOS, and the demo must not depend on it. Install it separately:

    .venv/bin/python -m pip install lightgbm
    .venv/bin/python lightgbm_check.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
SEED = 42
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType",
               "VisitorType", "Weekend"]

df = pd.read_csv(ROOT / "online_shoppers_intention.csv").drop_duplicates().reset_index(drop=True)
y = df["Revenue"].astype(int)
X = df.drop(columns=["Revenue", "PageValues"])
numeric = [c for c in X if c not in CATEGORICAL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
prep = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ],
    sparse_threshold=0,
)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)

candidates = {
    "LightGBM (defaults)": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1),
    "LightGBM (balanced)": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1,
                                          class_weight="balanced"),
    "LightGBM (tuned)": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1,
                                       class_weight="balanced", n_estimators=300,
                                       learning_rate=0.05, num_leaves=15,
                                       min_child_samples=30, reg_lambda=1.0),
}

print(f"{'model':<24} {'CV AP':>8} {'test AP':>9} {'test AUC':>9}")
print("-" * 54)
proba = {}
cv_scores = {}
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocess", prep), ("model", estimator)])
    cv_scores[name] = float(np.mean(
        cross_validate(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)["test_score"]))
    pipe.fit(X_train, y_train)
    proba[name] = pipe.predict_proba(X_test)[:, 1]
    print(f"{name:<24} {cv_scores[name]:>8.4f} "
          f"{average_precision_score(y_test, proba[name]):>9.4f} "
          f"{roc_auc_score(y_test, proba[name]):>9.4f}")

ours = joblib.load(ROOT / "artifacts" / "purchase_model.joblib")
ours_proba = ours.predict_proba(X_test)[:, 1]
ours_cv = 0.3907        # from artifacts/results.json, selected on training folds only
print(f"{'Ours (tuned forest)':<24} {ours_cv:>8.4f} "
      f"{average_precision_score(y_test, ours_proba):>9.4f} "
      f"{roc_auc_score(y_test, ours_proba):>9.4f}")

best = max(candidates, key=lambda n: average_precision_score(y_test, proba[n]))
print(f"\nSelection rule used by train.py is cross-validated AP on training data only.")
print(f"  ours {ours_cv:.4f}  vs  best LightGBM {cv_scores[best]:.4f}  -> "
      f"{'ours still selected' if ours_cv > cv_scores[best] else 'LightGBM would be selected'}")

# Is the holdout difference real, or inside the noise?
rng = np.random.default_rng(SEED)
y_true = y_test.to_numpy()
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true), len(y_true))
    if y_true[idx].sum() == 0:
        continue
    diffs.append(average_precision_score(y_true[idx], proba[best][idx])
                 - average_precision_score(y_true[idx], ours_proba[idx]))
diffs = np.asarray(diffs)
lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
print(f"\nPaired bootstrap on the holdout, {best} minus ours:")
print(f"  mean {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
      f"LightGBM ahead in {100 * (diffs > 0).mean():.1f}% of resamples")
print(f"  significant: {'YES' if (lo > 0 or hi < 0) else 'NO, the interval crosses zero'}")
