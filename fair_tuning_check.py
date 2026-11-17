"""Is the model comparison biased because only the forest was tuned?

train.py tunes the strongest family and leaves the others at their initial
settings. That is a fair criticism: the tuned model gets an advantage the
others never had. This script removes the asymmetry by grid-searching every
family under the same protocol - same cleaned data, same PageValues exclusion,
same stratified split at seed 42, same four folds, same average-precision
metric - and checks whether the selection actually changes.

Gradient boosting is deliberately given the largest grid (18 combinations
against the forest's 6), so the test is tilted against our own conclusion.

    .venv/bin/python fair_tuning_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
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
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)


def preprocessor():
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
        ],
        sparse_threshold=0,
    )


families = {
    "Logistic regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=15,
                                       l2_regularization=1.0, random_state=SEED),
        {"model__max_leaf_nodes": [7, 15, 31], "model__l2_regularization": [0.1, 1.0],
         "model__learning_rate": [0.03, 0.06, 0.12]},
    ),
    "Random forest": (
        RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
        {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [2, 5]},
    ),
}

print(f"{'family':<22} {'untuned CV':>11} {'tuned CV':>9} {'gain':>8} {'tuned test AP':>14} {'grid':>6}")
print("-" * 78)
results = {}
for name, (estimator, grid) in families.items():
    base = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
    untuned = float(np.mean(cross_validate(
        base, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)["test_score"]))
    search = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    results[name] = {"untuned": untuned, "tuned": search.best_score_,
                     "test": average_precision_score(y_test, proba),
                     "proba": proba, "params": search.best_params_}
    combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"{name:<22} {untuned:>11.4f} {search.best_score_:>9.4f} "
          f"{search.best_score_ - untuned:>+8.4f} {results[name]['test']:>14.4f} {combos:>6}")
print("-" * 78)

winner = max(results, key=lambda n: results[n]["tuned"])
print(f"\nSelection rule, highest tuned CV AP: {winner} at {results[winner]['tuned']:.4f}")
print(f"  best params: {results[winner]['params']}")
print(f"  ranking unchanged from train.py: {winner == 'Random forest'}")

# With both tuned, is the holdout difference significant?
rng = np.random.default_rng(SEED)
y_true = y_test.to_numpy()
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true), len(y_true))
    if y_true[idx].sum() == 0:
        continue
    diffs.append(average_precision_score(y_true[idx], results["Random forest"]["proba"][idx])
                 - average_precision_score(y_true[idx], results["Gradient boosting"]["proba"][idx]))
diffs = np.asarray(diffs)
lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
print(f"\nBoth tuned, holdout gap forest minus boosting:")
print(f"  mean {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
      f"forest ahead in {100 * (diffs > 0).mean():.1f}% of resamples")
print(f"  significant: {'YES' if (lo > 0 or hi < 0) else 'NO, still tied'}")
