"""Is the model comparison biased because only the forest was tuned?

train.py tunes the strongest family and leaves the others at their initial
settings. That is a fair criticism, and the fix is not simply to tune
everything - it is to give every family the *same tuning budget*. A bigger
grid for one model is just a different kind of unfairness.

So this script gives each family an identical allowance: 20 randomly sampled
candidate configurations, scored on the same four folds, with the same
average-precision metric and the same seed. 80 fits each, no exceptions. The
search spaces are wide enough that none of them is artificially constrained.

Everything else matches train.py - same cleaned data, same PageValues
exclusion, same stratified split at seed 42, same untouched holdout.

    .venv/bin/python fair_tuning_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
SEED = 42
N_CANDIDATES = 20          # identical allowance for every family
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


# Each family gets a search space over its own meaningful hyperparameters.
# Logistic regression simply has fewer knobs; that is a property of the model,
# not a handicap we imposed.
families = {
    "Logistic regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        {"model__C": loguniform(1e-3, 1e3)},
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"model__learning_rate": loguniform(0.01, 0.3),
         "model__max_leaf_nodes": randint(5, 64),
         "model__l2_regularization": loguniform(1e-3, 10),
         "model__min_samples_leaf": randint(5, 60),
         "model__max_iter": randint(100, 400)},
    ),
    "Random forest": (
        RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
        {"model__n_estimators": randint(100, 400),
         "model__max_depth": [4, 8, 12, 16, 24, None],
         "model__min_samples_leaf": randint(1, 20),
         "model__max_features": ["sqrt", "log2", 0.3, 0.5]},
    ),
}

print(f"Equal budget: {N_CANDIDATES} candidates x 4 folds = {N_CANDIDATES * 4} fits per family\n")
print(f"{'family':<22} {'untuned CV':>11} {'tuned CV':>9} {'gain':>8} {'test AP':>9}")
print("-" * 64)
results = {}
for name, (estimator, space) in families.items():
    base = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
    untuned = float(np.mean(cross_validate(
        base, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)["test_score"]))
    search = RandomizedSearchCV(base, space, n_iter=N_CANDIDATES, scoring="average_precision",
                                cv=cv, random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    results[name] = {"untuned": untuned, "tuned": search.best_score_,
                     "test": average_precision_score(y_test, proba),
                     "proba": proba, "params": search.best_params_}
    print(f"{name:<22} {untuned:>11.4f} {search.best_score_:>9.4f} "
          f"{search.best_score_ - untuned:>+8.4f} {results[name]['test']:>9.4f}")
print("-" * 64)

winner = max(results, key=lambda n: results[n]["tuned"])
print(f"\nSelection rule, highest tuned CV AP: {winner} at {results[winner]['tuned']:.4f}")
print(f"  ranking unchanged from train.py: {winner == 'Random forest'}")
for name in results:
    print(f"  {name:<22} {results[name]['params']}")

# With every family given the same budget, is the holdout difference real?
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
print(f"\nEqually tuned, holdout gap forest minus boosting:")
print(f"  mean {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
      f"forest ahead in {100 * (diffs > 0).mean():.1f}% of resamples")
print(f"  significant: {'YES' if (lo > 0 or hi < 0) else 'NO, still tied'}")
