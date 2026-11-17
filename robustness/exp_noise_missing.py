"""#4 and #5 Feature noise, feature-group ablation, and missing data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from robustness.common import CATEGORICAL, OUT, SEED, load_data, make_pipeline, metrics, save

GROUPS = {
    "page counts": ["Administrative", "Informational", "ProductRelated"],
    "durations": ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"],
    "bounce/exit": ["BounceRates", "ExitRates"],
    "calendar": ["Month", "SpecialDay", "Weekend"],
    "visitor/traffic": ["VisitorType", "TrafficType", "Region"],
    "device": ["OperatingSystems", "Browser"],
}
NOISE_LEVELS = [0, 0.1, 0.25, 0.5, 1.0]  # std of noise as a multiple of each column's std
MISSING_RATES = [0, 0.05, 0.10, 0.20, 0.40]


def main() -> None:
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    numeric = [c for c in X if c not in CATEGORICAL]
    rng = np.random.default_rng(SEED)

    pipe = make_pipeline("rf", columns=X.columns.tolist()).fit(X_tr, y_tr)
    base = metrics(y_te, pipe.predict_proba(X_te)[:, 1])

    # Noise at test time on numeric features (model trained on clean data).
    noise = []
    for level in NOISE_LEVELS:
        Xn = X_te.copy()
        Xn[numeric] = Xn[numeric] + rng.normal(0, 1, Xn[numeric].shape) * Xn[numeric].std().values * level
        Xn[numeric] = Xn[numeric].clip(lower=0)
        noise.append({"noise_std_multiple": level, **metrics(y_te, pipe.predict_proba(Xn)[:, 1])})

    # Missing values: mask at test time AND train time (imputer handles it).
    missing = []
    for rate in MISSING_RATES:
        def mask(frame):
            f = frame.copy().astype(object).where(rng.random(frame.shape) >= rate, np.nan)
            for c in numeric:
                f[c] = f[c].astype(float)
            return f
        p = make_pipeline("rf", columns=X.columns.tolist()).fit(mask(X_tr), y_tr)
        missing.append({"missing_rate": rate, **metrics(y_te, p.predict_proba(mask(X_te))[:, 1])})
        print(f"missing {rate}", flush=True)

    # Feature-group ablation: retrain without the group.
    ablation = []
    for name, cols in GROUPS.items():
        p = make_pipeline("rf", columns=X.columns.tolist(), drop=cols).fit(X_tr.drop(columns=cols), y_tr)
        r = metrics(y_te, p.predict_proba(X_te.drop(columns=cols))[:, 1])
        ablation.append({"dropped": name, "columns": cols, **r, "ap_change": r["ap"] - base["ap"]})
        print(f"dropped {name}", flush=True)

    save("noise_missing", {"baseline": base, "noise": noise, "missing": missing, "group_ablation": ablation})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].plot([n["noise_std_multiple"] for n in noise], [n["ap"] for n in noise], marker="o", color="#1876a3")
    ax[0].set_xlabel("Noise (x column std)"); ax[0].set_ylabel("AP"); ax[0].set_title("Noise on numeric features")
    ax[1].plot([m["missing_rate"] * 100 for m in missing], [m["ap"] for m in missing], marker="o", color="#d79032")
    ax[1].set_xlabel("Values missing (%)"); ax[1].set_ylabel("AP"); ax[1].set_title("Missing data")
    fig.savefig(OUT / "noise_missing.png", dpi=170)


if __name__ == "__main__":
    main()
