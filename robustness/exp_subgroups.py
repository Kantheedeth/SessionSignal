"""#6 Subgroup robustness: per-segment performance and train-on-one-segment tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from robustness.common import OUT, SEED, fit_score, load_data, make_pipeline, metrics, save

SEGMENTS = ["VisitorType", "Weekend", "Region", "TrafficType", "Browser"]
MIN_ROWS = 100


def main() -> None:
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    pipe = make_pipeline("rf", columns=X.columns.tolist()).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    overall = metrics(y_te, proba)

    per_group = []
    for seg in SEGMENTS:
        for val in sorted(X_te[seg].unique(), key=str):
            m = (X_te[seg] == val).to_numpy()
            if m.sum() >= MIN_ROWS and y_te[m].sum() >= 5:
                per_group.append({"segment": seg, "value": str(val), **metrics(y_te[m], proba[m])})
    worst = min(per_group, key=lambda r: r["ap"])

    # Train on returning visitors only, test on new visitors (and vice versa).
    cross = []
    for a, b in [("Returning_Visitor", "New_Visitor"), ("New_Visitor", "Returning_Visitor")]:
        tr, te = X["VisitorType"] == a, X["VisitorType"] == b
        cross.append({"train": a, "test": b, **fit_score("rf", X[tr], y[tr], X[te], y[te])})

    save("subgroups", {"overall": overall, "per_group": per_group, "worst_group": worst, "cross_segment": cross})

    top = sorted(per_group, key=lambda r: r["ap"])[:12]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh([f"{r['segment']}={r['value']} (n={r['n']})" for r in top], [r["ap"] for r in top], color="#00a89d")
    ax.axvline(overall["ap"], color="#d79032", linestyle="--", label=f"overall {overall['ap']:.2f}")
    ax.set_xlabel("AP"); ax.set_title("Weakest segments"); ax.legend(frameon=False)
    fig.savefig(OUT / "subgroups.png", dpi=170)


if __name__ == "__main__":
    main()
