"""#2 Split and seed stability: does the 'winner' and the score survive a different split?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from robustness.common import OUT, fit_score, load_data, save

N_SEEDS = 20
MODELS = {"rf": "Random forest", "gb": "Gradient boosting", "lr": "Logistic regression"}


def main() -> None:
    X, y = load_data()
    rows = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        for key in MODELS:
            rows.append({"seed": seed, "model": MODELS[key], **fit_score(key, X_tr, y_tr, X_te, y_te, seed)})
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["ap", "auc", "recall", "f1", "top10_lift"]].agg(["mean", "std", "min", "max"])
    wins = df.loc[df.groupby("seed")["ap"].idxmax(), "model"].value_counts().to_dict()
    save("seeds", {"n_seeds": N_SEEDS, "best_model_wins_by_ap": wins,
                   "summary": {m: {f"{c}_{s}": float(v) for (c, s), v in summary.loc[m].items()} for m in summary.index},
                   "runs": rows})
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    ax.boxplot([df[df.model == m]["ap"] for m in MODELS.values()], labels=list(MODELS.values()))
    ax.set_ylabel("Test average precision")
    ax.set_title(f"AP across {N_SEEDS} random splits")
    fig.savefig(OUT / "seeds.png", dpi=170)


if __name__ == "__main__":
    main()
