"""#7 Threshold, calibration and worklist stability across seeds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.model_selection import train_test_split

from robustness.common import OUT, THRESHOLD, load_data, make_pipeline, save

N_SEEDS = 20


def main() -> None:
    X, y = load_data()
    rows, curves = [], []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        p = make_pipeline("rf", seed, columns=X.columns.tolist()).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        yt = y_te.to_numpy()
        prec, rec, cuts = precision_recall_curve(yt, p)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
        best = int(np.argmax(f1[:-1]))
        fixed = p >= THRESHOLD
        top50 = np.argsort(p)[-50:]
        rows.append({
            "seed": seed,
            "best_f1_threshold": float(cuts[best]), "best_f1": float(f1[best]),
            "f1_at_fixed_0.446": float(f1[np.argmin(np.abs(cuts - THRESHOLD))]),
            "recall_at_fixed_0.446": float(yt[fixed].sum() / yt.sum()),
            "top50_buyers": int(yt[top50].sum()), "top50_expected_random": float(50 * yt.mean()),
            "brier": float(brier_score_loss(yt, p)),
        })
        curves.append(calibration_curve(yt, p, n_bins=8, strategy="quantile"))
        print(f"seed {seed}", flush=True)

    arr = lambda k: np.array([r[k] for r in rows])
    summary = {k: {"mean": float(arr(k).mean()), "std": float(arr(k).std()), "min": float(arr(k).min()), "max": float(arr(k).max())}
               for k in ("best_f1_threshold", "best_f1", "f1_at_fixed_0.446", "recall_at_fixed_0.446", "top50_buyers", "brier")}
    save("threshold", {"fixed_threshold": THRESHOLD, "summary": summary, "runs": rows})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax[0].hist(arr("best_f1_threshold"), bins=10, color="#1876a3")
    ax[0].axvline(THRESHOLD, color="#d79032", linestyle="--", label="fixed 0.446")
    ax[0].set_xlabel("Best-F1 threshold per split"); ax[0].set_title("Does the cutoff move?"); ax[0].legend(frameon=False)
    for frac, mean in curves:
        ax[1].plot(mean, frac, color="#00a89d", alpha=0.35)
    ax[1].plot([0, 1], [0, 1], color="#8da2ae", linestyle="--")
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Observed purchase rate"); ax[1].set_title("Calibration (20 splits)")
    fig.savefig(OUT / "threshold.png", dpi=170)


if __name__ == "__main__":
    main()
