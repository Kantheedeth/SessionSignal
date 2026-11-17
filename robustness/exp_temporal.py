"""#3 Temporal shift: train on earlier months, test on later ones; ablate Month."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import pandas as pd

from robustness.common import MONTH_ORDER, OUT, fit_score, load_data, save


def main() -> None:
    X, y = load_data()
    order = X["Month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    result = {"month_counts": X["Month"].value_counts().to_dict()}

    # Forward split: train through Aug, test Sep-Dec. Month cannot be a usable feature
    # here (test months never seen), so also report with Month dropped.
    train = order <= MONTH_ORDER.index("Aug")
    for label, drop in [("with_month", None), ("without_month", ["Month"])]:
        result[f"forward_{label}"] = fit_score("rf", X[train], y[train], X[~train], y[~train], drop=drop)

    # Leave-one-month-out: how badly does an unseen month hurt?
    lomo = []
    for m in MONTH_ORDER:
        te = X["Month"] == m
        for label, drop in [("with_month", None), ("without_month", ["Month"])]:
            r = fit_score("rf", X[~te], y[~te], X[te], y[te], drop=drop)
            lomo.append({"month": m, "variant": label, **r})
        print(f"held out {m}", flush=True)
    result["leave_one_month_out"] = lomo
    save("temporal", result)

    df = pd.DataFrame(lomo)
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    for label, colour in [("with_month", "#1876a3"), ("without_month", "#d79032")]:
        d = df[df.variant == label].set_index("month").reindex(MONTH_ORDER)
        ax.plot(d.index, d["ap"], marker="o", label=label.replace("_", " "), color=colour)
    ax.set_ylabel("AP on held-out month")
    ax.set_title("Leave-one-month-out average precision")
    ax.legend(frameon=False)
    fig.savefig(OUT / "temporal.png", dpi=170)


if __name__ == "__main__":
    main()
