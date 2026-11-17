from pathlib import Path
import matplotlib.pyplot as plt
from typing import Sequence
import numpy as np


def save_prediction_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    path: str,
    title: str | None = None,
) -> None:

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of same length")

    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    x = range(n)

    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]

    mosaic = "AA;BB;CC"

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 6),
        constrained_layout=True,
    )

    ax_series = axes["A"]
    ax_resid = axes["B"]
    ax_hist = axes["C"]

    # Série temporelle (vrai vs prédit)
    ax_series.plot(x, y_true, label="Vrai", color="tab:blue", linewidth=1.5)
    ax_series.plot(
        x, y_pred, label="Predicted", color="tab:orange", linewidth=1.5, linestyle="--"
    )
    ax_series.set_ylabel("Value")
    if title is not None:
        ax_series.set_title(title)
    ax_series.legend()
    ax_series.grid(True, alpha=0.3)

    # leftovers
    ax_resid.axhline(0, color="black", linewidth=1)
    ax_resid.scatter(x, residuals, s=8, color="tab:red", alpha=0.7)
    ax_resid.set_xlabel("Index")
    ax_resid.set_ylabel("Leftovers")
    ax_resid.grid(True, alpha=0.3)

    xmin, xmax = min(residuals), max(residuals)
    xticks = np.linspace(xmin, xmax, 20)
    ax_hist.set_xticks(xticks)
    ax_hist.hist(residuals, bins=30, color="tab:green", alpha=0.7)
    ax_hist.set_xlabel("Leftover values")
    ax_hist.set_ylabel("Leftover numbers")
    ax_hist.grid(True, alpha=0.3)

    output_path = Path(path)
    fig.savefig(output_path, dpi=150)


def save_prediction_plot_pct(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    path: str,
    title: str | None = None,
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of same length")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Divided by zero : NaN when y_true == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_residuals = (y_pred - y_true) / y_true * 100.0
        rel_residuals[~np.isfinite(rel_residuals)] = np.nan

    n = len(y_true)
    x = range(n)

    mosaic = "AA;BB;CD"

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 6),
        constrained_layout=True,
    )

    ax_series = axes["A"]
    ax_resid = axes["B"]
    ax_hist = axes["C"]
    ax_table = axes["D"]

    ax_series.plot(x, y_true, label="True", color="tab:blue", linewidth=1.5)
    ax_series.plot(
        x, y_pred, label="Predicted", color="tab:orange", linewidth=1.5, linestyle="--"
    )
    ax_series.set_ylabel("Value")
    if title is not None:
        ax_series.set_title(title + " (gap in %)")
    ax_series.legend()
    ax_series.grid(True, alpha=0.3)

    ax_resid.axhline(0, color="black", linewidth=1)
    ax_resid.scatter(x, rel_residuals, s=8, color="tab:red", alpha=0.7)
    ax_resid.set_xlabel("Index")
    ax_resid.set_ylabel("relative leftovers (%)")
    ax_resid.set_title("relative leftovers (%)")
    ax_resid.grid(True, alpha=0.3)

    # Distribution des résidus relatifs (%)
    valid = np.isfinite(rel_residuals)
    if np.any(valid):
        vals = rel_residuals[valid]
        ax_hist.hist(vals, bins=30, color="tab:green", alpha=0.7)
    ax_hist.set_xlabel("Relative leftovers (%)")
    ax_hist.set_ylabel("Leftovers number")
    ax_hist.grid(True, alpha=0.3)

    ax_table.axis("off")  # pas d’axes autour du tableau

    # bornes des intervalles (en pourcentage)
    bins = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150, 175, 200, 300])

    # histogramme sur ces classes
    counts, bin_edges = np.histogram(rel_residuals, bins=bins)

    interval_labels = [
        "[-100, -75[",
        "[-75, -50[",
        "[-50, -25[",
        "[ -25, 0[",
        "[ 0, 25[",
        "[ 25, 50[",
        "[ 50, 75[",
        "[ 75, 100[",
        "[ 100, 125[",
        "[ 125, 150[",
        "[ 150, 175[",
        "[ 175, 200[",
        "[ 200, 300[",
    ]

    counts = []
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (rel_residuals >= left) & (rel_residuals < right)
        counts.append(int(mask.sum()))

    mask_200_plus = rel_residuals >= 300
    counts.append(int(mask_200_plus.sum()))
    interval_labels.append(">= 300")

    proportions = [c / len(rel_residuals) * 100 for c in counts]

    cell_text = [
        [labels, c, round(p, 2)]
        for labels, c, p in zip(interval_labels, counts, proportions)
    ]

    table = ax_table.table(
        cellText=cell_text,
        colLabels=["intervals (gap in pct)", "Count", "proportion in pct"],
        loc="center",
    )
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # ligne d'en-tête
            cell.visible_edges = "B"  # uniquement bord bas
        else:
            cell.visible_edges = "open"  # aucun bord

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    output_path = Path(path)
    fig.savefig(output_path, dpi=150)


def loss_graph(model, path: Path) -> None:
    evals_result = model.get_evals_result()

    train_loss = evals_result["learn"]["RMSE"]
    valid_loss = evals_result["validation"]["RMSE"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train set RMSE")
    plt.plot(valid_loss, label="Test set RMSE")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("Deacreasing loss (RMSE) CatBoost")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
