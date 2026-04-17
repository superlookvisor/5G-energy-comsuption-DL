"""Redraw proxy-weight ablation figures in an IEEE TSG-friendly style.

The script reads the ablation outputs produced by the Phase-B day-ahead
experiment and writes publication-oriented PDF/SVG/PNG figures. It does not
modify the original experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "Physical": "Physical",
    "SemiPhysical_Ridge": "Semi-physical Ridge",
    "SemiPhysical_Lasso": "Semi-physical Lasso",
    "RandomForest": "Random Forest",
}

STRATEGY_LABELS = {
    "two_stage_proxy": "Two-stage proxy",
    "historical_proxy": "Historical proxy",
}

SERIES_COLORS = {
    "fixed": "#4C78A8",
    "learned": "#F58518",
    "two_stage_proxy": "#4C78A8",
    "historical_proxy": "#54A24B",
}


def configure_tsg_style() -> None:
    """Use restrained typography and line weights suitable for IEEE columns."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 8,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.2,
            "patch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi)


def pick_fixed_baseline_best(strict: pd.DataFrame) -> pd.DataFrame:
    """Pick one model per strategy using the fixed-weight baseline MAPE."""
    selected = (
        strict.sort_values(["strategy", "MAPE_fixed", "peak_error_fixed", "valley_error_fixed"])
        .groupby("strategy", as_index=False)
        .head(1)
        .copy()
    )
    strategy_order = ["two_stage_proxy", "historical_proxy"]
    selected["strategy_order"] = selected["strategy"].map({s: i for i, s in enumerate(strategy_order)})
    selected = selected.sort_values(["strategy_order", "MAPE_fixed"]).drop(columns=["strategy_order"])
    return selected


def plot_strict_metric_bars(strict: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Figure 4: grouped bars for strict 24 h trajectory metrics."""
    selected = pick_fixed_baseline_best(strict)
    labels = [
        f"{STRATEGY_LABELS.get(row.strategy, row.strategy)}\n({MODEL_LABELS.get(row.model, row.model)})"
        for row in selected.itertuples()
    ]
    x = np.arange(len(selected))
    width = 0.32

    metrics = [
        ("MAPE", "MAPE", "MAPE", "{:.3f}"),
        ("peak_error", "Peak error", "Energy units", "{:.2f}"),
        ("valley_error", "Valley error", "Energy units", "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), constrained_layout=True)
    for ax, (key, title, ylabel, fmt) in zip(axes, metrics):
        fixed = selected[f"{key}_fixed"].to_numpy(dtype=float)
        learned = selected[f"{key}_learned"].to_numpy(dtype=float)
        ax.bar(
            x - width / 2,
            fixed,
            width,
            label="Fixed",
            color=SERIES_COLORS["fixed"],
            edgecolor="black",
            linewidth=0.45,
        )
        ax.bar(
            x + width / 2,
            learned,
            width,
            label="Learned",
            color=SERIES_COLORS["learned"],
            edgecolor="black",
            linewidth=0.45,
            hatch="//",
        )
        for xpos, value in zip(x - width / 2, fixed):
            ax.text(xpos, value, fmt.format(value), ha="center", va="bottom", fontsize=6, rotation=90)
        for xpos, value in zip(x + width / 2, learned):
            ax.text(xpos, value, fmt.format(value), ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle=":", alpha=0.75)
        ax.set_axisbelow(True)
        ymax = max(fixed.max(), learned.max())
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)

    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    save_all(fig, out_dir, "fig4_strict_proxy_weight_ablation_tsg")
    plt.close(fig)

    selected.to_csv(out_dir / "fig4_selected_fixed_baseline_best.csv", index=False)
    return selected


def best_fixed_models_by_strategy(strict: pd.DataFrame) -> dict[str, str]:
    selected = pick_fixed_baseline_best(strict)
    return dict(zip(selected["strategy"], selected["model"]))


def plot_horizon_delta_mae(horizon: pd.DataFrame, strict: pd.DataFrame, out_dir: Path) -> None:
    """Draw per-horizon MAE change for the fixed-baseline-best models."""
    selected_models = best_fixed_models_by_strategy(strict)
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)

    for strategy, model in selected_models.items():
        sub = horizon[(horizon["strategy"] == strategy) & (horizon["model"] == model)].copy()
        sub = sub.sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["MAE_delta"],
            marker="o",
            label=f"{STRATEGY_LABELS.get(strategy, strategy)} ({MODEL_LABELS.get(model, model)})",
            color=SERIES_COLORS.get(strategy, "#333333"),
        )

    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Delta MAE (learned - fixed, energy units)")
    ax.set_xlim(1, 24)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.grid(axis="y", linestyle=":", alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    save_all(fig, out_dir, "fig_horizon_delta_mae_tsg")
    plt.close(fig)


def plot_prediction_delta_hist(aligned: pd.DataFrame, strict: pd.DataFrame, out_dir: Path) -> None:
    """Draw the prediction shift distribution for the selected final model."""
    selected = pick_fixed_baseline_best(strict)
    final = selected.sort_values("MAPE_learned").iloc[0]
    sub = aligned[(aligned["strategy"] == final["strategy"]) & (aligned["model"] == final["model"])].copy()

    fig, ax = plt.subplots(figsize=(3.5, 2.25), constrained_layout=True)
    ax.hist(
        sub["energy_pred_delta"].dropna().to_numpy(dtype=float),
        bins=40,
        color="#7F7F7F",
        edgecolor="black",
        linewidth=0.35,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    mean_delta = float(sub["energy_pred_delta"].mean())
    ax.axvline(mean_delta, color=SERIES_COLORS["learned"], linewidth=1.0, linestyle="--")
    ax.set_xlabel("Prediction change (learned - fixed, energy units)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{STRATEGY_LABELS.get(final['strategy'], final['strategy'])}, "
        f"{MODEL_LABELS.get(final['model'], final['model'])}"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.75)
    ax.set_axisbelow(True)
    save_all(fig, out_dir, "fig_prediction_delta_hist_tsg")
    plt.close(fig)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_proxy_impact_filtered",
        help="Directory containing compare_strict_metrics.csv and related ablation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures_tsg_proxy_impact",
        help="Directory for redrawn TSG-style figures.",
    )
    args = parser.parse_args()

    strict_path = args.input_dir / "compare_strict_metrics.csv"
    horizon_path = args.input_dir / "compare_horizon_metrics.csv"
    aligned_path = args.input_dir / "compare_predictions_aligned.csv"
    require_files([strict_path, horizon_path, aligned_path])

    configure_tsg_style()
    strict = pd.read_csv(strict_path)
    horizon = pd.read_csv(horizon_path)
    aligned = pd.read_csv(aligned_path)

    selected = plot_strict_metric_bars(strict, args.output_dir)
    plot_horizon_delta_mae(horizon, strict, args.output_dir)
    plot_prediction_delta_hist(aligned, strict, args.output_dir)

    print(f"Wrote TSG-style figures to: {args.output_dir}")
    print("Fixed-baseline-best models used in Fig. 4:")
    print(selected[["strategy", "model", "MAPE_fixed", "MAPE_learned"]].to_string(index=False))


if __name__ == "__main__":
    main()
