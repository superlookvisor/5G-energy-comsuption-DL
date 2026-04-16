"""Generate IEEE TSG-style figures for Phase C PG-RNP-CQR results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASEC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = PHASEC_DIR / "outputs"
DEFAULT_FIG_DIR = DEFAULT_OUTPUTS / "ieee_tsg_figures"

MODEL_LABELS = {
    "PhaseB_historical_proxy_Physical": "Hist.-Physical",
    "PhaseB_historical_proxy_RandomForest": "Hist.-RF",
    "PhaseB_historical_proxy_SemiPhysical_Lasso": "Hist.-Lasso",
    "PhaseB_historical_proxy_SemiPhysical_Ridge": "Hist.-Ridge",
    "PhaseB_two_stage_proxy_Physical": "Phys. baseline",
    "PhaseB_two_stage_proxy_RandomForest": "Two-stage RF",
    "PhaseB_two_stage_proxy_SemiPhysical_Lasso": "Two-stage Lasso",
    "PhaseB_two_stage_proxy_SemiPhysical_Ridge": "Two-stage Ridge",
    "A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR": "PG-RNP-CQR",
}


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "SimSun"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def load_tables(outputs: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(outputs / "metrics_overall.csv")
    horizon = pd.read_csv(outputs / "coverage_by_horizon.csv")
    pred = pd.read_csv(outputs / "pg_rnp_predictions.csv", parse_dates=["origin_time", "target_time"])
    viol = pd.read_csv(outputs / "physics_violation_report.csv")
    baseline = pd.read_csv(outputs / "phaseb_baseline_predictions.csv", parse_dates=["origin_time", "target_time"])
    return metrics, horizon, pred, viol, baseline


def plot_overall_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "PhaseB_two_stage_proxy_Physical",
        "PhaseB_two_stage_proxy_SemiPhysical_Ridge",
        "PhaseB_two_stage_proxy_SemiPhysical_Lasso",
        "PhaseB_two_stage_proxy_RandomForest",
        "A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR",
    ]
    sub = metrics[metrics["model_name"].isin(order)].copy()
    sub["label"] = sub["model_name"].map(MODEL_LABELS)
    sub["order"] = sub["model_name"].map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    colors = ["#8aa6c1", "#b8b8b8", "#c9a66b", "#7fa87f", "#c44e52"]
    axes[0].bar(sub["label"], sub["MAE"], color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("(a) Absolute error")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(sub["label"], sub["MAPE"] * 100.0, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("(b) Percentage error")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    savefig(fig, out_dir, "fig01_overall_point_accuracy")


def plot_horizon_performance(horizon: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.1, 3.0))
    ax1.plot(horizon["horizon"], horizon["mae"], marker="o", color="#1f77b4", label="MAE")
    ax1.set_xlabel("Forecast horizon (h)")
    ax1.set_ylabel("MAE", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(range(1, 25, 2))
    ax2 = ax1.twinx()
    ax2.plot(horizon["horizon"], horizon["coverage_90"] * 100.0, marker="s", color="#c44e52", label="90% coverage")
    ax2.axhline(90.0, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.75)
    ax2.set_ylabel("Coverage (%)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True)
    fig.tight_layout()
    savefig(fig, out_dir, "fig02_horizon_mae_coverage")


def plot_interval_width(horizon: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.bar(horizon["horizon"], horizon["avg_width_90"], color="#8aa6c1", edgecolor="black", linewidth=0.35)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Avg. 90% interval width")
    ax.set_xticks(range(1, 25, 4))
    fig.tight_layout()
    savefig(fig, out_dir, "fig03_interval_width_by_horizon")


def _representative_trajectory(pred: pd.DataFrame) -> pd.DataFrame:
    test = pred[(pred["split"] == "test") & (pred["is_complete_24"].astype(bool))].copy()
    if test.empty:
        counts = pred[pred["split"] == "test"].groupby("trajectory_id")["horizon"].nunique().sort_values(ascending=False)
        traj = counts.index[0]
        return pred[(pred["split"] == "test") & (pred["trajectory_id"] == traj)].sort_values("horizon")
    errors = (
        test.groupby("trajectory_id")
        .apply(lambda g: np.mean(np.abs(g["energy_true"] - g["energy_pred"])), include_groups=False)
        .sort_values()
    )
    traj = errors.index[len(errors) // 2]
    return test[test["trajectory_id"] == traj].sort_values("horizon")


def plot_representative_trajectory(pred: pd.DataFrame, baseline: pd.DataFrame, out_dir: Path) -> None:
    one = _representative_trajectory(pred)
    key = ["BS", "trajectory_id", "origin_time", "target_time", "horizon"]
    rf = baseline[(baseline["strategy"] == "two_stage_proxy") & (baseline["model"] == "RandomForest")]
    rf = rf[key + ["energy_pred"]].rename(columns={"energy_pred": "rf_pred"})
    one = one.merge(rf, on=key, how="left")

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.fill_between(one["horizon"], one["lower_90"], one["upper_90"], color="#4c78a8", alpha=0.18, label="90% interval")
    ax.plot(one["horizon"], one["energy_true"], marker="o", color="black", label="Actual")
    ax.plot(one["horizon"], one["energy_pred"], marker="s", color="#c44e52", label="PG-RNP-CQR")
    if one["rf_pred"].notna().any():
        ax.plot(one["horizon"], one["rf_pred"], marker="^", color="#59a14f", label="Two-stage RF")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Energy")
    ax.set_title(f"Trajectory: {one['trajectory_id'].iloc[0]}")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    savefig(fig, out_dir, "fig04_representative_uncertainty_trajectory")


def plot_prediction_scatter(pred: pd.DataFrame, out_dir: Path) -> None:
    test = pred[pred["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.scatter(test["energy_true"], test["energy_pred"], s=8, alpha=0.45, color="#4c78a8", edgecolors="none")
    lo = min(test["energy_true"].min(), test["energy_pred"].min())
    hi = max(test["energy_true"].max(), test["energy_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("PG-RNP-CQR test predictions")
    fig.tight_layout()
    savefig(fig, out_dir, "fig05_prediction_scatter")


def plot_residual_distribution(pred: pd.DataFrame, out_dir: Path) -> None:
    test = pred[pred["split"] == "test"].copy()
    residual = test["energy_true"] - test["energy_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    axes[0].hist(residual, bins=45, density=True, color="#8aa6c1", edgecolor="black", linewidth=0.35)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Prediction residual")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) Residual distribution")
    sorted_r = np.sort(residual.to_numpy())
    probs = (np.arange(len(sorted_r)) + 0.5) / len(sorted_r)
    axes[1].plot(sorted_r, probs, color="#c44e52")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction residual")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("(b) Empirical CDF")
    fig.tight_layout()
    savefig(fig, out_dir, "fig06_residual_distribution")


def plot_esmode_error(pred: pd.DataFrame, out_dir: Path) -> None:
    test = pred[pred["split"] == "test"].copy()
    rows = []
    for mode in [1, 2, 6]:
        col = f"S_ESMode{mode}_hat"
        if col not in test:
            continue
        active = test[test[col] > 0.05]
        inactive = test[test[col] <= 1e-6]
        if active.empty or inactive.empty:
            continue
        rows.append({"mode": f"ES{mode}", "state": "Active", "MAE": np.abs(active["energy_true"] - active["energy_pred"]).mean(), "n": len(active)})
        rows.append({"mode": f"ES{mode}", "state": "Inactive", "MAE": np.abs(inactive["energy_true"] - inactive["energy_pred"]).mean(), "n": len(inactive)})
    es = pd.DataFrame(rows)
    es.to_csv(out_dir / "esmode_error_summary.csv", index=False)
    if es.empty:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    modes = list(es["mode"].unique())
    x = np.arange(len(modes))
    width = 0.35
    for i, state in enumerate(["Inactive", "Active"]):
        vals = [float(es[(es["mode"] == m) & (es["state"] == state)]["MAE"].iloc[0]) for m in modes]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=state, edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("MAE")
    ax.set_title("Error under ES-mode activation")
    ax.legend(frameon=True)
    fig.tight_layout()
    savefig(fig, out_dir, "fig07_esmode_active_error")


def plot_physics_violation(viol: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    x = np.arange(len(viol))
    ax.bar(x - 0.18, viol["raw_interval_below_lower_rate"] * 100, width=0.36, label="Lower violation", edgecolor="black", linewidth=0.35)
    ax.bar(x + 0.18, viol["raw_interval_above_upper_rate"] * 100, width=0.36, label="Upper violation", edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(viol["split"])
    ax.set_ylabel("Raw interval violation (%)")
    ax.set_title("Physical feasibility before projection")
    ax.legend(frameon=True)
    fig.tight_layout()
    savefig(fig, out_dir, "fig08_physical_violation_rates")


def write_manifest(out_dir: Path, metrics: pd.DataFrame, horizon: pd.DataFrame) -> None:
    pg = metrics[metrics["model_name"] == "A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR"].iloc[0].to_dict()
    manifest = {
        "figure_dir": str(out_dir),
        "formats": ["png", "pdf"],
        "dpi": 600,
        "pg_rnp_cqr": pg,
        "mean_coverage_90": float(horizon["coverage_90"].mean()),
        "mean_interval_width_90": float(horizon["avg_width_90"].mean()),
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE TSG-style figures for Phase C")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    set_ieee_style()
    metrics, horizon, pred, viol, baseline = load_tables(args.outputs)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_overall_bars(metrics, args.fig_dir)
    plot_horizon_performance(horizon, args.fig_dir)
    plot_interval_width(horizon, args.fig_dir)
    plot_representative_trajectory(pred, baseline, args.fig_dir)
    plot_prediction_scatter(pred, args.fig_dir)
    plot_residual_distribution(pred, args.fig_dir)
    plot_esmode_error(pred, args.fig_dir)
    plot_physics_violation(viol, args.fig_dir)
    write_manifest(args.fig_dir, metrics, horizon)
    print(f"IEEE TSG-style figures written to: {args.fig_dir.resolve()}")


if __name__ == "__main__":
    main()

