"""Run Phase D BESS dispatchability analysis.

This phase converts existing day-ahead point/probabilistic forecasts into
backup reserve, dispatchable capacity, reliability, and value metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PHASEC_DEFAULT_OUTPUT_DIR = REPO_ROOT / "phaseC_pg_rnp_cqr_20260416" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"

KEY_COLS = ["BS", "trajectory_id", "origin_time", "target_time", "horizon"]
EPS = 1e-9

POLICIES: Mapping[str, Tuple[str, str]] = {
    "Oracle": ("R_safe_oracle", "oracle"),
    "Phase B Point": ("R_safe_phaseB", "phaseB"),
    "Phase C Point": ("R_safe_phaseC", "phaseC"),
    "Phase C Aggregate-CQR": ("R_safe_cqr", "cqr"),
    "Hourly Upper Sum": ("R_hourly_upper", "hourly_upper"),
}


@dataclass(frozen=True)
class PhaseDConfig:
    phasec_output_dir: str
    output_dir: str
    phaseb_strategy: str
    phaseb_model: str
    backup_duration: int
    epsilon: float
    capacity_hours: float
    soc: float
    soc_min: float
    dispatch_price: float
    shortfall_penalty: float
    evaluation_split: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D BESS dispatchability model")
    p.add_argument("--phasec-output-dir", type=Path, default=PHASEC_DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--phaseb-strategy", default="two_stage_proxy")
    p.add_argument("--phaseb-model", default="RandomForest")
    p.add_argument("--backup-duration", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument(
        "--capacity-hours",
        type=float,
        default=8.0,
        help="Synthetic BESS capacity C_b = capacity_hours * p_base.",
    )
    p.add_argument("--soc", type=float, default=1.0)
    p.add_argument("--soc-min", type=float, default=0.0)
    p.add_argument("--dispatch-price", type=float, default=1.0)
    p.add_argument("--shortfall-penalty", type=float, default=10.0)
    p.add_argument(
        "--evaluation-split",
        choices=["test", "calibration", "train", "all"],
        default="test",
        help="Split used for reserve_policy_metrics.csv.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_hourly_predictions(
    phasec_output_dir: Path,
    phaseb_strategy: str,
    phaseb_model: str,
) -> pd.DataFrame:
    phasec_path = phasec_output_dir / "pg_rnp_predictions.csv"
    phaseb_path = phasec_output_dir / "phaseb_baseline_predictions.csv"
    if not phasec_path.exists():
        raise FileNotFoundError(str(phasec_path))
    if not phaseb_path.exists():
        raise FileNotFoundError(str(phaseb_path))

    phasec = pd.read_csv(phasec_path, parse_dates=["origin_time", "target_time"])
    phaseb_all = pd.read_csv(phaseb_path, parse_dates=["origin_time", "target_time"])

    require_columns(
        phasec,
        KEY_COLS + ["split", "energy_true", "energy_pred", "lower_90", "upper_90", "p_base"],
        "pg_rnp_predictions.csv",
    )
    require_columns(
        phaseb_all,
        KEY_COLS + ["strategy", "model", "energy_pred"],
        "phaseb_baseline_predictions.csv",
    )

    phaseb = phaseb_all[
        (phaseb_all["strategy"].astype(str) == phaseb_strategy)
        & (phaseb_all["model"].astype(str) == phaseb_model)
    ].copy()
    if phaseb.empty:
        available = (
            phaseb_all[["strategy", "model"]]
            .drop_duplicates()
            .sort_values(["strategy", "model"])
            .to_dict(orient="records")
        )
        raise ValueError(
            f"No Phase B baseline rows for strategy={phaseb_strategy!r}, model={phaseb_model!r}. "
            f"Available combinations: {available}"
        )

    if phasec.duplicated(KEY_COLS).any():
        dup = int(phasec.duplicated(KEY_COLS).sum())
        raise ValueError(f"Phase C predictions contain duplicated hourly keys: {dup}")
    if phaseb.duplicated(KEY_COLS).any():
        dup = int(phaseb.duplicated(KEY_COLS).sum())
        raise ValueError(f"Selected Phase B predictions contain duplicated hourly keys: {dup}")

    c_cols = KEY_COLS + [
        "split",
        "energy_true",
        "energy_pred",
        "lower_90",
        "upper_90",
        "p_base",
    ]
    hourly = phasec[c_cols].rename(columns={"energy_pred": "energy_pred_phaseC"})
    phaseb = phaseb[KEY_COLS + ["energy_pred"]].rename(columns={"energy_pred": "energy_pred_phaseB"})
    hourly = hourly.merge(phaseb, on=KEY_COLS, how="left", validate="one_to_one")

    missing_phaseb = int(hourly["energy_pred_phaseB"].isna().sum())
    if missing_phaseb:
        raise ValueError(f"Phase B predictions are missing for {missing_phaseb} Phase C rows")

    for col in ["horizon", "energy_true", "energy_pred_phaseB", "energy_pred_phaseC", "lower_90", "upper_90", "p_base"]:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    before = len(hourly)
    hourly = hourly.dropna(
        subset=["horizon", "energy_true", "energy_pred_phaseB", "energy_pred_phaseC", "upper_90", "p_base"]
    ).copy()
    if len(hourly) != before:
        print(f"Dropped {before - len(hourly)} hourly rows with missing numeric values")

    hourly["horizon"] = hourly["horizon"].astype(int)
    hourly["hourly_error_phaseB"] = hourly["energy_pred_phaseB"] - hourly["energy_true"]
    hourly["hourly_error_phaseC"] = hourly["energy_pred_phaseC"] - hourly["energy_true"]
    return hourly


def split_label(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    return str(modes.iloc[0]) if not modes.empty else "unknown"


def safe_p_base(group: pd.DataFrame) -> float:
    pbase = pd.to_numeric(group["p_base"], errors="coerce").dropna()
    if not pbase.empty and float(pbase.median()) > 0:
        return float(pbase.median())
    fallback = pd.to_numeric(group["energy_true"], errors="coerce").dropna()
    if not fallback.empty:
        return float(max(fallback.median(), EPS))
    return 1.0


def build_window_dataset(
    hourly: pd.DataFrame,
    backup_duration: int,
    capacity_hours: float,
    soc: float,
    soc_min: float,
) -> pd.DataFrame:
    if backup_duration < 1 or backup_duration > 24:
        raise ValueError("--backup-duration must be in [1, 24]")
    if capacity_hours <= 0:
        raise ValueError("--capacity-hours must be positive")
    if soc < 0 or soc_min < 0 or soc < soc_min:
        raise ValueError("SOC parameters must satisfy soc >= soc_min >= 0")

    rows: List[Dict[str, object]] = []
    group_cols = ["BS", "trajectory_id", "origin_time"]

    for (bs, trajectory_id, origin_time), group in hourly.groupby(group_cols, sort=False):
        group = group.sort_values("horizon").drop_duplicates("horizon").set_index("horizon", drop=False)
        horizons = set(int(h) for h in group.index)
        base = safe_p_base(group)
        c_bess = capacity_hours * base
        usable_bess = max(0.0, (soc - soc_min) * c_bess)
        split = split_label(group["split"])

        max_start = 24 - backup_duration + 1
        for start_h in range(1, max_start + 1):
            need = list(range(start_h, start_h + backup_duration))
            if not all(h in horizons for h in need):
                continue
            w = group.loc[need]
            start_row = group.loc[start_h]
            r_true = float(w["energy_true"].sum())
            r_phaseb = float(w["energy_pred_phaseB"].sum())
            r_phasec = float(w["energy_pred_phaseC"].sum())
            r_upper = float(w["upper_90"].sum())

            rows.append(
                {
                    "BS": bs,
                    "trajectory_id": trajectory_id,
                    "origin_time": origin_time,
                    "outage_start_time": start_row["target_time"],
                    "outage_start_horizon": int(start_h),
                    "backup_duration": int(backup_duration),
                    "split": split,
                    "R_true": r_true,
                    "R_pred_phaseB": r_phaseb,
                    "R_pred_phaseC": r_phasec,
                    "R_hourly_upper": r_upper,
                    "hourly_error_start_phaseB": float(start_row["hourly_error_phaseB"]),
                    "hourly_error_start_phaseC": float(start_row["hourly_error_phaseC"]),
                    "cumulative_error_phaseB": r_phaseb - r_true,
                    "cumulative_error_phaseC": r_phasec - r_true,
                    "C_bess": c_bess,
                    "usable_bess_energy": usable_bess,
                    "p_base": base,
                    "soc": float(soc),
                    "soc_min": float(soc_min),
                }
            )

    if not rows:
        raise ValueError(
            f"No consecutive {backup_duration}-hour backup windows could be built from Phase C predictions"
        )
    windows = pd.DataFrame(rows)
    windows["origin_time"] = pd.to_datetime(windows["origin_time"])
    windows["outage_start_time"] = pd.to_datetime(windows["outage_start_time"])
    return windows.sort_values(["split", "BS", "origin_time", "outage_start_horizon"]).reset_index(drop=True)


def conformal_upper_quantile(scores: pd.Series, epsilon: float) -> float:
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("--epsilon must be in (0, 1)")
    clean = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("Cannot calibrate Aggregate-CQR because the calibration score set is empty")
    ordered = np.sort(clean)
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - epsilon)))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


def apply_reserve_policies(windows: pd.DataFrame, q_agg: float) -> pd.DataFrame:
    out = windows.copy()
    out["R_safe_oracle"] = out["R_true"]
    out["R_safe_phaseB"] = out["R_pred_phaseB"].clip(lower=0.0)
    out["R_safe_phaseC"] = out["R_pred_phaseC"].clip(lower=0.0)
    out["aggregate_cqr_q"] = float(q_agg)
    out["R_safe_cqr"] = (out["R_pred_phaseC"] + q_agg).clip(lower=0.0)
    out["R_hourly_upper"] = out["R_hourly_upper"].clip(lower=0.0)

    for _, (reserve_col, suffix) in POLICIES.items():
        reserve = out[reserve_col].astype(float)
        out[f"C_disp_{suffix}"] = (out["usable_bess_energy"] - reserve).clip(lower=0.0)
        out[f"shortfall_{suffix}"] = (out["R_true"] - reserve).clip(lower=0.0)
        out[f"overreserve_{suffix}"] = (reserve - out["R_true"]).clip(lower=0.0)
        out[f"reserve_error_{suffix}"] = reserve - out["R_true"]

    out["bess_capacity_shortfall"] = (out["R_true"] - out["usable_bess_energy"]).clip(lower=0.0)
    return out


def evaluation_frame(windows: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return windows.copy()
    sub = windows[windows["split"].astype(str) == split].copy()
    if sub.empty:
        available = sorted(windows["split"].dropna().astype(str).unique().tolist())
        raise ValueError(f"No windows for evaluation split {split!r}. Available splits: {available}")
    return sub


def metric_rows(
    windows_eval: pd.DataFrame,
    dispatch_price: float,
    shortfall_penalty: float,
    evaluation_split_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    oracle_mean_disp = float(windows_eval["C_disp_oracle"].mean())
    total_capacity = float(windows_eval["C_bess"].sum())
    capacity_violation_rate = float((windows_eval["bess_capacity_shortfall"] > 0).mean())
    mean_capacity_shortfall = float(windows_eval["bess_capacity_shortfall"].mean())

    for policy, (reserve_col, suffix) in POLICIES.items():
        disp = windows_eval[f"C_disp_{suffix}"].astype(float)
        short = windows_eval[f"shortfall_{suffix}"].astype(float)
        over = windows_eval[f"overreserve_{suffix}"].astype(float)
        err = windows_eval[f"reserve_error_{suffix}"].astype(float)
        reserve = windows_eval[reserve_col].astype(float)
        net_total = float((dispatch_price * disp - shortfall_penalty * short).sum())
        rows.append(
            {
                "policy": policy,
                "evaluation_split": evaluation_split_name,
                "n_windows": int(len(windows_eval)),
                "n_bs": int(windows_eval["BS"].nunique()),
                "mean_dispatchable_capacity": float(disp.mean()),
                "dispatchable_capacity_ratio": float(disp.sum() / max(total_capacity, EPS)),
                "dispatchable_capacity_loss_vs_oracle": float(oracle_mean_disp - disp.mean()),
                "shortfall_rate": float((short > 0).mean()),
                "mean_shortfall_energy": float(short.mean()),
                "p95_shortfall_energy": float(short.quantile(0.95)),
                "mean_overreserve_energy": float(over.mean()),
                "reliability_satisfaction_rate": float(1.0 - (short > 0).mean()),
                "mean_reserve_energy": float(reserve.mean()),
                "mean_reserve_error": float(err.mean()),
                "p05_reserve_error": float(err.quantile(0.05)),
                "p95_reserve_error": float(err.quantile(0.95)),
                "net_dispatch_value": float(net_total / max(len(windows_eval), 1)),
                "total_net_dispatch_value": net_total,
                "bess_capacity_violation_rate": capacity_violation_rate,
                "mean_bess_capacity_shortfall": mean_capacity_shortfall,
            }
        )
    return pd.DataFrame(rows)


def horizon_metrics(windows_eval: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for horizon, group in windows_eval.groupby("outage_start_horizon"):
        row: Dict[str, object] = {
            "outage_start_horizon": int(horizon),
            "n_windows": int(len(group)),
            "mean_R_true": float(group["R_true"].mean()),
        }
        for policy, (_, suffix) in POLICIES.items():
            row[f"{suffix}_mean_reserve_error"] = float(group[f"reserve_error_{suffix}"].mean())
            row[f"{suffix}_shortfall_rate"] = float((group[f"shortfall_{suffix}"] > 0).mean())
            row[f"{suffix}_mean_dispatchable_capacity"] = float(group[f"C_disp_{suffix}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("outage_start_horizon")


def write_csv_outputs(
    output_dir: Path,
    hourly: pd.DataFrame,
    windows: pd.DataFrame,
    metrics: pd.DataFrame,
    by_horizon: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_dir / "aligned_hourly_predictions.csv", index=False)
    windows.to_csv(output_dir / "bess_window_dataset.csv", index=False)
    metrics.to_csv(output_dir / "reserve_policy_metrics.csv", index=False)
    by_horizon.to_csv(output_dir / "reserve_error_by_horizon.csv", index=False)


def policy_order(metrics: pd.DataFrame) -> List[str]:
    order = list(POLICIES.keys())
    present = set(metrics["policy"].astype(str))
    return [p for p in order if p in present]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def save_fig1_error_propagation(windows_eval: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = windows_eval["hourly_error_start_phaseC"].to_numpy(dtype=float)
    y = windows_eval["cumulative_error_phaseC"].to_numpy(dtype=float)
    ax.scatter(x, y, s=10, alpha=0.45, color="#276FBF", edgecolors="none")
    if len(x) > 1 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#B23A48", linewidth=1.5, label="linear fit")
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.98, f"corr={corr:.2f}", transform=ax.transAxes, va="top", ha="left")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Hourly forecast error propagates to reserve error")
    ax.set_xlabel("Phase C outage-start hourly error")
    ax.set_ylabel("Phase C cumulative reserve error")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_error_propagation.png")
    plt.close(fig)


def save_fig2_dispatchable_capacity(metrics: pd.DataFrame, output_dir: Path) -> None:
    order = policy_order(metrics)
    sub = metrics.set_index("policy").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = ["#4D4D4D", "#276FBF", "#2A9D8F", "#B23A48", "#8A6F3D"]
    ax.bar(sub["policy"], sub["mean_dispatchable_capacity"], color=colors[: len(sub)])
    ax.set_title("Mean dispatchable capacity by reserve policy")
    ax.set_ylabel("Mean dispatchable capacity")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_mean_dispatchable_capacity.png")
    plt.close(fig)


def save_fig3_tradeoff(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = {
        "Oracle": "#4D4D4D",
        "Phase B Point": "#276FBF",
        "Phase C Point": "#2A9D8F",
        "Phase C Aggregate-CQR": "#B23A48",
        "Hourly Upper Sum": "#8A6F3D",
    }
    for _, row in metrics.iterrows():
        policy = str(row["policy"])
        ax.scatter(
            row["shortfall_rate"],
            row["mean_dispatchable_capacity"],
            s=60,
            color=colors.get(policy, "#555555"),
        )
        ax.annotate(policy, (row["shortfall_rate"], row["mean_dispatchable_capacity"]), xytext=(4, 3), textcoords="offset points")
    ax.set_title("Reliability-dispatchability trade-off")
    ax.set_xlabel("Shortfall rate")
    ax.set_ylabel("Mean dispatchable capacity")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_shortfall_dispatch_tradeoff.png")
    plt.close(fig)


def save_fig4_horizon_error(by_horizon: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    series = [
        ("Phase B Point", "phaseB_mean_reserve_error", "#276FBF"),
        ("Phase C Point", "phaseC_mean_reserve_error", "#2A9D8F"),
        ("Aggregate-CQR", "cqr_mean_reserve_error", "#B23A48"),
        ("Hourly Upper Sum", "hourly_upper_mean_reserve_error", "#8A6F3D"),
    ]
    for label, col, color in series:
        if col in by_horizon.columns:
            ax.plot(by_horizon["outage_start_horizon"], by_horizon[col], marker="o", linewidth=1.5, label=label, color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Reserve error by outage start horizon")
    ax.set_xlabel("Outage start horizon")
    ax.set_ylabel("Mean reserve error")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_reserve_error_by_horizon.png")
    plt.close(fig)


def save_fig5_cqr_vs_hourly(metrics: pd.DataFrame, output_dir: Path) -> None:
    keep = ["Phase C Point", "Phase C Aggregate-CQR", "Hourly Upper Sum"]
    sub = metrics[metrics["policy"].isin(keep)].set_index("policy").loc[keep].reset_index()
    x = np.arange(len(sub))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].bar(x, sub["mean_dispatchable_capacity"], color="#2A9D8F")
    axes[0].set_title("Capacity release")
    axes[0].set_ylabel("Mean dispatchable capacity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub["policy"], rotation=25, ha="right")

    axes[1].bar(x - width / 2, sub["shortfall_rate"], width=width, label="shortfall rate", color="#B23A48")
    axes[1].bar(x + width / 2, sub["mean_overreserve_energy"], width=width, label="mean overreserve", color="#8A6F3D")
    axes[1].set_title("Risk and conservatism")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub["policy"], rotation=25, ha="right")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_cqr_hourly_upper_comparison.png")
    plt.close(fig)


def write_figures(output_dir: Path, windows_eval: pd.DataFrame, metrics: pd.DataFrame, by_horizon: pd.DataFrame) -> None:
    setup_plot_style()
    save_fig1_error_propagation(windows_eval, output_dir)
    save_fig2_dispatchable_capacity(metrics, output_dir)
    save_fig3_tradeoff(metrics, output_dir)
    save_fig4_horizon_error(by_horizon, output_dir)
    save_fig5_cqr_vs_hourly(metrics, output_dir)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, cols: List[str], digits: int = 4) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def metric_lookup(metrics: pd.DataFrame, policy: str, col: str) -> float:
    sub = metrics[metrics["policy"] == policy]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0][col])


def write_report(
    output_dir: Path,
    config: PhaseDConfig,
    hourly: pd.DataFrame,
    windows: pd.DataFrame,
    windows_eval: pd.DataFrame,
    metrics: pd.DataFrame,
    q_agg: float,
) -> None:
    target = 1.0 - config.epsilon
    phaseb_short = metric_lookup(metrics, "Phase B Point", "shortfall_rate")
    phasec_short = metric_lookup(metrics, "Phase C Point", "shortfall_rate")
    cqr_rel = metric_lookup(metrics, "Phase C Aggregate-CQR", "reliability_satisfaction_rate")
    cqr_disp = metric_lookup(metrics, "Phase C Aggregate-CQR", "mean_dispatchable_capacity")
    upper_disp = metric_lookup(metrics, "Hourly Upper Sum", "mean_dispatchable_capacity")
    cqr_short = metric_lookup(metrics, "Phase C Aggregate-CQR", "mean_shortfall_energy")
    upper_short = metric_lookup(metrics, "Hourly Upper Sum", "mean_shortfall_energy")
    cqr_over = metric_lookup(metrics, "Phase C Aggregate-CQR", "mean_overreserve_energy")
    upper_over = metric_lookup(metrics, "Hourly Upper Sum", "mean_overreserve_energy")
    cqr_net = metric_lookup(metrics, "Phase C Aggregate-CQR", "net_dispatch_value")
    upper_net = metric_lookup(metrics, "Hourly Upper Sum", "net_dispatch_value")
    if cqr_short > upper_short:
        breakeven_penalty = config.dispatch_price * (cqr_disp - upper_disp) / max(cqr_short - upper_short, EPS)
        breakeven_text = (
            f"At dispatch price `{config.dispatch_price:.3f}`, Aggregate-CQR has higher net value than "
            f"Hourly Upper Sum when the shortfall penalty is below approximately `{breakeven_penalty:.4f}`."
        )
    else:
        breakeven_text = (
            "Aggregate-CQR has no higher mean shortfall than Hourly Upper Sum in this run, so its extra "
            "dispatchable capacity is not offset by the shortfall penalty."
        )

    cols = [
        "policy",
        "mean_dispatchable_capacity",
        "dispatchable_capacity_ratio",
        "shortfall_rate",
        "mean_shortfall_energy",
        "mean_overreserve_energy",
        "reliability_satisfaction_rate",
        "mean_reserve_error",
        "net_dispatch_value",
    ]

    report = f"""# Phase D BESS Dispatchability Analysis Report

## Research Design

Phase D maps the existing day-ahead load forecasts into BESS reserve and dispatchability metrics. It does not retrain Phase B or Phase C models. The pipeline is:

1. Align hourly Phase B point forecasts, Phase C point forecasts, Phase C CQR upper bounds, and true energy.
2. Build consecutive `{config.backup_duration}`-hour outage windows within each day-ahead trajectory.
3. Calibrate Aggregate-CQR on calibration windows with one-sided scores `R_true - R_pred_phaseC`.
4. Evaluate reserve policies on `{config.evaluation_split}` windows.
5. Convert reserve into dispatchable BESS capacity, reserve shortfall, overreserve, and simplified dispatch value.

## Inputs And Assumptions

- Phase C output directory: `{config.phasec_output_dir}`
- Phase B point baseline: `{config.phaseb_strategy} + {config.phaseb_model}`
- Backup duration: `{config.backup_duration}` hours
- CQR risk level epsilon: `{config.epsilon:.3f}`; target reliability: `{target:.3f}`
- Aggregate-CQR reserve add-on `q_(1-epsilon)`: `{q_agg:.6f}`
- BESS scenario: `C_b = {config.capacity_hours:.2f} * p_base`, `SOC={config.soc:.2f}`, `SOC_min={config.soc_min:.2f}`
- Economic scenario: dispatch price `{config.dispatch_price:.3f}`, shortfall penalty `{config.shortfall_penalty:.3f}`

## Data Coverage

- Aligned hourly rows: `{len(hourly)}`
- All BESS windows: `{len(windows)}`
- Evaluation windows: `{len(windows_eval)}`
- Evaluation base stations: `{windows_eval['BS'].nunique()}`
- Evaluation outage-start horizons: `{int(windows_eval['outage_start_horizon'].min())}` to `{int(windows_eval['outage_start_horizon'].max())}`
- BESS capacity violation rate under this fixed-capacity scenario: `{(windows_eval['bess_capacity_shortfall'] > 0).mean():.4f}`

## Main Metrics

{markdown_table(metrics, cols)}

## Findings

- Phase C point forecasts change the reserve shortfall rate from `{phaseb_short:.4f}` to `{phasec_short:.4f}` relative to the selected Phase B point baseline.
- Aggregate-CQR reaches reliability satisfaction `{cqr_rel:.4f}` against the target `{target:.4f}` on the evaluation split.
- Compared with Hourly Upper Sum, Aggregate-CQR releases `{cqr_disp - upper_disp:.4f}` more mean dispatchable capacity per window.
- Aggregate-CQR mean overreserve is `{cqr_over:.4f}`, while Hourly Upper Sum mean overreserve is `{upper_over:.4f}`; this quantifies the cost of simply summing hourly upper bounds.
- Under the configured economic scenario, Aggregate-CQR net value is `{cqr_net:.4f}` and Hourly Upper Sum net value is `{upper_net:.4f}`. {breakeven_text}
- Reserve error is cumulative: Fig. 1 links outage-start hourly forecast error to the `{config.backup_duration}`-hour reserve error, and Fig. 4 shows how this error varies across outage start horizons.

## Outputs

- `aligned_hourly_predictions.csv`
- `bess_window_dataset.csv`
- `reserve_policy_metrics.csv`
- `reserve_error_by_horizon.csv`
- `run_meta.json`
- `fig1_error_propagation.png`
- `fig2_mean_dispatchable_capacity.png`
- `fig3_shortfall_dispatch_tradeoff.png`
- `fig4_reserve_error_by_horizon.png`
- `fig5_cqr_hourly_upper_comparison.png`

## Interpretation

The results directly support the Phase D claim: day-ahead forecast error is not only an energy-prediction issue, because it propagates into the backup reserve requirement. Point forecasts can release more dispatchable capacity when they underestimate reserve demand, but that capacity is purchased with higher shortfall risk. Aggregate-CQR uses calibration at the cumulative backup-window level, so it targets the actual BESS reliability constraint more directly than summing hourly upper bounds.
"""
    (output_dir / "dispatchability_analysis_report.md").write_text(report, encoding="utf-8")


def write_meta(output_dir: Path, config: PhaseDConfig, hourly: pd.DataFrame, windows: pd.DataFrame, q_agg: float) -> None:
    meta = {
        "config": asdict(config),
        "aggregate_cqr_q": q_agg,
        "hourly_rows": int(len(hourly)),
        "hourly_base_stations": int(hourly["BS"].nunique()),
        "hourly_trajectories": int(hourly["trajectory_id"].nunique()),
        "window_rows": int(len(windows)),
        "window_base_stations": int(windows["BS"].nunique()),
        "window_trajectories": int(windows["trajectory_id"].nunique()),
        "windows_by_split": windows["split"].value_counts().to_dict(),
        "windows_by_start_horizon": {
            str(int(k)): int(v) for k, v in windows["outage_start_horizon"].value_counts().sort_index().items()
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = PhaseDConfig(
        phasec_output_dir=str(args.phasec_output_dir),
        output_dir=str(args.output_dir),
        phaseb_strategy=args.phaseb_strategy,
        phaseb_model=args.phaseb_model,
        backup_duration=args.backup_duration,
        epsilon=args.epsilon,
        capacity_hours=args.capacity_hours,
        soc=args.soc,
        soc_min=args.soc_min,
        dispatch_price=args.dispatch_price,
        shortfall_penalty=args.shortfall_penalty,
        evaluation_split=args.evaluation_split,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_predictions(args.phasec_output_dir, args.phaseb_strategy, args.phaseb_model)
    windows = build_window_dataset(
        hourly=hourly,
        backup_duration=args.backup_duration,
        capacity_hours=args.capacity_hours,
        soc=args.soc,
        soc_min=args.soc_min,
    )

    calibration = windows[windows["split"].astype(str) == "calibration"].copy()
    q_agg = conformal_upper_quantile(calibration["R_true"] - calibration["R_pred_phaseC"], args.epsilon)
    windows = apply_reserve_policies(windows, q_agg)
    windows_eval = evaluation_frame(windows, args.evaluation_split)
    metrics = metric_rows(windows_eval, args.dispatch_price, args.shortfall_penalty, args.evaluation_split)
    by_horizon = horizon_metrics(windows_eval)

    write_csv_outputs(args.output_dir, hourly, windows, metrics, by_horizon)
    write_figures(args.output_dir, windows_eval, metrics, by_horizon)
    write_report(args.output_dir, config, hourly, windows, windows_eval, metrics, q_agg)
    write_meta(args.output_dir, config, hourly, windows, q_agg)

    print(f"Phase D finished. Outputs: {args.output_dir.resolve()}")
    print(metrics[["policy", "mean_dispatchable_capacity", "shortfall_rate", "reliability_satisfaction_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
