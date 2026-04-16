"""Evaluation and horizon-wise conformal calibration for PG-RNP-CQR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


def horizonwise_cqr(pred: pd.DataFrame, alpha: float = 0.10) -> Tuple[pd.DataFrame, Dict[int, float]]:
    cal = pred[pred["split"] == "calibration"].copy()
    q_by_h: Dict[int, float] = {}
    for h in range(1, 25):
        sub = cal[cal["horizon"] == h]
        if sub.empty:
            q_by_h[h] = 1.645
            continue
        scores = np.abs(sub["residual_true"] - sub["residual_mu"]) / np.maximum(sub["residual_sigma"], EPS)
        level = min(np.ceil((len(scores) + 1) * (1 - alpha)) / max(len(scores), 1), 1.0)
        q_by_h[h] = float(np.quantile(scores, level))

    out = pred.copy()
    out["cqr_q"] = out["horizon"].map(q_by_h).astype(float)
    out["lower_90_raw"] = out["E_phys_hat"] + out["residual_mu"] - out["cqr_q"] * out["residual_sigma"]
    out["upper_90_raw"] = out["E_phys_hat"] + out["residual_mu"] + out["cqr_q"] * out["residual_sigma"]
    out["lower_90"] = np.maximum(out["lower_90_raw"], out["physical_lower"])
    out["upper_90"] = np.minimum(out["upper_90_raw"], out["physical_upper"])
    lo = np.minimum(out["lower_90"], out["upper_90"])
    hi = np.maximum(out["lower_90"], out["upper_90"])
    out["lower_90"], out["upper_90"] = lo, hi
    out["model_name"] = "A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR"
    return out, q_by_h


def _trajectory_extreme_error(df: pd.DataFrame, peak: bool) -> float:
    vals = []
    for _, g in df.groupby("trajectory_id"):
        idx = g["energy_true"].idxmax() if peak else g["energy_true"].idxmin()
        vals.append(abs(float(g.loc[idx, "energy_true"] - g.loc[idx, "energy_pred"])))
    return float(np.mean(vals)) if vals else np.nan


def metric_row(df: pd.DataFrame, model_name: str) -> Dict[str, float | str | int]:
    err = df["energy_true"] - df["energy_pred"]
    ape = np.abs(err) / np.maximum(np.abs(df["energy_true"]), EPS)
    complete = df[df["is_complete_24"].astype(bool)] if "is_complete_24" in df.columns else df.iloc[0:0]
    if not complete.empty:
        strict = complete.groupby("trajectory_id").apply(
            lambda g: (np.abs(g["energy_true"] - g["energy_pred"]) / np.maximum(np.abs(g["energy_true"]), EPS)).mean(),
            include_groups=False,
        )
        strict_mape = float(strict.mean())
    else:
        strict_mape = np.nan
    return {
        "model_name": model_name,
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(ape.mean()),
        "strict_24h_trajectory_MAPE": strict_mape,
        "peak_error": float(_trajectory_extreme_error(df, peak=True)),
        "valley_error": float(_trajectory_extreme_error(df, peak=False)),
        "n_samples": int(len(df)),
        "n_bs": int(df["BS"].nunique()),
        "n_trajectories": int(df["trajectory_id"].nunique()),
    }


def metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("horizon"):
        err = g["energy_true"] - g["energy_pred"]
        covered = (g["energy_true"] >= g["lower_90"]) & (g["energy_true"] <= g["upper_90"])
        rows.append({
            "horizon": int(h),
            "coverage_90": float(covered.mean()),
            "avg_width_90": float((g["upper_90"] - g["lower_90"]).mean()),
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mape": float((np.abs(err) / np.maximum(np.abs(g["energy_true"]), EPS)).mean()),
            "n_samples": int(len(g)),
            "n_bs": int(g["BS"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("horizon")


def baseline_ablation_rows(phaseb_predictions: pd.DataFrame, test_bs: Iterable[str]) -> pd.DataFrame:
    test_bs = set(map(str, test_bs))
    sub = phaseb_predictions[phaseb_predictions["BS"].astype(str).isin(test_bs)].copy()
    if sub.empty:
        return pd.DataFrame()
    complete = sub.groupby("trajectory_id")["horizon"].nunique()
    complete_ids = set(complete[complete == 24].index)
    sub["is_complete_24"] = sub["trajectory_id"].isin(complete_ids)
    rows = []
    for (strategy, model), g in sub.groupby(["strategy", "model"]):
        rows.append(metric_row(g.copy(), f"PhaseB_{strategy}_{model}"))
    return pd.DataFrame(rows)


def physics_violation_report(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in pred.groupby("split"):
        rows.append({
            "split": split,
            "n_samples": int(len(g)),
            "energy_pred_below_lower_rate": float((g["energy_pred"] < g["physical_lower"]).mean()),
            "energy_pred_above_upper_rate": float((g["energy_pred"] > g["physical_upper"]).mean()),
            "raw_interval_below_lower_rate": float((g["lower_90_raw"] < g["physical_lower"]).mean()),
            "raw_interval_above_upper_rate": float((g["upper_90_raw"] > g["physical_upper"]).mean()),
            "avg_projection_delta": float((np.abs(g["lower_90"] - g["lower_90_raw"]) + np.abs(g["upper_90"] - g["upper_90_raw"])).mean()),
        })
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    overall: pd.DataFrame,
    by_horizon: pd.DataFrame,
    violations: pd.DataFrame,
    q_by_h: Dict[int, float],
    metadata: Dict[str, object],
) -> None:
    best = overall.sort_values("MAPE").iloc[0]
    pg = overall[overall["model_name"].astype(str).str.startswith("A6_")]
    pg_line = "PG-RNP-CQR result is unavailable."
    if not pg.empty and not by_horizon.empty:
        row = pg.iloc[0]
        pg_line = (
            f"PG-RNP-CQR test MAPE={row['MAPE']:.4f}, MAE={row['MAE']:.4f}, "
            f"mean horizon coverage={by_horizon['coverage_90'].mean():.3f}, "
            f"mean width={by_horizon['avg_width_90'].mean():.3f}."
        )
    report = f"""# Phase C PG-RNP-CQR Analysis Report

## Task
This experiment predicts day-ahead 5G base-station energy by learning the residual of the Phase B physical day-ahead baseline.

## Data
- Rows: {metadata.get('n_rows')}
- Base stations: {metadata.get('n_bs')}
- Trajectories: {metadata.get('n_trajectories')}
- Complete 24h trajectories: {metadata.get('n_complete_24_trajectories')}
- Physical baseline: `{metadata.get('baseline_strategy')} + {metadata.get('physics_model')}`

## Main Result
- Best test model by MAPE: `{best['model_name']}` with MAPE={best['MAPE']:.4f}.
- {pg_line}

## Calibration And Feasibility
- Horizon-wise CQR uses an independent residual score quantile per horizon.
- Mean q_hat across horizons: {np.mean(list(q_by_h.values())):.4f}
- Final intervals are projected to `[physical_lower, physical_upper]`; details are in `physics_violation_report.csv`.

## Outputs
- `residual_trajectory_dataset.csv`
- `pg_rnp_predictions.csv`
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `coverage_by_horizon.csv`
- `ablation_metrics.csv`
- `physics_violation_report.csv`
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")

