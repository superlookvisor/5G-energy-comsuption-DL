"""Run Phase C PG-RNP-CQR residual day-ahead experiment."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Phase C PG-RNP-CQR requires PyTorch for model training. "
        "Install torch in the active Python environment, then rerun this script."
    ) from exc

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phaseC_pg_rnp_cqr_20260416.build_residual_trajectory_dataset import PHASEB_DEFAULT_DIR, build_residual_dataset
from phaseC_pg_rnp_cqr_20260416.evaluate_pg_rnp_cqr import (
    baseline_ablation_rows,
    horizonwise_cqr,
    metric_row,
    metrics_by_horizon,
    physics_violation_report,
    write_report,
)
from phaseC_pg_rnp_cqr_20260416.pg_rnp_model import PGRNPConfig, predict_rows, split_bs, train_model

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C PG-RNP-CQR residual trajectory forecasting")
    p.add_argument("--phaseb-dir", type=Path, default=PHASEB_DEFAULT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--likelihood", choices=["student_t", "gaussian"], default="student_t")
    p.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test cap after dataset build.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def leakage_checks(df: pd.DataFrame, feature_cols: list[str]) -> None:
    forbidden = {"Energy", "energy_true", "dynamic_energy", "y_true", "residual_true"}
    bad = sorted(forbidden.intersection(feature_cols))
    if bad:
        raise ValueError(f"Target leakage columns found in feature set: {bad}")
    delta = (pd.to_datetime(df["target_time"]) - pd.to_datetime(df["origin_time"])).dt.total_seconds() / 3600.0
    if not (delta == df["horizon"]).all():
        raise ValueError("Trajectory consistency check failed: target_time-origin_time != horizon")
    resid = df["energy_true"] - df["E_phys_hat"]
    if (resid - df["residual_true"]).abs().max() > 1e-6:
        raise ValueError("Physical baseline residual check failed")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    build = build_residual_dataset(args.phaseb_dir, args.output_dir)
    df = build.dataset.copy()
    if args.max_rows is not None and args.max_rows > 0:
        keep_traj = df[["trajectory_id"]].drop_duplicates().head(max(1, args.max_rows // 24))["trajectory_id"]
        df = df[df["trajectory_id"].isin(keep_traj)].copy()
        df.to_csv(args.output_dir / "residual_trajectory_dataset.csv", index=False)

    leakage_checks(df, build.feature_columns)

    config = PGRNPConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        likelihood=args.likelihood,
    )
    splits = split_bs(df, config)
    if set(splits["train"]) & set(splits["calibration"]) or set(splits["train"]) & set(splits["test"]) or set(splits["calibration"]) & set(splits["test"]):
        raise ValueError("BS split overlap detected")
    (args.output_dir / "split_bs.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    device = torch.device(args.device) if args.device else None
    model, scaler, residual_stats, history = train_model(df, build.feature_columns, splits, config, args.output_dir, device=device)
    with open(args.output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    raw_pred = predict_rows(model, df, build.feature_columns, splits, scaler, residual_stats["mean"], residual_stats["std"], config, device=device)
    cqr_pred, q_by_h = horizonwise_cqr(raw_pred, alpha=0.10)
    cqr_pred.to_csv(args.output_dir / "pg_rnp_predictions.csv", index=False)
    (args.output_dir / "cqr_quantiles.json").write_text(json.dumps(q_by_h, indent=2), encoding="utf-8")

    test_pred = cqr_pred[cqr_pred["split"] == "test"].copy()
    pg_overall = pd.DataFrame([metric_row(test_pred, "A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR")])
    baseline_rows = baseline_ablation_rows(build.baseline_predictions, splits["test"])
    overall = pd.concat([baseline_rows, pg_overall], ignore_index=True, sort=False)
    overall.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    overall.to_csv(args.output_dir / "ablation_metrics.csv", index=False)

    by_h = metrics_by_horizon(test_pred)
    by_h.to_csv(args.output_dir / "metrics_by_horizon.csv", index=False)
    by_h.to_csv(args.output_dir / "coverage_by_horizon.csv", index=False)
    violations = physics_violation_report(cqr_pred)
    violations.to_csv(args.output_dir / "physics_violation_report.csv", index=False)

    run_meta = {
        "config": asdict(config),
        "residual_stats": residual_stats,
        "history_tail": {k: v[-5:] for k, v in history.items()},
        "dataset": build.metadata,
        "n_features": len(build.feature_columns),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output_dir, overall, by_h, violations, q_by_h, build.metadata)
    print(f"Phase C PG-RNP-CQR finished. Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
