"""Build residual day-ahead trajectory data for Phase C PG-RNP-CQR."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASEB_DEFAULT_DIR = REPO_ROOT / "phaseB_dayahead_only_20260415" / "outputs_filter_compare_learned"
PHASEB_REBUILD_SCRIPT = REPO_ROOT / "phaseB_dynamic_energy_20260414_rebuild" / "run_phaseB_dynamic_energy.py"

BASELINE_STRATEGY = "two_stage_proxy"
PHYSICS_MODEL = "Physical"
STRONG_BASELINE_MODEL = "RandomForest"


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: pd.DataFrame
    baseline_predictions: pd.DataFrame
    complete_trajectory_ids: List[str]
    feature_columns: List[str]
    metadata: Dict[str, object]


def _load_phaseb_rebuild():
    name = "phaseb_rebuild_for_phasec"
    spec = importlib.util.spec_from_file_location(name, PHASEB_REBUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase B script: {PHASEB_REBUILD_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, parse_dates=list(parse_dates or []))


def _load_phaseb_tables(phaseb_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = _read_csv(phaseb_dir / "dayahead_predictions.csv", parse_dates=["origin_time", "target_time"])
    panel = _read_csv(phaseb_dir / "panel_dataset.csv", parse_dates=["Time"])
    metrics = _read_csv(phaseb_dir / "dayahead_metrics.csv")
    return predictions, panel, metrics


def _build_two_stage_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Recreate Phase B two-stage proxy input columns from the saved panel."""
    pb = _load_phaseb_rebuild()
    raw_datasets, _ = pb.build_dayahead_dataset(panel.copy())
    two_stage = pb.prepare_dayahead_features(raw_datasets["two_stage_proxy"], "two_stage_proxy")

    static_cols = ["n_cells", "sum_pmax", "sum_bandwidth", "sum_antennas", "mean_frequency", "high_freq_ratio"]
    ratio_cols = [c for c in panel.columns if c.startswith("mode_ratio_") or c.startswith("ru_ratio_")]
    static = panel.sort_values("Time").groupby("BS", as_index=False).first()
    static = static[["BS"] + [c for c in static_cols + ratio_cols if c in static.columns]]
    two_stage = two_stage.merge(static, on="BS", how="left", suffixes=("", "_static"))

    for col in static_cols:
        alt = f"{col}_static"
        if alt in two_stage.columns:
            if col not in two_stage.columns:
                two_stage[col] = two_stage[alt]
            else:
                two_stage[col] = two_stage[col].fillna(two_stage[alt])
    return two_stage


def _horizon_completeness(df: pd.DataFrame) -> List[str]:
    counts = df.groupby("trajectory_id")["horizon"].nunique()
    return counts[counts == 24].index.astype(str).tolist()


def _feature_columns(df: pd.DataFrame) -> List[str]:
    explicit = [
        "horizon", "target_hour", "target_day_of_week", "target_hour_sin", "target_hour_cos",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "p_base", "dynamic_phys_hat", "E_phys_hat",
        "load_mean_hat", "load_pmax_hat", "load_std_hat", "D1_hat", "D2_hat", "D3_hat",
        "load_mean_roll24", "load_pmax_roll24", "load_std_roll24", "n_cells", "sum_pmax",
        "sum_bandwidth", "sum_antennas", "mean_frequency", "high_freq_ratio",
    ]
    prefixes = ("S_ESMode", "I_ESMode", "mode_ratio_", "ru_ratio_")
    prefixed = [c for c in df.columns if c.startswith(prefixes) and not c.endswith("_hour_prior")]
    return list(dict.fromkeys([c for c in explicit + prefixed if c in df.columns]))


def build_residual_dataset(phaseb_dir: Path = PHASEB_DEFAULT_DIR, output_dir: Optional[Path] = None) -> DatasetBuildResult:
    predictions, panel, phaseb_metrics = _load_phaseb_tables(phaseb_dir)
    features = _build_two_stage_features(panel)
    key = ["BS", "trajectory_id", "origin_time", "target_time", "horizon"]

    phys = predictions[(predictions["strategy"] == BASELINE_STRATEGY) & (predictions["model"] == PHYSICS_MODEL)]
    phys = phys[key + ["y_pred", "energy_pred"]].rename(columns={"y_pred": "dynamic_phys_hat", "energy_pred": "E_phys_hat"})
    if phys.empty:
        raise ValueError(f"Missing {BASELINE_STRATEGY} + {PHYSICS_MODEL} rows in Phase B predictions")

    strong = predictions[(predictions["strategy"] == BASELINE_STRATEGY) & (predictions["model"] == STRONG_BASELINE_MODEL)]
    strong = strong[key + ["energy_pred"]].rename(columns={"energy_pred": "rf_energy_pred"})

    dataset = features.merge(phys, on=key, how="inner").merge(strong, on=key, how="left")
    dataset["residual_true"] = dataset["energy_true"] - dataset["E_phys_hat"]
    dataset["origin_date"] = dataset["origin_time"].dt.date.astype(str)
    complete_ids = _horizon_completeness(dataset)
    dataset["is_complete_24"] = dataset["trajectory_id"].isin(complete_ids)
    dataset["physical_lower"] = 0.0
    dataset["physical_upper"] = dataset["p_base"] + 2.0 * dataset["sum_pmax"].fillna(0.0) + 10.0
    dataset["physical_upper"] = np.maximum(dataset["physical_upper"], dataset["E_phys_hat"] + 1.0)

    feat_cols = _feature_columns(dataset)
    for col in feat_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    metadata = {
        "phaseb_dir": str(phaseb_dir),
        "baseline_strategy": BASELINE_STRATEGY,
        "physics_model": PHYSICS_MODEL,
        "strong_baseline_model": STRONG_BASELINE_MODEL,
        "n_rows": int(len(dataset)),
        "n_bs": int(dataset["BS"].nunique()),
        "n_trajectories": int(dataset["trajectory_id"].nunique()),
        "n_complete_24_trajectories": int(len(complete_ids)),
        "feature_columns": feat_cols,
        "phaseb_best_rows": phaseb_metrics.head(8).to_dict(orient="records"),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_dir / "residual_trajectory_dataset.csv", index=False)
        predictions.to_csv(output_dir / "phaseb_baseline_predictions.csv", index=False)
        (output_dir / "dataset_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return DatasetBuildResult(dataset, predictions.copy(), complete_ids, feat_cols, metadata)

