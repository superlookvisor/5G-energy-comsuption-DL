"""
Phase B 仅日前：瘦 panel（无日内滞后/交互平方项）、可选 BS 过滤、复用 phaseB_rebuild 的日前训练与物理检查。

运行（在 energy_model_anp 根目录）:
  python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py
    -> 产物写入 outputs/
  python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py --min-merged-obs-per-bs 24 \\
    --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv
    -> 启用任一 BS 过滤时，产物写入 outputs_filter/，与全量 outputs/ 并存
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
PHASEB_SCRIPT = REPO_ROOT / "phaseB_dynamic_energy_20260414_rebuild" / "run_phaseB_dynamic_energy.py"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _configure_windows_utf8_console() -> None:
    """
    Windows 下默认代码页可能导致中文打印乱码；尽量将 stdout/stderr 切到 UTF-8。
    不改变文件读写编码（文件读写已在各处显式指定 encoding）。
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            try:
                reconfig(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _load_phaseb_rebuild():
    name = "phaseb_rebuild_dayahead_only"
    spec = importlib.util.spec_from_file_location(name, PHASEB_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def add_dayahead_panel_features(panel: pd.DataFrame, s_cols: List[str]) -> pd.DataFrame:
    """仅含日前 + run_physics_checks 所需的时间编码、动态项、滚动负载、S 的历史同小时均值。"""
    panel = panel.sort_values(["BS", "Time"]).copy()
    panel["hour"] = panel["Time"].dt.hour
    panel["day_of_week"] = panel["Time"].dt.dayofweek
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"] / 24.0)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"] / 24.0)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["day_of_week"] / 7.0)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["day_of_week"] / 7.0)

    g = panel.groupby("BS", group_keys=False)
    panel["dynamic_energy"] = panel["Energy"] - panel["p_base"]
    panel["load_mean_roll24"] = g["load_mean"].transform(lambda s: s.rolling(24, min_periods=6).mean())
    panel["load_pmax_roll24"] = g["load_pmax_weighted"].transform(lambda s: s.rolling(24, min_periods=6).mean())
    panel["load_std_roll24"] = g["load_mean"].transform(lambda s: s.rolling(24, min_periods=6).std()).fillna(0.0)

    for col in s_cols:
        by_hour = panel.groupby(["BS", "hour"])[col]
        hist_sum = by_hour.cumsum() - panel[col]
        hist_cnt = by_hour.cumcount()
        panel[f"{col}_hour_prior"] = hist_sum / hist_cnt.replace(0, np.nan)
    return panel


def build_dayahead_panel(pb) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    ec, cl, bsinfo = pb.load_raw_data()
    static_features = pb.build_bs_static_features(bsinfo)
    dynamic_features = pb.aggregate_cell_to_bs_time(cl, bsinfo)
    pbase_full, pbase_meta = pb.load_phasea_pbase(static_features)

    panel = ec.merge(dynamic_features, on=["BS", "Time"], how="inner")
    panel = panel.merge(static_features, on="BS", how="left")
    panel = panel.merge(pbase_full, on="BS", how="left")
    panel = panel.dropna(subset=["Energy", "load_pmax_weighted", "load_mean", "p_base"]).copy()
    panel = add_dayahead_panel_features(panel, list(pb.S_COLS))
    return panel, pbase_full, pbase_meta


def _validate_simplex(w: np.ndarray, name: str) -> None:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(w < -1e-10):
        raise ValueError(f"{name} must be non-negative")
    s = float(np.sum(w))
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1, got {s}")


def build_dayahead_dataset_weighted(
    pb,
    panel: pd.DataFrame,
    weights: Dict[str, List[float]],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    复制 pb.build_dayahead_dataset 的逻辑，但将固定权重替换为 weights（全局常数）。
    返回结构与列名合同保持一致，以复用 pb.build_dayahead_model_specs / pb.prepare_dayahead_features。
    """
    w_lm = np.asarray(weights["load_mean"], dtype=float)
    w_lp = np.asarray(weights["load_pmax"], dtype=float)
    w_ls = np.asarray(weights["load_std"], dtype=float)
    w_s = np.asarray(weights["s_hat"], dtype=float)
    _validate_simplex(w_lm, "weights.load_mean")
    _validate_simplex(w_lp, "weights.load_pmax")
    _validate_simplex(w_ls, "weights.load_std")
    _validate_simplex(w_s, "weights.s_hat")

    panel = panel.sort_values(["BS", "Time"]).copy()
    origins = panel[panel["hour"] == 0].copy()
    g = panel.groupby("BS", group_keys=False)
    panel_idx = panel.set_index(["BS", "Time"], drop=False)
    strategy_frames: Dict[str, List[pd.DataFrame]] = {"two_stage_proxy": [], "historical_proxy": []}
    complete_tracker: List[pd.DataFrame] = []

    for horizon in pb.DAYAHEAD_HORIZONS:
        sample = origins.copy()
        sample["origin_time"] = sample["Time"]
        sample["horizon"] = horizon
        sample["target_time"] = g["Time"].shift(-horizon).reindex(sample.index)
        sample["target_hour"] = g["hour"].shift(-horizon).reindex(sample.index)
        sample["target_day_of_week"] = g["day_of_week"].shift(-horizon).reindex(sample.index)
        sample["target_hour_sin"] = g["hour_sin"].shift(-horizon).reindex(sample.index)
        sample["target_hour_cos"] = g["hour_cos"].shift(-horizon).reindex(sample.index)
        sample["y_true"] = g["dynamic_energy"].shift(-horizon).reindex(sample.index)
        sample["energy_true"] = g["Energy"].shift(-horizon).reindex(sample.index)

        # Align prevday/prevweek by calendar time, not by row shift.
        # "prevday_samehour" is defined relative to the target_time: target_time - 24h.
        prevday_time = sample["target_time"] - pd.Timedelta(hours=24)
        prevweek_time = sample["target_time"] - pd.Timedelta(hours=168)
        prevday_key = pd.MultiIndex.from_arrays([sample["BS"], prevday_time])
        prevweek_key = pd.MultiIndex.from_arrays([sample["BS"], prevweek_time])

        sample["load_mean_prevday_samehour"] = panel_idx["load_mean"].reindex(prevday_key).to_numpy()
        sample["load_pmax_prevday_samehour"] = panel_idx["load_pmax_weighted"].reindex(prevday_key).to_numpy()
        sample["load_std_prevday_samehour"] = panel_idx["load_std"].reindex(prevday_key).to_numpy()
        sample["load_mean_prevweek_samehour"] = panel_idx["load_mean"].reindex(prevweek_key).to_numpy()
        sample["load_pmax_prevweek_samehour"] = panel_idx["load_pmax_weighted"].reindex(prevweek_key).to_numpy()

        for col in pb.S_COLS:
            sample[f"{col}_prevday_samehour"] = panel_idx[col].reindex(prevday_key).to_numpy()

        # weighted convex combination (with same fallback chain as original)
        lm_prev = sample["load_mean_prevday_samehour"].fillna(sample["load_mean_roll24"])
        lm_roll = sample["load_mean_roll24"].fillna(sample["load_mean"])
        lm_cur = sample["load_mean"].fillna(0.0)
        sample["load_mean_hat"] = w_lm[0] * lm_prev + w_lm[1] * lm_roll + w_lm[2] * lm_cur

        lp_prev = sample["load_pmax_prevday_samehour"].fillna(sample["load_pmax_roll24"])
        lp_roll = sample["load_pmax_roll24"].fillna(sample["load_pmax_weighted"])
        lp_cur = sample["load_pmax_weighted"].fillna(0.0)
        sample["load_pmax_hat"] = w_lp[0] * lp_prev + w_lp[1] * lp_roll + w_lp[2] * lp_cur

        ls_prev = sample["load_std_prevday_samehour"].fillna(sample["load_std_roll24"])
        ls_roll = sample["load_std_roll24"].fillna(sample["load_std"])
        sample["load_std_hat"] = w_ls[0] * ls_prev + w_ls[1] * ls_roll

        sample["D1_hat"] = sample["sum_pmax"] * sample["load_pmax_hat"]
        sample["D2_hat"] = sample["sum_pmax"] * (sample["load_pmax_hat"] ** 2)
        sample["D3_hat"] = sample["load_std_hat"]

        for col in pb.S_COLS:
            s_prev = sample[f"{col}_prevday_samehour"].fillna(sample[f"{col}_hour_prior"])
            s_prior = sample[f"{col}_hour_prior"].fillna(sample[col])
            sample[f"{col}_hat"] = (w_s[0] * s_prev + w_s[1] * s_prior).fillna(0.0)
            sample[f"I_{col.split('_', 1)[1]}_hat"] = sample[f"{col}_hat"] * sample["load_pmax_hat"]

        # historical proxy side: keep same behavior as original (no learned weights)
        sample["load_mean_proxy_24"] = sample["load_mean_prevday_samehour"]
        sample["load_mean_proxy_168"] = sample["load_mean_prevweek_samehour"]
        sample["load_mean_roll_proxy"] = sample["load_mean_roll24"]
        sample["load_std_roll_proxy"] = sample["load_std_roll24"]
        sample["load_pmax_proxy_24"] = sample["load_pmax_prevday_samehour"]
        sample["load_pmax_proxy_168"] = sample["load_pmax_prevweek_samehour"]
        sample["load_mean_proxy_24"] = sample["load_mean_proxy_24"].fillna(sample["load_mean_roll24"]).fillna(sample["load_mean"])
        sample["load_mean_proxy_168"] = sample["load_mean_proxy_168"].fillna(sample["load_mean_proxy_24"])
        sample["load_mean_roll_proxy"] = sample["load_mean_roll_proxy"].fillna(sample["load_mean_proxy_24"])
        sample["load_std_roll_proxy"] = sample["load_std_roll_proxy"].fillna(sample["load_std_hat"]).fillna(0.0)
        sample["load_pmax_proxy_24"] = sample["load_pmax_proxy_24"].fillna(sample["load_pmax_roll24"]).fillna(sample["load_pmax_weighted"])
        sample["load_pmax_proxy_168"] = sample["load_pmax_proxy_168"].fillna(sample["load_pmax_proxy_24"])

        valid = (sample["target_time"] - sample["origin_time"]).dt.total_seconds().eq(horizon * 3600)
        sample = sample.loc[valid].copy()
        sample["trajectory_id"] = sample["BS"].astype(str) + "@" + sample["origin_time"].astype(str)
        complete_tracker.append(sample[["trajectory_id", "horizon"]])

        two_stage_cols = [
            "BS",
            "trajectory_id",
            "origin_time",
            "target_time",
            "target_hour",
            "target_day_of_week",
            "target_hour_sin",
            "target_hour_cos",
            "p_base",
            "pbase_source",
            "y_true",
            "energy_true",
            "horizon",
            "load_mean_hat",
            "load_pmax_hat",
            "load_std_hat",
            "D1_hat",
            "D2_hat",
            "D3_hat",
            "n_cells",
            "sum_pmax",
            "sum_antennas",
            "load_mean_roll24",
            "load_pmax_roll24",
            "load_std_roll24",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        historical_cols = [
            "BS",
            "trajectory_id",
            "origin_time",
            "target_time",
            "target_hour",
            "target_day_of_week",
            "target_hour_sin",
            "target_hour_cos",
            "p_base",
            "pbase_source",
            "y_true",
            "energy_true",
            "horizon",
            "load_mean_proxy_24",
            "load_mean_proxy_168",
            "load_mean_roll_proxy",
            "load_std_roll_proxy",
            "load_pmax_proxy_24",
            "load_pmax_proxy_168",
            "sum_pmax",
            "n_cells",
            "sum_antennas",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        for col in pb.S_COLS:
            two_stage_cols.extend([f"{col}_hat", f"I_{col.split('_', 1)[1]}_hat"])
            historical_cols.append(f"{col}_hour_prior")

        strategy_frames["two_stage_proxy"].append(sample[two_stage_cols].copy())
        strategy_frames["historical_proxy"].append(sample[historical_cols].copy())

    complete = pd.concat(complete_tracker, ignore_index=True)
    complete_24 = complete.groupby("trajectory_id")["horizon"].nunique().reset_index()
    complete_24 = complete_24[complete_24["horizon"] == 24].rename(columns={"horizon": "n_horizons"})
    datasets = {k: pd.concat(v, ignore_index=True) for k, v in strategy_frames.items()}
    return datasets, complete_24


def run_dayahead_models_weighted(
    pb,
    panel: pd.DataFrame,
    weights: Dict[str, List[float]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_datasets, complete_24 = build_dayahead_dataset_weighted(pb, panel, weights)
    pred_rows: List[pd.DataFrame] = []
    coef_rows: List[pd.DataFrame] = []

    for strategy, raw_dataset in raw_datasets.items():
        dataset = pb.prepare_dayahead_features(raw_dataset, strategy)
        for spec in pb.build_dayahead_model_specs(strategy):
            _, preds = pb.evaluate_cv(
                dataset,
                spec,
                group_col="BS",
                extra_cols=["BS", "trajectory_id", "origin_time", "target_time", "horizon"],
            )
            preds["task"] = "dayahead"
            preds["strategy"] = strategy
            preds["model"] = spec.name
            pred_rows.append(preds)

            fitted = pb.fit_full_model(dataset.dropna(subset=["y_true"]), spec)
            coef = pb.get_coefficients(fitted, spec)
            coef["task"] = "dayahead"
            coef["strategy"] = strategy
            coef["horizon"] = -1
            coef["model"] = spec.name
            coef_rows.append(coef)

    predictions = pd.concat(pred_rows, ignore_index=True)
    eps = float(getattr(pb, "EPS", 1e-9))
    horizon_metrics = (
        predictions.groupby(["strategy", "model", "horizon"])
        .apply(
            lambda g: pd.Series(
                {
                    "MAE": float(mean_absolute_error(g["energy_true"], g["energy_pred"])),
                    "RMSE": float(np.sqrt(mean_squared_error(g["energy_true"], g["energy_pred"]))),
                    "MAPE": float((np.abs(g["energy_true"] - g["energy_pred"]) / np.maximum(np.abs(g["energy_true"]), eps)).mean()),
                    "n_samples": int(len(g)),
                    "n_bs": int(g["BS"].nunique()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    strict = predictions.merge(complete_24[["trajectory_id"]], on="trajectory_id", how="inner")
    strict_metrics = (
        strict.groupby(["strategy", "model", "trajectory_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "MAPE": float((np.abs(g["energy_true"] - g["energy_pred"]) / np.maximum(np.abs(g["energy_true"]), eps)).mean()),
                    "peak_error": float(abs(g.loc[g["energy_true"].idxmax(), "energy_true"] - g.loc[g["energy_true"].idxmax(), "energy_pred"])),
                    "valley_error": float(abs(g.loc[g["energy_true"].idxmin(), "energy_true"] - g.loc[g["energy_true"].idxmin(), "energy_pred"])),
                }
            ),
            include_groups=False,
        )
        .groupby(["strategy", "model"])
        .agg(MAPE=("MAPE", "mean"), peak_error=("peak_error", "mean"), valley_error=("valley_error", "mean"), n_trajectories=("MAPE", "size"))
        .reset_index()
        .sort_values(["MAPE", "peak_error"])
    )

    return strict_metrics, horizon_metrics, predictions, pd.concat(coef_rows, ignore_index=True)


def apply_bs_filters(
    panel: pd.DataFrame,
    min_merged_obs: Optional[int],
    exclude_bs_csv: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "n_bs_before": int(panel["BS"].nunique()),
        "n_rows_before": int(len(panel)),
        "min_merged_obs_per_bs": min_merged_obs,
        "exclude_bs_csv": str(exclude_bs_csv) if exclude_bs_csv else None,
    }
    out = panel.copy()
    if min_merged_obs is not None:
        counts = out.groupby("BS").size()
        keep = counts[counts >= min_merged_obs].index
        out = out[out["BS"].isin(keep)].copy()
        meta["after_min_obs_n_bs"] = int(out["BS"].nunique())
        meta["after_min_obs_n_rows"] = int(len(out))
    else:
        meta["after_min_obs_n_bs"] = meta["n_bs_before"]
        meta["after_min_obs_n_rows"] = meta["n_rows_before"]

    if exclude_bs_csv is not None:
        bad = set(pd.read_csv(exclude_bs_csv)["BS"].astype(str))
        n_before_ex = len(out)
        out = out[~out["BS"].astype(str).isin(bad)].copy()
        meta["excluded_bs_count_from_file"] = len(bad)
        meta["n_rows_removed_by_exclude_csv"] = int(n_before_ex - len(out))
    else:
        meta["excluded_bs_count_from_file"] = 0
        meta["n_rows_removed_by_exclude_csv"] = 0

    meta["n_bs_after"] = int(out["BS"].nunique())
    meta["n_rows_after"] = int(len(out))
    return out, meta


def plot_load_vs_energy(panel: pd.DataFrame, out_dir: Path) -> None:
    sample = panel.sample(min(12000, len(panel)), random_state=42)
    plt.figure(figsize=(8, 6))
    plt.scatter(sample["load_pmax_weighted"], sample["Energy"], s=10, alpha=0.25)
    coeff = np.polyfit(sample["load_pmax_weighted"], sample["Energy"], deg=2)
    x = np.linspace(sample["load_pmax_weighted"].min(), sample["load_pmax_weighted"].max(), 200)
    plt.plot(x, coeff[0] * x**2 + coeff[1] * x + coeff[2], color="red", linewidth=2)
    plt.xlabel("load_pmax_weighted")
    plt.ylabel("Energy")
    plt.title("Load vs Energy（仅日前流程 panel）")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_load_vs_energy.png", dpi=180)
    plt.close()


def plot_es_mode_impact(es_effects: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(es_effects["mode"], es_effects["effect_on_dynamic_energy"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Active - Inactive Dynamic Energy")
    plt.title("ES模式影响")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_es_mode_impact.png", dpi=180)
    plt.close()


def plot_dayahead_trajectory(dayahead_predictions: pd.DataFrame, strict_metrics: pd.DataFrame, out_dir: Path) -> None:
    best = strict_metrics.sort_values("MAPE").iloc[0]
    sub = dayahead_predictions[
        (dayahead_predictions["strategy"] == best["strategy"]) & (dayahead_predictions["model"] == best["model"])
    ].copy()
    trajectory = sub.groupby("trajectory_id")["horizon"].nunique().sort_values(ascending=False).index[0]
    one = sub[sub["trajectory_id"] == trajectory].sort_values("horizon")
    
    # 保存轨迹详细信息
    trajectory_info = one.copy()
    trajectory_info.to_csv(out_dir / f"trajectory_details_{trajectory}.csv", index=False)
    
    # 保存轨迹摘要信息（包含硬件参数等关键信息）
    # 检查列是否存在
    summary_cols = ["BS", "origin_time", "target_time", "horizon", "p_base", "energy_true", "energy_pred"]
    hardware_cols = ["n_cells", "sum_pmax", "sum_antennas"]
    
    # 只添加存在的硬件列
    for col in hardware_cols:
        if col in one.columns:
            summary_cols.append(col)
    
    trajectory_summary = one[summary_cols].copy()
    trajectory_summary.to_csv(out_dir / f"trajectory_summary_{trajectory}.csv", index=False)
    
    plt.figure(figsize=(9, 4.5))
    plt.plot(one["horizon"], one["energy_true"], marker="o", label="日前真实")
    plt.plot(one["horizon"], one["energy_pred"], marker="s", label="日前预测")
    plt.xlabel("预测步长 / 小时")
    plt.ylabel("Energy")
    plt.title(f"日前轨迹示例 ({best['strategy']} + {best['model']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_dayahead_trajectory.png", dpi=180)
    plt.close()


def plot_prediction_vs_actual_dayahead_only(
    dayahead_predictions: pd.DataFrame, strict_metrics: pd.DataFrame, out_dir: Path
) -> None:
    strategies = sorted(dayahead_predictions["strategy"].unique())
    plt.figure(figsize=(10, 5))
    for i, strat in enumerate(strategies[:2]):
        sub_m = strict_metrics[strict_metrics["strategy"] == strat].sort_values("MAPE")
        if sub_m.empty:
            continue
        best = sub_m.iloc[0]
        sample = dayahead_predictions[
            (dayahead_predictions["strategy"] == strat) & (dayahead_predictions["model"] == best["model"])
        ]
        plt.subplot(1, 2, i + 1)
        plt.scatter(sample["energy_true"], sample["energy_pred"], alpha=0.35, s=12)
        lo = min(sample["energy_true"].min(), sample["energy_pred"].min())
        hi = max(sample["energy_true"].max(), sample["energy_pred"].max())
        plt.plot([lo, hi], [lo, hi], color="red")
        plt.title(f"日前 {strat}\n{best['model']} MAPE={best['MAPE']:.4f}")
        plt.xlabel("真实")
        plt.ylabel("预测")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_prediction_vs_actual.png", dpi=180)
    plt.close()


def plot_error_by_horizon_dayahead(dayahead_horizon_metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    for strat in sorted(dayahead_horizon_metrics["strategy"].unique()):
        sub = dayahead_horizon_metrics[dayahead_horizon_metrics["strategy"] == strat]
        best_per_h = sub.sort_values("MAE").groupby("horizon").head(1).sort_values("horizon")
        plt.plot(best_per_h["horizon"], best_per_h["MAE"], marker="o", label=f"{strat} 各步最佳 MAE")
    plt.xlabel("预测步长 / 小时")
    plt.ylabel("MAE")
    plt.title("日前：按步长 MAE（每步取该步 MAE 最小的模型）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_error_by_horizon.png", dpi=180)
    plt.close()


def build_report(
    pbase_meta: Dict[str, float],
    strict_metrics: pd.DataFrame,
    dayahead_horizon_metrics: pd.DataFrame,
    physics_checks: pd.DataFrame,
    es_effects: pd.DataFrame,
    filter_meta: Dict[str, Any],
    out_dir: Path,
) -> None:
    best_day = strict_metrics.sort_values(["MAPE", "peak_error"]).iloc[0]
    monotonic_ratio = float(physics_checks.loc[physics_checks["check"] == "load_monotonicity_ratio", "value"].iloc[0])
    spearman = float(physics_checks.loc[physics_checks["check"] == "load_energy_spearman", "value"].iloc[0])
    pbase_ncells_corr = float(physics_checks.loc[physics_checks["check"] == "pbase_ncells_corr", "value"].iloc[0])
    top_es = es_effects.head(3)
    es_lines = "\n".join([f"- {row.mode}: {row.effect_on_dynamic_energy:.4f}" for row in top_es.itertuples(index=False)])
    best_horizon = dayahead_horizon_metrics.sort_values("MAE").iloc[0]

    filt = json.dumps(filter_meta, ensure_ascii=False, indent=2)

    report = f"""# 阶段B（仅日前）交付说明

## 1. 任务范围
- **仅日前预测**：每日 `00:00` 起报，预测未来 24 小时动态能耗分解项；策略 `two_stage_proxy` 与 `historical_proxy` 均保留。
- **不做日内滚动**；panel 不计算日内滞后特征（`energy_lag*`、`dynamic_lag*`、`load_*_lag1`、`S_*_lag1`）及 `load_sq` / `load_x_*`。
- `P_base` 仍来自阶段A：`{pbase_meta.get("chosen_window", "")} + {pbase_meta.get("chosen_method", "")}`。

## 2. BS 过滤（与稀疏 QC 对齐）
```json
{filt}
```

## 3. 关键输出文件
- `panel_dataset.csv`：瘦 panel。
- `dayahead_metrics.csv`、`dayahead_horizon_metrics.csv`、`dayahead_predictions.csv`、`model_coefficients.csv`（仅日前）。
- `physics_checks.csv`、`es_mode_effects.csv`、`pbase_complete.csv`。
- 图：`fig_load_vs_energy.png`、`fig_es_mode_impact.png`、`fig_dayahead_trajectory.png`、`fig_prediction_vs_actual.png`、`fig_error_by_horizon.png`。

## 4. 结果摘要
- 最佳日前（按完整 24h 轨迹 MAPE）：`{best_day['strategy']}` + `{best_day['model']}`，MAPE={best_day['MAPE']:.4f}。
- 负载单调分箱比例 {monotonic_ratio:.2%}，Spearman={spearman:.4f}；`p_base` 与 `n_cells` 相关 {pbase_ncells_corr:.4f}。
- ES 模式均值差（Top 3）：  
{es_lines}
- 各步 MAE 最优单步：h={int(best_horizon['horizon'])}，`{best_horizon['strategy']}` + `{best_horizon['model']}`。

## 5. 输出目录
本报告与 CSV/图位于同一目录：`{out_dir.name}/`。
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    _configure_windows_utf8_console()
    parser = argparse.ArgumentParser(description="Phase B 仅日前 + 可选 BS 过滤")
    parser.add_argument(
        "--min-merged-obs-per-bs",
        type=int,
        default=None,
        help="每站在 merged panel 中至少保留的观测行数；默认不过滤",
    )
    parser.add_argument(
        "--exclude-bs-csv",
        type=Path,
        default=None,
        help="含 BS 列的 CSV，列出需从训练中剔除的基站（如 bs_below_hour_threshold.csv）",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="指定输出子目录名（相对 phaseB_dayahead_only_20260415/），用于保留多次运行结果，如 outputs_filter_learned",
    )
    parser.add_argument(
        "--learn-proxy-weights",
        action="store_true",
        help="学习多源代理权重（w>=0,sum=1）并用于 two_stage_proxy 的 *_hat 构造；权重保存到输出目录",
    )
    parser.add_argument(
        "--proxy-weights-json",
        type=Path,
        default=None,
        help="读取已有 proxy_weights.json 并用于 *_hat 构造（与 --learn-proxy-weights 互斥）",
    )
    args = parser.parse_args()

    if args.output_subdir:
        out_dir = BASE_DIR / args.output_subdir
    else:
        use_filtered_output = (args.min_merged_obs_per_bs is not None) or (args.exclude_bs_csv is not None)
        out_dir = BASE_DIR / "outputs_filter" if use_filtered_output else BASE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pb = _load_phaseb_rebuild()

    panel, pbase_full, pbase_meta = build_dayahead_panel(pb)
    panel, filter_meta = apply_bs_filters(panel, args.min_merged_obs_per_bs, args.exclude_bs_csv)
    if panel.empty:
        raise SystemExit("过滤后 panel 为空，请放宽条件或检查 CSV。")

    if args.learn_proxy_weights and args.proxy_weights_json is not None:
        raise SystemExit("--learn-proxy-weights 与 --proxy-weights-json 不能同时使用。")

    proxy_weights: Optional[Dict[str, List[float]]] = None
    proxy_meta: Optional[Dict[str, Any]] = None
    if args.learn_proxy_weights:
        from proxy_weight_learning.learn_proxy_weights import learn_global_proxy_weights, save_weights

        learned, meta = learn_global_proxy_weights(panel, horizons=pb.DAYAHEAD_HORIZONS, s_cols=list(pb.S_COLS))
        proxy_weights = dict(learned.__dict__)
        proxy_meta = meta
        save_weights(out_dir, learned, meta)
    elif args.proxy_weights_json is not None:
        proxy_weights = json.loads(args.proxy_weights_json.read_text(encoding="utf-8"))

    filter_meta["output_subdir"] = out_dir.name
    filter_meta["proxy_weight_mode"] = (
        "learned" if args.learn_proxy_weights else ("loaded" if args.proxy_weights_json is not None else "fixed")
    )
    filter_meta["proxy_weights_json"] = str(args.proxy_weights_json) if args.proxy_weights_json is not None else None
    (out_dir / "filter_meta.json").write_text(
        json.dumps(filter_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if proxy_weights is None:
        strict_metrics, dayahead_horizon_metrics, dayahead_predictions, dayahead_coefficients = pb.run_dayahead_models(panel)
    else:
        strict_metrics, dayahead_horizon_metrics, dayahead_predictions, dayahead_coefficients = run_dayahead_models_weighted(
            pb, panel, proxy_weights
        )
    physics_checks, es_effects = pb.run_physics_checks(panel, pbase_full)

    plot_load_vs_energy(panel, out_dir)
    plot_es_mode_impact(es_effects, out_dir)
    plot_dayahead_trajectory(dayahead_predictions, strict_metrics, out_dir)
    plot_prediction_vs_actual_dayahead_only(dayahead_predictions, strict_metrics, out_dir)
    plot_error_by_horizon_dayahead(dayahead_horizon_metrics, out_dir)
    build_report(pbase_meta, strict_metrics, dayahead_horizon_metrics, physics_checks, es_effects, filter_meta, out_dir)

    pbase_full.to_csv(out_dir / "pbase_complete.csv", index=False)
    panel.to_csv(out_dir / "panel_dataset.csv", index=False)
    dayahead_coefficients.sort_values(["strategy", "model", "feature"]).to_csv(
        out_dir / "model_coefficients.csv", index=False
    )
    strict_metrics.to_csv(out_dir / "dayahead_metrics.csv", index=False)
    dayahead_horizon_metrics.sort_values(["horizon", "MAE"]).to_csv(out_dir / "dayahead_horizon_metrics.csv", index=False)
    dayahead_predictions.sort_values(["strategy", "model", "trajectory_id", "horizon"]).to_csv(
        out_dir / "dayahead_predictions.csv", index=False
    )
    physics_checks.to_csv(out_dir / "physics_checks.csv", index=False)
    es_effects.to_csv(out_dir / "es_mode_effects.csv", index=False)

    print("Phase B（仅日前）完成。输出目录：", out_dir.resolve())


if __name__ == "__main__":
    main()
