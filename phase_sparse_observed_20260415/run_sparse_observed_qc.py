"""
稀疏观测（策略 A）质检：复用 Phase B 清洗与聚合，输出 EC / merged 间隔统计与少样本站列表。

运行（在 energy_model_anp 目录下）:
  python phase_sparse_observed_20260415/run_sparse_observed_qc.py
  python phase_sparse_observed_20260415/run_sparse_observed_qc.py --plot-bs B_0 --plot-start 2023-01-01 --plot-end 2023-01-07
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PHASEB_SCRIPT = Path(__file__).resolve().parents[1] / "phaseB_dynamic_energy_20260414_rebuild" / "run_phaseB_dynamic_energy.py"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_phaseb_rebuild():
    name = "phaseb_rebuild_sparse_qc"
    spec = importlib.util.spec_from_file_location(name, PHASEB_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Python 3.10+：dataclass 解析注解时需要模块已在 sys.modules 中
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gap_stats(times: pd.Series) -> Dict[str, Any]:
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    n = int(len(t))
    if n == 0:
        return {
            "n_obs": 0,
            "time_min": pd.NaT,
            "time_max": pd.NaT,
            "gap_median_h": np.nan,
            "gap_p90_h": np.nan,
            "gap_max_h": np.nan,
            "prop_step_eq_1h": np.nan,
            "ideal_hour_slots": 0,
            "obs_to_ideal_hour_ratio": np.nan,
        }
    t_min, t_max = t.iloc[0], t.iloc[-1]
    delta = t.diff().dropna()
    if len(delta) == 0:
        gap_h = np.array([], dtype=float)
        prop_eq = np.nan
        gap_med = gap_p90 = gap_max = np.nan
    else:
        gap_h = delta.dt.total_seconds().to_numpy(dtype=float) / 3600.0
        prop_eq = float(np.mean(np.isclose(gap_h, 1.0, rtol=0.0, atol=1e-6)))
        gap_med = float(np.nanmedian(gap_h))
        gap_p90 = float(np.nanpercentile(gap_h, 90))
        gap_max = float(np.nanmax(gap_h))
    ideal = pd.date_range(t_min.floor("h"), t_max.ceil("h"), freq="h")
    n_ideal = max(1, len(ideal))
    ratio = float(n / n_ideal)
    return {
        "n_obs": n,
        "time_min": t_min,
        "time_max": t_max,
        "gap_median_h": gap_med,
        "gap_p90_h": gap_p90,
        "gap_max_h": gap_max,
        "prop_step_eq_1h": prop_eq,
        "ideal_hour_slots": int(n_ideal),
        "obs_to_ideal_hour_ratio": ratio,
    }


def _ec_by_bs(ec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bs, sub in ec.groupby("BS", sort=False):
        st = _gap_stats(sub["Time"])
        rows.append({"BS": bs, **st})
    return pd.DataFrame(rows)


def _merged_by_bs(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bs, sub in merged.groupby("BS", sort=False):
        st = _gap_stats(sub["Time"])
        rows.append({"BS": bs, **st})
    return pd.DataFrame(rows)


def _plot_bs_energy_sparse(
    merged: pd.DataFrame,
    bs: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_path: Path,
) -> None:
    sub = merged.loc[(merged["BS"] == bs) & (merged["Time"] >= start) & (merged["Time"] < end)].sort_values("Time")
    if sub.empty:
        raise ValueError(f"merged 中无数据: BS={bs}, [{start}, {end})")

    times = sub["Time"].to_numpy()
    energy = sub["Energy"].to_numpy(dtype=float)
    t_list: list[pd.Timestamp | float] = []
    e_list: list[float] = []
    for i in range(len(times)):
        if i > 0:
            dt_h = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 3600.0
            if dt_h > 1.0 + 1e-6:
                t_list.append(float("nan"))
                e_list.append(float("nan"))
        t_list.append(pd.Timestamp(times[i]))
        e_list.append(float(energy[i]))

    x_num: list[float] = []
    for t in t_list:
        if isinstance(t, float) and np.isnan(t):
            x_num.append(float("nan"))
        else:
            x_num.append(float(mdates.date2num(pd.Timestamp(t))))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x_num, e_list, marker="o", markersize=3, linewidth=1)
    ax.xaxis_date()
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title(f"稀疏观测（断线处为 >1h 间隔，未插补） BS={bs}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="策略 A：稀疏观测质检（不插补 Energy）")
    parser.add_argument("--hour-threshold", type=int, default=24, help="低于该 merged 观测数的 BS 写入 bs_below_hour_threshold.csv")
    parser.add_argument("--plot-bs", type=str, default=None, help="若指定则输出该站稀疏能耗曲线图")
    parser.add_argument("--plot-start", type=str, default=None, help="作图区间起点 ISO 日期，默认取该站 merged 最早时间")
    parser.add_argument("--plot-end", type=str, default=None, help="作图区间终点（开区间），默认取该站 merged 最晚时间+1天")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = _load_phaseb_rebuild()
    ec, cl, bsinfo = mod.load_raw_data()
    dynamic = mod.aggregate_cell_to_bs_time(cl, bsinfo)
    merged = ec.merge(dynamic, on=["BS", "Time"], how="inner")
    merged = merged.dropna(subset=["Energy", "load_pmax_weighted", "load_mean"]).copy()

    ec_cov = _ec_by_bs(ec)
    merged_cov = _merged_by_bs(merged)

    ec_cov.to_csv(OUTPUT_DIR / "ec_coverage_by_bs.csv", index=False)
    merged_cov.to_csv(OUTPUT_DIR / "merged_gap_by_bs.csv", index=False)

    below = merged_cov.loc[merged_cov["n_obs"] < args.hour_threshold, ["BS", "n_obs"]].sort_values("n_obs")
    below.to_csv(OUTPUT_DIR / "bs_below_hour_threshold.csv", index=False)

    all_gaps = []
    for _, sub in merged.groupby("BS", sort=False):
        t = sub["Time"].sort_values()
        d = t.diff().dropna()
        if len(d):
            all_gaps.extend((d.dt.total_seconds() / 3600.0).tolist())

    summary: Dict[str, Any] = {
        "policy": "sparse_observed_A_no_energy_imputation",
        "phaseb_loader_script": str(PHASEB_SCRIPT),
        "hour_threshold": int(args.hour_threshold),
        "n_bs_ec": int(ec["BS"].nunique()),
        "n_bs_merged": int(merged["BS"].nunique()),
        "n_rows_ec": int(len(ec)),
        "n_rows_merged": int(len(merged)),
        "merged_global_gap_median_h": float(np.median(all_gaps)) if all_gaps else None,
        "merged_global_gap_p90_h": float(np.percentile(all_gaps, 90)) if all_gaps else None,
        "merged_global_gap_max_h": float(np.max(all_gaps)) if all_gaps else None,
        "n_bs_below_threshold": int((merged_cov["n_obs"] < args.hour_threshold).sum()),
        "note_dayahead_row_shift": (
            "日前任务中 load_mean_prevday_samehour 等仍按 panel 行位移；若 panel 非整点连续小时，"
            "语义不是严格日历「昨日同时刻」。日内样本已用 make_continuity_mask 校验整小时间隔。"
        ),
    }
    (OUTPUT_DIR / "global_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "run_summary.csv", index=False)

    if args.plot_bs:
        bs = args.plot_bs
        sub_m = merged.loc[merged["BS"] == bs]
        if sub_m.empty:
            raise SystemExit(f"merged 中不存在 BS={bs!r}")
        t0 = pd.Timestamp(args.plot_start) if args.plot_start else sub_m["Time"].min()
        t1 = pd.Timestamp(args.plot_end) if args.plot_end else sub_m["Time"].max() + pd.Timedelta(days=1)
        tag = f"{t0.date()}_{t1.date()}"
        png = OUTPUT_DIR / f"fig_sparse_energy_{bs}_{tag}.png"
        _plot_bs_energy_sparse(merged, bs, t0, t1, png)

    print("稀疏观测质检完成。输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
