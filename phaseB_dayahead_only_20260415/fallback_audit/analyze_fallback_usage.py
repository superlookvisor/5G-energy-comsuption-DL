from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DAYAHEAD_HORIZONS_DEFAULT = list(range(1, 25))


def _pick_primary_source(prevday: pd.Series, roll24: pd.Series, current: pd.Series) -> pd.Series:
    """
    Return per-row primary source label following the fallback chain:
    prevday -> roll24 -> current -> zero.
    """
    out = pd.Series(np.full(len(current), "zero", dtype=object), index=current.index)
    out[current.notna()] = "current"
    out[roll24.notna()] = "roll24"
    out[prevday.notna()] = "prevday"
    return out


def _normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for col in ["origin_time", "target_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _required_columns_present(panel: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = [
        "Time",
        "BS",
        "hour",
        "load_mean",
        "load_pmax_weighted",
        "load_std",
        "load_mean_roll24",
        "load_pmax_roll24",
        "load_std_roll24",
    ]
    missing = [c for c in required if c not in panel.columns]
    return (len(missing) == 0), missing


def compute_fallback_audit(panel: pd.DataFrame, horizons: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct the day-ahead sampling logic enough to measure fallback usage
    for prevday/roll24/current chains.

    Returns:
      - per_row: one row per (trajectory_id, horizon) with primary sources for each variable
      - summary: aggregated ratios by horizon
    """
    panel = panel.copy()
    panel = _normalize_time_columns(panel)
    panel = panel.sort_values(["BS", "Time"])

    origins = panel[panel["hour"] == 0].copy()
    g = panel.groupby("BS", group_keys=False)
    panel_idx = panel.set_index(["BS", "Time"], drop=False)

    rows: List[pd.DataFrame] = []
    for horizon in horizons:
        sample = origins[["BS", "Time", "load_mean", "load_pmax_weighted", "load_std", "load_mean_roll24", "load_pmax_roll24", "load_std_roll24"]].copy()
        sample = sample.rename(columns={"Time": "origin_time"})
        sample["horizon"] = int(horizon)

        # Time-aligned prevday_samehour (matches the main pipeline logic):
        # target_time := origin_time + horizon hours (by per-BS time shift),
        # prevday_samehour := value at (BS, target_time - 24h).
        sample["target_time"] = g["Time"].shift(-int(horizon)).reindex(origins.index).to_numpy()
        prevday_time = pd.to_datetime(sample["target_time"], errors="coerce") - pd.Timedelta(hours=24)
        prevday_key = pd.MultiIndex.from_arrays([sample["BS"], prevday_time])
        sample["load_mean_prevday_samehour"] = panel_idx["load_mean"].reindex(prevday_key).to_numpy()
        sample["load_pmax_prevday_samehour"] = panel_idx["load_pmax_weighted"].reindex(prevday_key).to_numpy()
        sample["load_std_prevday_samehour"] = panel_idx["load_std"].reindex(prevday_key).to_numpy()

        sample["trajectory_id"] = sample["BS"].astype(str) + "@" + sample["origin_time"].dt.strftime("%Y-%m-%d")

        # Primary source labels (what the fallback chain effectively used)
        sample["src_load_mean"] = _pick_primary_source(
            sample["load_mean_prevday_samehour"], sample["load_mean_roll24"], sample["load_mean"]
        )
        sample["src_load_pmax"] = _pick_primary_source(
            sample["load_pmax_prevday_samehour"], sample["load_pmax_roll24"], sample["load_pmax_weighted"]
        )
        sample["src_load_std"] = _pick_primary_source(
            sample["load_std_prevday_samehour"], sample["load_std_roll24"], sample["load_std"]
        )

        # Extra: whether prevday was missing and thus fell back
        sample["fallback_prevday_to_roll24_load_mean"] = sample["load_mean_prevday_samehour"].isna() & sample["load_mean_roll24"].notna()
        sample["fallback_prevday_to_roll24_load_pmax"] = sample["load_pmax_prevday_samehour"].isna() & sample["load_pmax_roll24"].notna()
        sample["fallback_prevday_to_roll24_load_std"] = sample["load_std_prevday_samehour"].isna() & sample["load_std_roll24"].notna()
        sample["fallback_roll24_to_current_load_mean"] = sample["load_mean_prevday_samehour"].isna() & sample["load_mean_roll24"].isna() & sample["load_mean"].notna()
        sample["fallback_roll24_to_current_load_pmax"] = sample["load_pmax_prevday_samehour"].isna() & sample["load_pmax_roll24"].isna() & sample["load_pmax_weighted"].notna()
        sample["fallback_roll24_to_current_load_std"] = sample["load_std_prevday_samehour"].isna() & sample["load_std_roll24"].isna() & sample["load_std"].notna()

        rows.append(
            sample[
                [
                    "BS",
                    "trajectory_id",
                    "origin_time",
                    "horizon",
                    "src_load_mean",
                    "src_load_pmax",
                    "src_load_std",
                    "fallback_prevday_to_roll24_load_mean",
                    "fallback_prevday_to_roll24_load_pmax",
                    "fallback_prevday_to_roll24_load_std",
                    "fallback_roll24_to_current_load_mean",
                    "fallback_roll24_to_current_load_pmax",
                    "fallback_roll24_to_current_load_std",
                ]
            ].copy()
        )

    per_row = pd.concat(rows, ignore_index=True)

    def _src_ratio(frame: pd.DataFrame, col: str) -> Dict[str, float]:
        vc = frame[col].value_counts(dropna=False)
        denom = float(len(frame)) if len(frame) else 1.0
        return {k: float(v) / denom for k, v in vc.items()}

    summary_rows: List[Dict[str, object]] = []
    for horizon, frame in per_row.groupby("horizon", sort=True):
        rec: Dict[str, object] = {"horizon": int(horizon), "n": int(len(frame))}
        for var, col in [("load_mean", "src_load_mean"), ("load_pmax", "src_load_pmax"), ("load_std", "src_load_std")]:
            ratios = _src_ratio(frame, col)
            for src in ["prevday", "roll24", "current", "zero"]:
                rec[f"{var}_ratio_{src}"] = float(ratios.get(src, 0.0))
        # fallback-trigger ratios
        for bcol in [
            "fallback_prevday_to_roll24_load_mean",
            "fallback_prevday_to_roll24_load_pmax",
            "fallback_prevday_to_roll24_load_std",
            "fallback_roll24_to_current_load_mean",
            "fallback_roll24_to_current_load_pmax",
            "fallback_roll24_to_current_load_std",
        ]:
            rec[f"{bcol}_ratio"] = float(frame[bcol].mean()) if len(frame) else 0.0
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows).sort_values("horizon").reset_index(drop=True)
    return per_row, summary


def write_report(summary: pd.DataFrame, out_md: Path) -> None:
    def _to_md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"

        def _fmt(x: object) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)

        rows = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.itertuples(index=False, name=None)]
        return "\n".join([header, sep, *rows])

    lines: List[str] = []
    lines.append("# Fallback audit（prevday / roll24 / current）\n")
    lines.append("本报告统计日前特征构造中 `prevday_samehour -> roll24 -> current -> zero` 回退链的实际触发比例。\n")
    lines.append("## 关键读法\n")
    lines.append("- `*_ratio_prevday/roll24/current/zero`：某个变量在该 horizon 上，最终主要落到哪一档（按链优先级）。\n")
    lines.append("- `fallback_prevday_to_roll24_*_ratio`：prevday 缺失但 roll24 存在的比例。\n")
    lines.append("- `fallback_roll24_to_current_*_ratio`：prevday 和 roll24 都缺失但 current 存在的比例。\n\n")

    show_cols = [
        "horizon",
        "n",
        "load_mean_ratio_prevday",
        "load_mean_ratio_roll24",
        "load_mean_ratio_current",
        "load_pmax_ratio_prevday",
        "load_pmax_ratio_roll24",
        "load_pmax_ratio_current",
        "load_std_ratio_prevday",
        "load_std_ratio_roll24",
        "load_std_ratio_current",
    ]
    tbl = summary[show_cols].copy()
    lines.append("## 按 horizon 汇总（主要来源占比）\n")
    lines.append(_to_md_table(tbl))
    lines.append("\n\n")

    fb_cols = [
        "horizon",
        "fallback_prevday_to_roll24_load_mean_ratio",
        "fallback_prevday_to_roll24_load_pmax_ratio",
        "fallback_prevday_to_roll24_load_std_ratio",
        "fallback_roll24_to_current_load_mean_ratio",
        "fallback_roll24_to_current_load_pmax_ratio",
        "fallback_roll24_to_current_load_std_ratio",
    ]
    lines.append("## 回退触发比例（按 horizon）\n")
    lines.append(_to_md_table(summary[fb_cols].copy()))
    lines.append("\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panel",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "panel_dataset.csv"),
        help="Path to panel_dataset.csv (must include roll24/current columns).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs_fallback_audit"),
        help="Output directory for audit artifacts.",
    )
    ap.add_argument(
        "--horizons",
        type=str,
        default="1-24",
        help='Comma list like "1,3,6" or range like "1-24".',
    )
    args = ap.parse_args()

    panel_path = Path(args.panel)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "-" in args.horizons:
        a, b = args.horizons.split("-", 1)
        horizons = list(range(int(a), int(b) + 1))
    else:
        horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        horizons = DAYAHEAD_HORIZONS_DEFAULT

    panel = pd.read_csv(panel_path)
    ok, missing = _required_columns_present(panel)
    if not ok:
        raise SystemExit(f"panel缺少必要列: {missing}")

    per_row, summary = compute_fallback_audit(panel, horizons=horizons)

    per_row.to_csv(outdir / "fallback_per_row.csv", index=False)
    summary.to_csv(outdir / "fallback_summary_by_horizon.csv", index=False)

    by_bs = (
        per_row.groupby(["BS", "horizon"], as_index=False)
        .agg(
            n=("trajectory_id", "size"),
            load_mean_prevday_ratio=("src_load_mean", lambda s: float((s == "prevday").mean())),
            load_mean_roll24_ratio=("src_load_mean", lambda s: float((s == "roll24").mean())),
            load_mean_current_ratio=("src_load_mean", lambda s: float((s == "current").mean())),
            load_pmax_prevday_ratio=("src_load_pmax", lambda s: float((s == "prevday").mean())),
            load_pmax_roll24_ratio=("src_load_pmax", lambda s: float((s == "roll24").mean())),
            load_pmax_current_ratio=("src_load_pmax", lambda s: float((s == "current").mean())),
            load_std_prevday_ratio=("src_load_std", lambda s: float((s == "prevday").mean())),
            load_std_roll24_ratio=("src_load_std", lambda s: float((s == "roll24").mean())),
            load_std_current_ratio=("src_load_std", lambda s: float((s == "current").mean())),
            fb_prev_to_roll24_lm=("fallback_prevday_to_roll24_load_mean", "mean"),
            fb_prev_to_roll24_lp=("fallback_prevday_to_roll24_load_pmax", "mean"),
            fb_prev_to_roll24_ls=("fallback_prevday_to_roll24_load_std", "mean"),
            fb_roll_to_cur_lm=("fallback_roll24_to_current_load_mean", "mean"),
            fb_roll_to_cur_lp=("fallback_roll24_to_current_load_pmax", "mean"),
            fb_roll_to_cur_ls=("fallback_roll24_to_current_load_std", "mean"),
        )
        .sort_values(["horizon", "BS"])
    )
    by_bs.to_csv(outdir / "fallback_by_bs_and_horizon.csv", index=False)

    report_path = outdir / "analysis_report.md"
    write_report(summary, report_path)
    print(f"[ok] wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

