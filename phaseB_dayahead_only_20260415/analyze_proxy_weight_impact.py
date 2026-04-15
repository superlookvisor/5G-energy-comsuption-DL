"""
对比固定权重 vs 学习权重的影响（可视化 + 汇总表）。

示例：
  python phaseB_dayahead_only_20260415/analyze_proxy_weight_impact.py ^
    --fixed-dir phaseB_dayahead_only_20260415/outputs_filter_fixed ^
    --learned-dir phaseB_dayahead_only_20260415/outputs_filter_learned ^
    --out-dir phaseB_dayahead_only_20260415/outputs_proxy_impact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _strict_compare(fixed: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    key = ["strategy", "model"]
    cols = ["MAPE", "peak_error", "valley_error", "n_trajectories"]
    a = fixed[key + cols].rename(columns={c: f"{c}_fixed" for c in cols})
    b = learned[key + cols].rename(columns={c: f"{c}_learned" for c in cols})
    m = a.merge(b, on=key, how="outer")
    m["MAPE_delta"] = m["MAPE_learned"] - m["MAPE_fixed"]
    m["peak_error_delta"] = m["peak_error_learned"] - m["peak_error_fixed"]
    m["valley_error_delta"] = m["valley_error_learned"] - m["valley_error_fixed"]
    return m.sort_values(["MAPE_delta", "strategy", "model"])


def _horizon_compare(fixed: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    key = ["strategy", "model", "horizon"]
    cols = ["MAE", "RMSE", "MAPE", "n_samples", "n_bs"]
    a = fixed[key + cols].rename(columns={c: f"{c}_fixed" for c in cols})
    b = learned[key + cols].rename(columns={c: f"{c}_learned" for c in cols})
    m = a.merge(b, on=key, how="outer")
    for c in ["MAE", "RMSE", "MAPE"]:
        m[f"{c}_delta"] = m[f"{c}_learned"] - m[f"{c}_fixed"]
    return m.sort_values(["strategy", "model", "horizon"])


def _pred_compare(fixed: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    key = ["strategy", "model", "trajectory_id", "horizon"]
    keep = key + ["energy_true", "energy_pred", "y_true", "y_pred"]
    a = fixed[keep].rename(columns={c: f"{c}_fixed" for c in ["energy_pred", "y_pred"]})
    b = learned[keep].rename(columns={c: f"{c}_learned" for c in ["energy_pred", "y_pred"]})
    # energy_true/y_true 理论上应一致；以 fixed 为准
    b = b.drop(columns=["energy_true", "y_true"])
    m = a.merge(b, on=key, how="inner")
    m["energy_pred_delta"] = m["energy_pred_learned"] - m["energy_pred_fixed"]
    m["y_pred_delta"] = m["y_pred_learned"] - m["y_pred_fixed"]
    m["abs_err_fixed"] = np.abs(m["energy_true"] - m["energy_pred_fixed"])
    m["abs_err_learned"] = np.abs(m["energy_true"] - m["energy_pred_learned"])
    m["abs_err_delta"] = m["abs_err_learned"] - m["abs_err_fixed"]
    return m


def plot_strict_bar(best_row: pd.Series, out_path: Path) -> None:
    labels = ["MAPE", "peak_error", "valley_error"]
    fixed_vals = [best_row["MAPE_fixed"], best_row["peak_error_fixed"], best_row["valley_error_fixed"]]
    learned_vals = [best_row["MAPE_learned"], best_row["peak_error_learned"], best_row["valley_error_learned"]]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - w / 2, fixed_vals, width=w, label="fixed")
    plt.bar(x + w / 2, learned_vals, width=w, label="learned")
    plt.xticks(x, labels)
    plt.title(f"严格24h最佳模型指标对比\n{best_row['strategy']} + {best_row['model']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_delta_mae_by_horizon(hcmp: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    for strat in sorted(hcmp["strategy"].dropna().unique()):
        sub = hcmp[hcmp["strategy"] == strat].copy()
        # 每个 horizon 取该 horizon 下 MAE_fixed 最好的模型（和你原图口径接近）
        sub = sub.dropna(subset=["MAE_fixed", "MAE_learned"])
        best_fixed = sub.sort_values("MAE_fixed").groupby("horizon").head(1).sort_values("horizon")
        if best_fixed.empty:
            continue
        plt.plot(best_fixed["horizon"], best_fixed["MAE_delta"], marker="o", label=f"{strat}（按fixed每步最佳模型）")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("预测步长 / 小时")
    plt.ylabel("ΔMAE (learned - fixed)")
    plt.title("按预测步长的 MAE 差值曲线（learned - fixed）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_energy_pred_delta_hist(pcmp: pd.DataFrame, out_path: Path) -> None:
    x = pcmp["energy_pred_delta"].dropna().to_numpy(dtype=float)
    plt.figure(figsize=(9, 4.5))
    plt.hist(x, bins=60, alpha=0.85)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("ΔEnergy_pred (learned - fixed)")
    plt.ylabel("Count")
    plt.title("预测值差异分布（同一 trajectory_id×horizon 对齐）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="对比 fixed vs learned 权重的影响")
    p.add_argument("--fixed-dir", type=Path, required=True)
    p.add_argument("--learned-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.out_dir
    _ensure_dir(out_dir)

    strict_fixed = _read_csv(args.fixed_dir / "dayahead_metrics.csv")
    strict_learned = _read_csv(args.learned_dir / "dayahead_metrics.csv")
    strict_cmp = _strict_compare(strict_fixed, strict_learned)
    strict_cmp.to_csv(out_dir / "compare_strict_metrics.csv", index=False)

    horizon_fixed = _read_csv(args.fixed_dir / "dayahead_horizon_metrics.csv")
    horizon_learned = _read_csv(args.learned_dir / "dayahead_horizon_metrics.csv")
    horizon_cmp = _horizon_compare(horizon_fixed, horizon_learned)
    horizon_cmp.to_csv(out_dir / "compare_horizon_metrics.csv", index=False)

    preds_fixed = _read_csv(args.fixed_dir / "dayahead_predictions.csv")
    preds_learned = _read_csv(args.learned_dir / "dayahead_predictions.csv")
    pred_cmp = _pred_compare(preds_fixed, preds_learned)
    pred_cmp.to_csv(out_dir / "compare_predictions_aligned.csv", index=False)

    # best strict model by fixed MAPE (same selection rule)
    best_fixed = strict_fixed.sort_values("MAPE").iloc[0]
    best_row = strict_cmp[(strict_cmp["strategy"] == best_fixed["strategy"]) & (strict_cmp["model"] == best_fixed["model"])].iloc[0]

    plot_strict_bar(best_row, out_dir / "fig_strict_best_metrics_compare.png")
    plot_delta_mae_by_horizon(horizon_cmp, out_dir / "fig_delta_mae_by_horizon.png")
    plot_energy_pred_delta_hist(pred_cmp, out_dir / "fig_energy_pred_delta_hist.png")

    # record weights if available
    w_fixed = _read_json(args.fixed_dir / "proxy_weights.json")
    w_learned = _read_json(args.learned_dir / "proxy_weights.json")
    meta = {
        "fixed_dir": str(args.fixed_dir),
        "learned_dir": str(args.learned_dir),
        "out_dir": str(out_dir),
        "fixed_has_weights": bool(w_fixed),
        "learned_has_weights": bool(w_learned),
        "best_fixed_strategy": str(best_fixed["strategy"]),
        "best_fixed_model": str(best_fixed["model"]),
        "best_fixed_mape": float(best_fixed["MAPE"]),
        "n_aligned_predictions": int(len(pred_cmp)),
    }
    (out_dir / "compare_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("对比完成。输出目录：", out_dir.resolve())


if __name__ == "__main__":
    main()

