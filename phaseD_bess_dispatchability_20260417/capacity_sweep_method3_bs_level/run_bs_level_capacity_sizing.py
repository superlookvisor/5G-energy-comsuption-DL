"""Method 3 (基站级差异化容量配置): per-BS capacity sizing for Phase D.

For each BS b, find the minimum capacity multiplier χ_b such that the station's
own reliability violation rate does not exceed ε_bs. Compared to the uniform
method, the per-BS configuration exploits heterogeneity in p_base, load
dynamics and prediction uncertainty, and therefore typically reduces total
installed BESS capacity.

Outputs (in capacity_sweep_method3_bs_level/outputs/):
- bs_level_per_bs_metrics.csv
- bs_level_chi_star.csv
- bs_level_total_capacity_comparison.csv
- fig_chi_b_distribution.png
- fig_capacity_comparison.png
- fig_bs_chi_heatmap.png
- bs_level_report.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASED_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASED_DIR.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EPS = 1e-9


def _import_phase_d_module():
    if str(PHASED_DIR) not in sys.path:
        sys.path.insert(0, str(PHASED_DIR))
    import run_phaseD_bess_dispatchability as phase_d  # type: ignore

    return phase_d


def parse_capacity_hours_list(text: str) -> List[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError as e:
            raise ValueError(f"Invalid --capacity-hours-list value: {p!r}") from e
    if not out:
        raise ValueError("--capacity-hours-list is empty")
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    phase_d = _import_phase_d_module()

    p = argparse.ArgumentParser(description="Phase D Method-3 per-BS differentiated capacity sizing")
    p.add_argument("--phasec-output-dir", type=Path, default=Path(phase_d.PHASEC_DEFAULT_OUTPUT_DIR))
    p.add_argument("--phaseb-strategy", default="two_stage_proxy")
    p.add_argument("--phaseb-model", default="RandomForest")
    p.add_argument("--evaluation-split", choices=["test", "calibration", "train", "all"], default="test")

    p.add_argument("--capacity-hours-list", type=str, default="4,6,8,10,12,14,16,18,20,24")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--soc", type=float, default=1.0)
    p.add_argument("--soc-min", type=float, default=0.0)
    p.add_argument("--dispatch-price", type=float, default=1.0)
    p.add_argument("--interruption-penalty", type=float, default=10.0)
    p.add_argument("--unserved-traffic-penalty", type=float, default=1.0)

    p.add_argument("--repair-distribution", choices=["exponential", "weibull"], default="exponential")
    p.add_argument("--repair-rate", type=float, default=0.7)
    p.add_argument("--weibull-shape", type=float, default=1.5)
    p.add_argument("--weibull-scale", type=float, default=4.0)
    p.add_argument("--reliability-target", type=float, default=0.99)
    p.add_argument("--energy-risk-epsilon", type=float, default=0.10)
    p.add_argument("--outage-rate", type=float, default=0.01)
    p.add_argument("--max-support-hours", type=int, default=24)
    p.add_argument("--traffic-column", default=None)

    p.add_argument(
        "--bs-violation-threshold",
        type=float,
        default=0.01,
        help="Per-BS max allowed reliability violation rate ε_bs for χ_b★ selection",
    )
    p.add_argument(
        "--uniform-violation-threshold",
        type=float,
        default=0.01,
        help="Global violation threshold for uniform χ★ baseline (Method 1 logic)",
    )
    return p.parse_args()


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
        }
    )


def per_bs_policy_metrics(
    phase_d,
    decisions_eval: pd.DataFrame,
    reliability_target: float,
) -> pd.DataFrame:
    """Aggregate per-BS, per-policy metrics from decisions_eval.

    Only (BS, policy) pairs with at least one valid (C_disp not NaN) decision
    are returned; BS that are entirely horizon-infeasible at this χ have no
    evaluable reliability and are therefore excluded from per-BS sizing at
    this χ level (they may still be evaluable at smaller/larger χ because
    C_disp depends on usable energy, but the horizon filter does not).
    """
    rows: List[Dict[str, object]] = []
    policies = list(phase_d.RELIABILITY_POLICIES.items())
    bs_list = sorted(decisions_eval["BS"].unique().tolist())

    for bs in bs_list:
        sub = decisions_eval[decisions_eval["BS"] == bs]
        if sub.empty:
            continue
        p_base_bs = pd.to_numeric(sub["p_base"], errors="coerce").dropna()
        p_base_value = float(p_base_bs.median()) if not p_base_bs.empty else float("nan")

        for policy, suffix in policies:
            rel_col = f"comm_reliability_{suffix}"
            disp_col = f"C_disp_{suffix}"
            auto_col = f"autonomy_time_true_{suffix}"
            req_col = f"reliable_energy_requirement_{suffix}"

            valid = sub[sub[disp_col].notna()].copy()
            n_valid = int(len(valid))
            if n_valid == 0:
                continue  # skip BS with no evaluable decisions under this policy

            rel = pd.to_numeric(valid[rel_col], errors="coerce")
            disp = pd.to_numeric(valid[disp_col], errors="coerce")
            auto = pd.to_numeric(valid[auto_col], errors="coerce")
            req = pd.to_numeric(valid[req_col], errors="coerce")

            violation_rate = float(((rel + EPS) < reliability_target).mean())
            rows.append(
                {
                    "BS": bs,
                    "policy": policy,
                    "p_base": p_base_value,
                    "n_decisions": int(len(sub)),
                    "n_valid_decisions": n_valid,
                    "mean_comm_reliability": float(rel.mean()),
                    "violation_rate_bs": violation_rate,
                    "mean_dispatchable_capacity": float(disp.mean()),
                    "mean_autonomy_time_true": float(auto.mean()),
                    "mean_reliable_energy_requirement": float(req.mean()),
                }
            )
    return pd.DataFrame(rows)


def search_chi_star_per_bs(
    per_chi_per_bs: pd.DataFrame,
    threshold: float,
    chi_grid: List[float],
) -> pd.DataFrame:
    """For each (BS, policy), find the minimum chi with violation_rate_bs <= threshold."""
    rows: List[Dict[str, object]] = []
    for (bs, policy), grp in per_chi_per_bs.groupby(["BS", "policy"]):
        grp = grp.sort_values("capacity_hours")
        p_base_val = float(grp["p_base"].dropna().median()) if grp["p_base"].notna().any() else float("nan")

        ok = grp[
            grp["violation_rate_bs"].notna()
            & (grp["violation_rate_bs"].astype(float) <= float(threshold) + EPS)
        ]
        if ok.empty:
            chi_b = float("nan")
            feasible = False
            chi_b_capped = float(max(chi_grid))
        else:
            chi_b = float(ok.iloc[0]["capacity_hours"])
            feasible = True
            chi_b_capped = chi_b

        n_dec_values = grp["n_decisions"].dropna()
        n_decisions = int(n_dec_values.iloc[0]) if not n_dec_values.empty else 0

        rows.append(
            {
                "BS": bs,
                "policy": policy,
                "p_base": p_base_val,
                "n_decisions": n_decisions,
                "chi_b_star": chi_b,
                "chi_b_star_feasible": feasible,
                "chi_b_capped": chi_b_capped,
                "installed_capacity_bs_level": chi_b_capped * p_base_val if not math.isnan(p_base_val) else float("nan"),
                "mean_reliability_at_chi_b": (
                    float(ok.iloc[0]["mean_comm_reliability"]) if feasible else float("nan")
                ),
                "violation_at_chi_b": (
                    float(ok.iloc[0]["violation_rate_bs"]) if feasible else float("nan")
                ),
                "dispatchable_at_chi_b": (
                    float(ok.iloc[0]["mean_dispatchable_capacity"]) if feasible else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def search_uniform_chi_star(
    per_chi_per_bs: pd.DataFrame,
    threshold: float,
) -> Dict[str, Optional[float]]:
    """Uniform χ★: smallest χ such that global violation rate ≤ threshold (Method 1 style)."""
    out: Dict[str, Optional[float]] = {}
    for policy, grp in per_chi_per_bs.groupby("policy"):
        global_rows: List[Tuple[float, float]] = []
        for chi, g in grp.groupby("capacity_hours"):
            n = float(g["n_valid_decisions"].sum())
            if n <= 0:
                continue
            # weighted by n_valid_decisions to recover global violation fraction
            viol_counts = (g["violation_rate_bs"].astype(float) * g["n_valid_decisions"].astype(float)).sum()
            global_rows.append((float(chi), viol_counts / n))
        if not global_rows:
            out[str(policy)] = None
            continue
        global_rows.sort(key=lambda x: x[0])
        chosen: Optional[float] = None
        for chi, r in global_rows:
            if r <= float(threshold) + EPS:
                chosen = chi
                break
        out[str(policy)] = chosen
    return out


def build_capacity_comparison(
    chi_star_df: pd.DataFrame,
    uniform_chi_star: Dict[str, Optional[float]],
    chi_grid: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    chi_grid_max = float(max(chi_grid))
    for policy, grp in chi_star_df.groupby("policy"):
        n_bs = int(len(grp))
        n_feasible = int(grp["chi_b_star_feasible"].sum())
        p_base_sum = float(grp["p_base"].sum())

        uniform_chi = uniform_chi_star.get(str(policy))
        if uniform_chi is None:
            uniform_chi_used = float("nan")
            uniform_capacity = float("nan")
        else:
            uniform_chi_used = float(uniform_chi)
            uniform_capacity = float(uniform_chi) * p_base_sum

        uniform_capacity_at_grid_max = chi_grid_max * p_base_sum
        bs_level_capacity = float(grp["installed_capacity_bs_level"].sum())

        savings_pct = (
            float(1.0 - bs_level_capacity / uniform_capacity) * 100.0
            if uniform_capacity and uniform_capacity > 0 and not math.isnan(uniform_capacity)
            else float("nan")
        )
        savings_pct_vs_grid_max = (
            float(1.0 - bs_level_capacity / uniform_capacity_at_grid_max) * 100.0
            if uniform_capacity_at_grid_max and uniform_capacity_at_grid_max > 0
            else float("nan")
        )

        rows.append(
            {
                "policy": policy,
                "n_bs": n_bs,
                "n_bs_feasible": n_feasible,
                "n_bs_capped_at_grid_max": int(n_bs - n_feasible),
                "chi_grid_max": chi_grid_max,
                "uniform_chi_star": uniform_chi_used,
                "uniform_total_installed_capacity": uniform_capacity,
                "uniform_total_installed_capacity_at_grid_max": uniform_capacity_at_grid_max,
                "bs_level_total_installed_capacity": bs_level_capacity,
                "capacity_savings_vs_uniform_pct": savings_pct,
                "capacity_savings_vs_grid_max_pct": savings_pct_vs_grid_max,
                "mean_chi_b_star": (
                    float(grp.loc[grp["chi_b_star_feasible"], "chi_b_star"].mean())
                    if n_feasible > 0
                    else float("nan")
                ),
                "median_chi_b_star": (
                    float(grp.loc[grp["chi_b_star_feasible"], "chi_b_star"].median())
                    if n_feasible > 0
                    else float("nan")
                ),
                "min_chi_b_star": (
                    float(grp.loc[grp["chi_b_star_feasible"], "chi_b_star"].min())
                    if n_feasible > 0
                    else float("nan")
                ),
                "max_chi_b_star": (
                    float(grp.loc[grp["chi_b_star_feasible"], "chi_b_star"].max())
                    if n_feasible > 0
                    else float("nan")
                ),
                "sum_p_base": p_base_sum,
            }
        )
    return pd.DataFrame(rows)


def plot_chi_b_distribution(chi_star_df: pd.DataFrame, chi_grid: List[float], out_path: Path) -> None:
    setup_plot_style()
    policies = sorted(chi_star_df["policy"].unique().tolist())
    n = len(policies)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    bins = [x - 0.5 for x in chi_grid] + [max(chi_grid) + 1.5]
    for ax, policy in zip(axes, policies):
        g = chi_star_df[chi_star_df["policy"].astype(str) == str(policy)]
        vals_feasible = g.loc[g["chi_b_star_feasible"], "chi_b_star"].astype(float).to_numpy()
        vals_capped = g.loc[~g["chi_b_star_feasible"], "chi_b_capped"].astype(float).to_numpy()
        if len(vals_feasible):
            ax.hist(vals_feasible, bins=bins, alpha=0.85, label="feasible")
        if len(vals_capped):
            ax.hist(vals_capped, bins=bins, alpha=0.55, label="capped @ grid max")
        ax.set_title(str(policy), fontsize=9)
        ax.set_xlabel("χ_b★")
        if ax is axes[0]:
            ax.set_ylabel("# BS")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_capacity_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    df = comparison.copy().sort_values("policy").reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - width, df["uniform_total_installed_capacity"].astype(float), width, label="Uniform χ★")
    ax.bar(x, df["uniform_total_installed_capacity_at_grid_max"].astype(float), width, label="Uniform χ=grid max")
    ax.bar(x + width, df["bs_level_total_installed_capacity"].astype(float), width, label="Per-BS χ_b★")

    for i, row in df.iterrows():
        sp_u = row.get("capacity_savings_vs_uniform_pct")
        sp_g = row.get("capacity_savings_vs_grid_max_pct")
        txt_parts: List[str] = []
        if isinstance(sp_u, (int, float)) and not math.isnan(float(sp_u)):
            txt_parts.append(f"vs χ★: {float(sp_u):.1f}%")
        if isinstance(sp_g, (int, float)) and not math.isnan(float(sp_g)):
            txt_parts.append(f"vs max: {float(sp_g):.1f}%")
        if txt_parts:
            ys = [
                float(row.get("uniform_total_installed_capacity", np.nan) or np.nan),
                float(row.get("uniform_total_installed_capacity_at_grid_max", np.nan) or np.nan),
                float(row.get("bs_level_total_installed_capacity", np.nan) or np.nan),
            ]
            y_top = np.nanmax(ys)
            ax.annotate("; ".join(txt_parts), (i, y_top), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["policy"].astype(str).tolist(), rotation=20)
    ax.set_ylabel("Total installed BESS capacity")
    ax.set_title("Total installed capacity: uniform vs per-BS sizing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bs_chi_heatmap(chi_star_df: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    pivot = chi_star_df.pivot_table(
        index="BS",
        columns="policy",
        values="chi_b_capped",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(7.0, 0.4 * len(pivot.columns) + 5), max(4.0, 0.12 * len(pivot) + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str).tolist(), rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str).tolist(), fontsize=6)
    ax.set_title("Per-BS χ_b★ across policies (capped at grid max for infeasible)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("χ_b★")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    chi_grid: List[float],
    reliability_target: float,
    bs_violation_threshold: float,
    uniform_violation_threshold: float,
    uniform_chi_star: Dict[str, Optional[float]],
    comparison: pd.DataFrame,
    chi_star_df: pd.DataFrame,
) -> None:
    def fmt(v: object, digits: int = 4) -> str:
        if isinstance(v, (float, np.floating)):
            x = float(v)
            return "nan" if math.isnan(x) else f"{x:.{digits}f}"
        return str(v)

    lines: List[str] = []
    lines.append("# 装机容量调整方法三：基站级差异化容量配置（Method 3, Per-BS Sizing）")
    lines.append("")
    lines.append("## 方法说明")
    lines.append("- 为每座基站 b 独立选择最小容量倍率 χ_b，使得该基站自身的可靠性违约率不超过阈值 ε_bs。")
    lines.append("- 基站装机容量为 \\(C_b=\\chi_b\\cdot p_b^{\\mathrm{base}}\\)；总装机容量为 \\(\\sum_b \\chi_b\\cdot p_b^{\\mathrm{base}}\\)。")
    lines.append(f"- χ 网格：`{','.join(str(int(x)) for x in sorted(chi_grid))}`")
    lines.append(f"- 可靠性目标 R_min：`{reliability_target}`；基站违约阈值 ε_bs：`{bs_violation_threshold}`；全网违约阈值（基线）：`{uniform_violation_threshold}`")
    lines.append("")
    lines.append("## 基线：统一容量倍率 χ★（Method 1 逻辑）")
    for policy, chi in uniform_chi_star.items():
        lines.append(f"- `{policy}`：χ★ = `{fmt(chi, 2)}`")
    lines.append("")
    lines.append("## 总装机容量对比")
    cols = [
        "policy",
        "n_bs",
        "n_bs_feasible",
        "n_bs_capped_at_grid_max",
        "uniform_chi_star",
        "uniform_total_installed_capacity",
        "uniform_total_installed_capacity_at_grid_max",
        "bs_level_total_installed_capacity",
        "capacity_savings_vs_uniform_pct",
        "capacity_savings_vs_grid_max_pct",
        "mean_chi_b_star",
        "median_chi_b_star",
        "min_chi_b_star",
        "max_chi_b_star",
        "sum_p_base",
    ]
    show = comparison[cols].copy().sort_values("policy")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    lines.append("")

    lines.append("## 基站级 χ_b★ 统计（前 20 条，按策略→BS 排序）")
    lines.append("")
    lines.append("| policy | BS | p_base | chi_b_star | feasible | mean_reliability_at_chi_b | violation_at_chi_b | dispatchable_at_chi_b | installed_capacity_bs_level |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    head = chi_star_df.sort_values(["policy", "BS"]).head(20)
    for _, r in head.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["policy"]),
                    str(r["BS"]),
                    fmt(r["p_base"]),
                    fmt(r["chi_b_star"]),
                    str(bool(r["chi_b_star_feasible"])),
                    fmt(r["mean_reliability_at_chi_b"]),
                    fmt(r["violation_at_chi_b"]),
                    fmt(r["dispatchable_at_chi_b"]),
                    fmt(r["installed_capacity_bs_level"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("> 完整 χ_b★ 明细参见 `bs_level_chi_star.csv`；每 χ、每 BS 的原始指标参见 `bs_level_per_bs_metrics.csv`。")

    (output_dir / "bs_level_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    phase_d = _import_phase_d_module()
    args = parse_args()

    chi_grid = parse_capacity_hours_list(args.capacity_hours_list)
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Load Phase B/C aligned hourly predictions once
    hourly = phase_d.load_hourly_predictions(
        args.phasec_output_dir,
        args.phaseb_strategy,
        args.phaseb_model,
        args.traffic_column,
    )

    per_chi_per_bs_rows: List[pd.DataFrame] = []
    for chi in chi_grid:
        c = phase_d.PhaseDConfig(
            phasec_output_dir=str(args.phasec_output_dir),
            output_dir=str(out_root / f"chi_{int(round(chi)):02d}_decisions"),
            phaseb_strategy=args.phaseb_strategy,
            phaseb_model=args.phaseb_model,
            backup_duration=4,
            epsilon=0.10,
            capacity_hours=float(chi),
            soc=float(args.soc),
            soc_min=float(args.soc_min),
            dispatch_price=float(args.dispatch_price),
            shortfall_penalty=10.0,
            evaluation_split=str(args.evaluation_split),
            repair_distribution=str(args.repair_distribution),
            repair_rate=float(args.repair_rate),
            weibull_shape=float(args.weibull_shape),
            weibull_scale=float(args.weibull_scale),
            reliability_target=float(args.reliability_target),
            energy_risk_epsilon=float(args.energy_risk_epsilon),
            outage_rate=float(args.outage_rate),
            interruption_penalty=float(args.interruption_penalty),
            unserved_traffic_penalty=float(args.unserved_traffic_penalty),
            traffic_column=str(args.traffic_column) if args.traffic_column is not None else None,
            max_support_hours=int(args.max_support_hours),
        )
        phase_d.validate_config(c)

        trajectories = phase_d.build_reliability_trajectory_dataset(
            hourly, args.max_support_hours, float(chi), args.soc, args.soc_min
        )
        q_by_duration = phase_d.calibrate_energy_quantiles_by_duration(
            trajectories, args.max_support_hours, args.energy_risk_epsilon
        )
        decisions, _t_rel, _t_rel_cont = phase_d.apply_reliability_policies(trajectories, q_by_duration, c)
        decisions_eval = phase_d.evaluation_frame(decisions, args.evaluation_split)

        per_bs = per_bs_policy_metrics(phase_d, decisions_eval, float(args.reliability_target))
        per_bs.insert(0, "capacity_hours", float(chi))
        per_chi_per_bs_rows.append(per_bs)

    per_chi_per_bs = pd.concat(per_chi_per_bs_rows, ignore_index=True)
    per_chi_per_bs.to_csv(out_root / "bs_level_per_bs_metrics.csv", index=False)

    chi_star_df = search_chi_star_per_bs(
        per_chi_per_bs,
        threshold=float(args.bs_violation_threshold),
        chi_grid=chi_grid,
    )
    chi_star_df.to_csv(out_root / "bs_level_chi_star.csv", index=False)

    uniform_chi_star = search_uniform_chi_star(
        per_chi_per_bs,
        threshold=float(args.uniform_violation_threshold),
    )

    comparison = build_capacity_comparison(chi_star_df, uniform_chi_star, chi_grid)
    comparison.to_csv(out_root / "bs_level_total_capacity_comparison.csv", index=False)

    plot_chi_b_distribution(chi_star_df, chi_grid, out_root / "fig_chi_b_distribution.png")
    plot_capacity_comparison(comparison, out_root / "fig_capacity_comparison.png")
    plot_bs_chi_heatmap(chi_star_df, out_root / "fig_bs_chi_heatmap.png")

    write_report(
        out_root,
        chi_grid=chi_grid,
        reliability_target=float(args.reliability_target),
        bs_violation_threshold=float(args.bs_violation_threshold),
        uniform_violation_threshold=float(args.uniform_violation_threshold),
        uniform_chi_star=uniform_chi_star,
        comparison=comparison,
        chi_star_df=chi_star_df,
    )

    print(f"Method-3 per-BS capacity sizing finished. Outputs: {out_root.resolve()}")
    print("Per-policy total capacity (uniform vs per-BS):")
    for _, row in comparison.sort_values("policy").iterrows():
        print(
            f"  {row['policy']:<30s} "
            f"uniform_chi*={row['uniform_chi_star']:.2f}  "
            f"uniform_total={row['uniform_total_installed_capacity']:.2f}  "
            f"bs_total={row['bs_level_total_installed_capacity']:.2f}  "
            f"savings={row['capacity_savings_vs_uniform_pct']:.2f}%  "
            f"feasible={int(row['n_bs_feasible'])}/{int(row['n_bs'])}"
        )


if __name__ == "__main__":
    main()
