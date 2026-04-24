"""Method 1 (统一容量倍率扫描): sweep capacity-hours (χ) for Phase D.

This script reuses Phase D's core functions and produces:
- per-chi outputs in: capacity_sweep_method1/outputs/chi_XX/
- cross-chi summaries + comparison figures in: capacity_sweep_method1/outputs/
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASED_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASED_DIR.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _import_phase_d_module():
    # Import sibling file as a module (repo isn't necessarily packaged)
    import sys

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
    return out


def parse_args() -> argparse.Namespace:
    phase_d = _import_phase_d_module()

    p = argparse.ArgumentParser(description="Phase D Method-1 capacity-hours sweep (统一容量倍率扫描)")
    p.add_argument("--phasec-output-dir", type=Path, default=Path(phase_d.PHASEC_DEFAULT_OUTPUT_DIR))
    p.add_argument("--phaseb-strategy", default="two_stage_proxy")
    p.add_argument("--phaseb-model", default="RandomForest")
    p.add_argument("--evaluation-split", choices=["test", "calibration", "train", "all"], default="test")

    p.add_argument("--capacity-hours-list", type=str, default="4,6,8,10,12,14,16,18,20,24")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--skip-per-chi-figures", action="store_true", help="Skip per-chi figures for speed")

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

    p.add_argument("--violation-threshold", type=float, default=0.01, help="For χ★ selection")
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


def _policy_order(phase_d) -> List[str]:
    return list(phase_d.RELIABILITY_POLICIES.keys())


def _policy_suffix(phase_d, policy_name: str) -> str:
    return str(phase_d.RELIABILITY_POLICIES[policy_name])


def compute_mean_autonomy_by_policy(phase_d, decisions_eval: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for policy in _policy_order(phase_d):
        suffix = _policy_suffix(phase_d, policy)
        col = f"autonomy_time_true_{suffix}"
        if col not in decisions_eval.columns:
            out[policy] = float("nan")
            continue
        s = pd.to_numeric(decisions_eval[col], errors="coerce").dropna()
        out[policy] = float(s.mean()) if len(s) else float("nan")
    return out


def compute_chi_star(summary: pd.DataFrame, policy: str, violation_threshold: float) -> Optional[float]:
    sub = summary[(summary["policy"].astype(str) == str(policy))].copy()
    if sub.empty:
        return None
    sub["capacity_hours"] = pd.to_numeric(sub["capacity_hours"], errors="coerce")
    sub["reliability_violation_rate"] = pd.to_numeric(sub["reliability_violation_rate"], errors="coerce")
    sub = sub.dropna(subset=["capacity_hours", "reliability_violation_rate"]).sort_values("capacity_hours")
    ok = sub[sub["reliability_violation_rate"] <= float(violation_threshold)]
    if ok.empty:
        return None
    return float(ok.iloc[0]["capacity_hours"])


def plot_sweep_lines(
    output_dir: Path,
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    filename: str,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    summary = summary.copy()
    summary[x_col] = pd.to_numeric(summary[x_col], errors="coerce")
    summary[y_col] = pd.to_numeric(summary[y_col], errors="coerce")
    summary = summary.dropna(subset=[x_col, y_col])

    for policy, g in summary.groupby("policy"):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", linewidth=2, label=str(policy))

    if hline is not None:
        ax.axhline(float(hline), linestyle="--", linewidth=1.5, label=hline_label or "target")

    ax.set_title(title)
    ax.set_xlabel("capacity_hours (χ)")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_tradeoff(output_dir: Path, summary: pd.DataFrame, filename: str) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    df = summary.copy()
    for col in ["capacity_hours", "mean_comm_reliability", "mean_dispatchable_capacity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["capacity_hours", "mean_comm_reliability", "mean_dispatchable_capacity"])

    for policy, g in df.groupby("policy"):
        g = g.sort_values("capacity_hours")
        ax.plot(g["mean_comm_reliability"], g["mean_dispatchable_capacity"], marker="o", linewidth=1.5, label=str(policy))
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['capacity_hours'])}", (r["mean_comm_reliability"], r["mean_dispatchable_capacity"]), xytext=(4, 3), textcoords="offset points", fontsize=7)

    ax.set_title("Reliability vs dispatchable capacity (annotated by χ)")
    ax.set_xlabel("Mean communication reliability")
    ax.set_ylabel("Mean dispatchable capacity")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def write_capacity_sweep_report(
    output_dir: Path,
    summary: pd.DataFrame,
    reliability_target: float,
    violation_threshold: float,
    chi_star_full_backup: Optional[float],
    chi_star_ra_cqr: Optional[float],
) -> None:
    def fmt(x: object, digits: int = 4) -> str:
        if isinstance(x, (float, np.floating)):
            return "nan" if math.isnan(float(x)) else f"{float(x):.{digits}f}"
        return str(x)

    lines: List[str] = []
    lines.append("# 装机容量调整方法一：统一容量倍率扫描（Method 1）")
    lines.append("")
    lines.append("## 方法说明")
    lines.append("- 统一容量倍率 χ 扫描，并令 `capacity_hours=χ`，即每个基站容量满足 \(C_b(\\chi)=\\chi\\,p_b^{\\mathrm{base}}\)。")
    lines.append(f"- 本次扫描 χ 集合：`{','.join(str(int(x)) for x in sorted(summary['capacity_hours'].unique()))}`")
    lines.append(f"- 可靠性目标：`{reliability_target}`；用于 χ★ 的违约率阈值：`{violation_threshold}`")
    lines.append("")
    lines.append("## χ★（最小可行统一容量倍率）")
    lines.append(f"- `Full BESS Backup`：χ★ = `{fmt(chi_star_full_backup, 2)}`")
    lines.append(f"- `Reliability-Aware CQR`：χ★ = `{fmt(chi_star_ra_cqr, 2)}`")
    lines.append("")
    lines.append("## 指标汇总（按 χ 与策略）")
    cols = [
        "capacity_hours",
        "policy",
        "mean_comm_reliability",
        "reliability_violation_rate",
        "mean_dispatchable_capacity",
        "dispatchable_capacity_ratio",
        "capacity_infeasibility_rate",
        "mean_expected_interruption_duration",
        "mean_expected_unserved_traffic",
        "net_dispatch_value",
        "mean_autonomy_time_true",
        "n_decisions",
        "n_bs",
    ]
    show = summary[cols].copy()
    show = show.sort_values(["capacity_hours", "policy"])
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines.append("")

    (output_dir / "capacity_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    phase_d = _import_phase_d_module()
    args = parse_args()

    capacity_hours_list = parse_capacity_hours_list(args.capacity_hours_list)
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # One-time load of Phase B/C aligned hourly predictions
    hourly = phase_d.load_hourly_predictions(args.phasec_output_dir, args.phaseb_strategy, args.phaseb_model, args.traffic_column)

    summary_rows: List[Dict[str, object]] = []
    for chi in capacity_hours_list:
        chi_int = int(round(float(chi)))
        chi_dir = out_root / f"chi_{chi_int:02d}"
        chi_dir.mkdir(parents=True, exist_ok=True)

        c = phase_d.PhaseDConfig(
            phasec_output_dir=str(args.phasec_output_dir),
            output_dir=str(chi_dir),
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

        trajectories = phase_d.build_reliability_trajectory_dataset(hourly, args.max_support_hours, float(chi), args.soc, args.soc_min)
        q_by_duration = phase_d.calibrate_energy_quantiles_by_duration(trajectories, args.max_support_hours, args.energy_risk_epsilon)
        decisions, t_rel, t_rel_cont = phase_d.apply_reliability_policies(trajectories, q_by_duration, c)
        decisions_eval = phase_d.evaluation_frame(decisions, args.evaluation_split)

        metrics = phase_d.reliability_metric_rows(decisions_eval, c, t_rel, t_rel_cont)
        by_horizon = phase_d.reliability_by_horizon(decisions_eval)

        phase_d.write_outputs(chi_dir, hourly, trajectories, decisions, metrics, by_horizon, q_by_duration)
        if not bool(args.skip_per_chi_figures):
            phase_d.save_figures(chi_dir, decisions_eval, metrics, q_by_duration, c)
        phase_d.write_report(chi_dir, c, hourly, trajectories, decisions_eval, metrics, t_rel, t_rel_cont, q_by_duration)
        phase_d.write_meta(chi_dir, c, hourly, trajectories, decisions, q_by_duration, t_rel, t_rel_cont)

        autonomy_mean = compute_mean_autonomy_by_policy(phase_d, decisions_eval)
        for _, r in metrics.iterrows():
            policy = str(r["policy"])
            row = {k: r.get(k) for k in metrics.columns}
            row["capacity_hours"] = float(chi)
            row["mean_autonomy_time_true"] = float(autonomy_mean.get(policy, float("nan")))
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("Empty sweep summary; no metrics were produced.")

    # Write summaries
    summary_path = out_root / "capacity_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    pivot_rel = summary.pivot_table(index="capacity_hours", columns="policy", values="mean_comm_reliability", aggfunc="mean").reset_index()
    pivot_rel.to_csv(out_root / "capacity_sweep_summary_pivot_reliability.csv", index=False)

    pivot_disp = summary.pivot_table(index="capacity_hours", columns="policy", values="mean_dispatchable_capacity", aggfunc="mean").reset_index()
    pivot_disp.to_csv(out_root / "capacity_sweep_summary_pivot_dispatch.csv", index=False)

    # Cross-chi figures
    plot_sweep_lines(
        out_root,
        summary,
        x_col="capacity_hours",
        y_col="mean_comm_reliability",
        title="Mean communication reliability vs capacity_hours (χ)",
        y_label="Mean communication reliability",
        filename="fig_sweep_reliability_vs_chi.png",
        hline=float(args.reliability_target),
        hline_label=f"target={args.reliability_target}",
    )
    plot_sweep_lines(
        out_root,
        summary,
        x_col="capacity_hours",
        y_col="reliability_violation_rate",
        title="Reliability violation rate vs capacity_hours (χ)",
        y_label="Reliability violation rate",
        filename="fig_sweep_violation_vs_chi.png",
        hline=float(args.violation_threshold),
        hline_label=f"threshold={args.violation_threshold}",
    )
    plot_sweep_lines(
        out_root,
        summary,
        x_col="capacity_hours",
        y_col="mean_dispatchable_capacity",
        title="Mean dispatchable capacity vs capacity_hours (χ)",
        y_label="Mean dispatchable capacity",
        filename="fig_sweep_dispatch_vs_chi.png",
    )
    plot_sweep_lines(
        out_root,
        summary,
        x_col="capacity_hours",
        y_col="capacity_infeasibility_rate",
        title="Capacity infeasibility rate vs capacity_hours (χ)",
        y_label="Capacity infeasibility rate",
        filename="fig_sweep_infeasibility_vs_chi.png",
    )
    plot_sweep_lines(
        out_root,
        summary,
        x_col="capacity_hours",
        y_col="net_dispatch_value",
        title="Net dispatch value vs capacity_hours (χ)",
        y_label="Net dispatch value",
        filename="fig_sweep_net_value_vs_chi.png",
    )
    plot_tradeoff(out_root, summary, filename="fig_sweep_tradeoff_reliability_dispatch.png")

    # χ★ + report
    chi_star_full = compute_chi_star(summary, policy="Full BESS Backup", violation_threshold=float(args.violation_threshold))
    chi_star_ra = compute_chi_star(summary, policy="Reliability-Aware CQR", violation_threshold=float(args.violation_threshold))
    write_capacity_sweep_report(
        out_root,
        summary=summary,
        reliability_target=float(args.reliability_target),
        violation_threshold=float(args.violation_threshold),
        chi_star_full_backup=chi_star_full,
        chi_star_ra_cqr=chi_star_ra,
    )

    print(f"Method-1 capacity sweep finished. Outputs: {out_root.resolve()}")
    if chi_star_full is not None:
        print(f"chi* (Full BESS Backup) = {chi_star_full}")
    else:
        print("chi* (Full BESS Backup) not found under threshold")
    if chi_star_ra is not None:
        print(f"chi* (Reliability-Aware CQR) = {chi_star_ra}")
    else:
        print("chi* (Reliability-Aware CQR) not found under threshold")


if __name__ == "__main__":
    main()

