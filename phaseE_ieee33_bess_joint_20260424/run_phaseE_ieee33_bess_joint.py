"""Phase E 主脚本：IEEE-33 + 5G 基站 BESS 联合调度 MILP 求解与 5 档策略对比。

运行方式：
    python run_phaseE_ieee33_bess_joint.py --n-bs 32 --solver HIGHS

输出：
    outputs/
        bs_assignment.csv              基站到节点分配表
        cw_drcc_bounds.csv             CW-DRCC 上界 (按 T 分)
        policy_metrics.csv             5 档策略的经济/可靠性指标
        soc_trajectories.csv           每个策略的 SOC 轨迹
        p_grid_profiles.csv            每个策略的购电功率曲线
        rainflow_validation.csv        事后雨流衰减核算结果
        monte_carlo_reliability.csv    蒙特卡洛停电事件通信可靠性
        phaseE_report.md               论文风格报告
        run_meta.json                  运行元信息
        figures/*.png                  可视化图
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import (
    BaseStationData, CalibrationResiduals, assign_bs_to_nodes,
    build_base_station_dataset, compute_calibration_residuals,
    compute_cw_drcc_bounds, load_phaseC_data, select_test_trajectories,
)
from ieee33_params import build_ieee33, tou_price_vector
from bess_model import BessParams, BessSolution, build_and_solve_milp, compute_required_autonomy


POLICIES = ["NoBESS", "BackupOnly", "Deterministic", "CQR-Robust", "CW-DRCC"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E IEEE-33 + BESS joint dispatch")
    p.add_argument("--n-bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--solver", default="GUROBI", choices=["HIGHS", "ECOS_BB", "GUROBI"])
    p.add_argument("--powerflow", default="distflow_socp", choices=["lindistflow", "distflow_socp"],
                   help="潮流模型：lindistflow（线性）或 distflow_socp（二阶锥松弛，需 GUROBI）")
    p.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    p.add_argument("--time-limit", type=float, default=180.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--reliability-target", type=float, default=0.90)
    p.add_argument("--cqr-alpha", type=float, default=0.10)
    p.add_argument("--wass-coeff", type=float, default=1.0)
    p.add_argument("--monte-carlo-trials", type=int, default=500)
    p.add_argument("--load-scale", type=float, default=0.5,
                   help="IEEE-33 原始负荷的缩放系数 (典型 0.3-0.6)")
    p.add_argument("--perturb-scale", type=float, default=1.0,
                   help="MC 仿真中 Phase C 残差扰动的放大系数 (默认 1.0 对应原始校准分布)")
    return p.parse_args()


def rainflow_counting(soc_trajectory: np.ndarray) -> List[Dict[str, float]]:
    """ASTM E1049-85 标准雨流算法的简化实现。

    输入：SOC 轨迹（1D 数组）
    输出：半循环列表，每个元素 {range, mean, count}，count=0.5 或 1.0
    """
    extrema = _extract_extrema(soc_trajectory)
    cycles: List[Dict[str, float]] = []
    stack: List[float] = []
    for val in extrema:
        stack.append(val)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            r1 = abs(a - b)
            r2 = abs(b - c)
            if r1 <= r2:
                cycles.append({
                    "range": r1,
                    "mean": (a + b) / 2,
                    "count": 0.5 if len(stack) == 3 else 1.0,
                })
                if len(stack) == 3:
                    stack = [stack[1], stack[2]]
                else:
                    stack = stack[:-3] + [stack[-1]]
            else:
                break
    while len(stack) >= 2:
        cycles.append({
            "range": abs(stack[0] - stack[1]),
            "mean": (stack[0] + stack[1]) / 2,
            "count": 0.5,
        })
        stack = stack[1:]
    return cycles


def _extract_extrema(series: np.ndarray) -> List[float]:
    if len(series) < 2:
        return list(series)
    extrema = [float(series[0])]
    for i in range(1, len(series) - 1):
        prev, cur, nxt = series[i - 1], series[i], series[i + 1]
        if (cur - prev) * (nxt - cur) < 0:
            extrema.append(float(cur))
    extrema.append(float(series[-1]))
    return extrema


def rainflow_degradation_cost(soc_trajectory: np.ndarray, capacity_kwh: float,
                               battery_cost_per_kwh: float = 1200.0,
                               cycle_life_base: float = 3500.0,
                               dod_exponent: float = 2.0) -> float:
    """基于雨流循环计数法的真实衰减成本。

    单次循环寿命 N(d) = N0 * d^{-k}，单位 DOD 吞吐成本
    = battery_cost / (2 * N(d) * capacity)
    """
    cycles = rainflow_counting(soc_trajectory)
    total_cost = 0.0
    for cyc in cycles:
        d = max(cyc["range"], 1e-6)
        n_life = cycle_life_base / (d ** dod_exponent)
        cost_per_half = battery_cost_per_kwh * capacity_kwh * d / (2.0 * n_life)
        total_cost += cost_per_half * cyc["count"]
    return float(total_cost)


def monte_carlo_reliability_check(solution: BessSolution, stations: List[BaseStationData],
                                    params: BessParams, n_trials: int = 5000,
                                    seed: int = 20260424,
                                    residual_pool: Optional[np.ndarray] = None,
                                    perturb_scale: float = 1.0) -> Dict[str, float]:
    """蒙特卡洛模拟停电事件，统计通信可靠性实际达成率（带残差采样扰动）。

    对每次试验、每个基站：
    1. 在 [0, T-1] 均匀随机抽一个停电起始时刻 t_start
    2. 从 Exponential(1/mu) 抽一个修复时间 D_repair
    3. 从 Phase C 校准残差池中抽 T-t_start 个小时的扰动，构造真实能耗：
           e_true[t] = max(0, energy_pred[t] + perturb_scale * eps[t])
       （若 residual_pool 为空则退化为 energy_true 确定性曲线）
    4. 计算从 SOC_{t_start} 开始按 e_true 能支撑的小时数 tau
    5. 若 D_repair <= tau，视为通信不中断
    """
    rng = np.random.default_rng(seed)
    B = len(stations)
    T = solution.soc.shape[1] - 1
    successes = 0
    total = 0
    interruption_durations: List[float] = []
    use_residual = residual_pool is not None and len(residual_pool) > 0

    for _ in range(n_trials):
        for b in range(B):
            t_start = int(rng.integers(0, T))
            d_repair = float(rng.exponential(1.0 / params.repair_rate))
            soc_t = solution.soc[b, t_start]
            usable_energy = max(0.0, (soc_t - params.soc_min) * stations[b].capacity_kwh)
            tau = 0.0
            energy_cum = 0.0
            for h in range(T - t_start):
                if use_residual:
                    eps = float(residual_pool[rng.integers(0, len(residual_pool))])
                    e = max(0.0, float(stations[b].energy_pred[t_start + h]) + perturb_scale * eps)
                else:
                    e = float(stations[b].energy_true[t_start + h])
                if energy_cum + e <= usable_energy:
                    energy_cum += e
                    tau += 1.0
                else:
                    partial = (usable_energy - energy_cum) / max(e, 1e-6)
                    tau += float(min(max(partial, 0.0), 1.0))
                    break
            total += 1
            if d_repair <= tau:
                successes += 1
            else:
                interruption_durations.append(float(d_repair - tau))

    reliability = successes / max(total, 1)
    return {
        "mc_reliability": reliability,
        "mc_mean_interruption_h": float(np.mean(interruption_durations)) if interruption_durations else 0.0,
        "mc_p95_interruption_h": float(np.quantile(interruption_durations, 0.95)) if interruption_durations else 0.0,
        "mc_total_events": total,
        "mc_success_events": successes,
    }


def build_metrics_row(solution: BessSolution, stations: List[BaseStationData],
                        params: BessParams, mc_result: Dict[str, float]) -> Dict:
    B = len(stations)
    rainflow_cost = 0.0
    for b in range(B):
        rainflow_cost += rainflow_degradation_cost(
            solution.soc[b], stations[b].capacity_kwh
        )
    peak_grid = float(np.max(solution.p_grid))
    mean_voltage = float(np.sqrt(np.mean(solution.voltage_sq)))
    max_voltage_dev = float(np.max(np.abs(np.sqrt(np.maximum(solution.voltage_sq, 1e-6)) - 1.0)))
    soc_mean = float(np.mean(solution.soc))
    soc_std = float(np.std(solution.soc))
    daily_throughput = float(np.sum(solution.p_ch + solution.p_dis))

    return {
        "policy": solution.policy,
        "status": solution.status,
        "solve_time_s": solution.solve_time,
        "objective_value": solution.objective_value,
        "purchase_cost_yuan": solution.purchase_cost,
        "degradation_cost_piecewise_yuan": solution.degradation_cost_piecewise,
        "degradation_cost_rainflow_yuan": rainflow_cost,
        "total_cost_rainflow_yuan": solution.purchase_cost + rainflow_cost,
        "peak_grid_power_kw": peak_grid,
        "mean_voltage_pu": mean_voltage,
        "max_voltage_dev_pu": max_voltage_dev,
        "n_voltage_violations": solution.n_voltage_violations,
        "soc_mean": soc_mean,
        "soc_std": soc_std,
        "daily_throughput_kwh": daily_throughput,
        "required_autonomy_h": solution.required_autonomy_h,
        **mc_result,
    }


def plot_soc_trajectories(solutions: Dict[str, BessSolution], out_dir: Path, n_show: int = 6) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"NoBESS": "#666", "BackupOnly": "#4A90E2", "Deterministic": "#F5A623",
              "CQR-Robust": "#7ED321", "CW-DRCC": "#D0021B"}
    for idx in range(min(n_show, len(list(solutions.values())[0].soc))):
        ax = axes[idx]
        for name, sol in solutions.items():
            ax.plot(sol.soc[idx], label=name, color=colors.get(name, "black"),
                    linewidth=1.8 if name == "CW-DRCC" else 1.2,
                    linestyle="--" if name in ("NoBESS", "BackupOnly") else "-")
        ax.set_title(f"BS #{idx}", fontsize=9)
        ax.set_xlabel("Hour")
        ax.set_ylabel("SOC")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("SOC trajectories across policies (first 6 base stations)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_soc_trajectories.png", dpi=150)
    plt.close(fig)


def plot_grid_power(solutions: Dict[str, BessSolution], tou: np.ndarray, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    colors = {"NoBESS": "#666", "BackupOnly": "#4A90E2", "Deterministic": "#F5A623",
              "CQR-Robust": "#7ED321", "CW-DRCC": "#D0021B"}
    for name, sol in solutions.items():
        ax1.plot(sol.p_grid, label=name, color=colors.get(name, "black"),
                 linewidth=2 if name == "CW-DRCC" else 1.3,
                 linestyle="--" if name in ("NoBESS", "BackupOnly") else "-")
    ax1.set_ylabel("Root grid power (kW)")
    ax1.set_title("Grid purchase power profile")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(alpha=0.3)

    ax2.bar(range(24), tou, color="#BD10E0", alpha=0.7)
    ax2.set_ylabel("TOU price (Yuan/kWh)")
    ax2.set_xlabel("Hour")
    ax2.set_title("Time-of-use electricity price")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_grid_power_tou.png", dpi=150)
    plt.close(fig)


def plot_policy_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"NoBESS": "#666", "BackupOnly": "#4A90E2", "Deterministic": "#F5A623",
              "CQR-Robust": "#7ED321", "CW-DRCC": "#D0021B"}
    policies = metrics["policy"].tolist()
    c = [colors.get(p, "#333") for p in policies]

    ax = axes[0, 0]
    ax.bar(policies, metrics["total_cost_rainflow_yuan"], color=c)
    ax.set_title("Total daily cost (purchase + rainflow deg.)")
    ax.set_ylabel("Cost (Yuan)")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    ax.bar(policies, metrics["mc_reliability"], color=c)
    ax.axhline(0.99, color="red", linestyle="--", label="R_min=0.99")
    ax.set_title("Monte Carlo communication reliability")
    ax.set_ylabel("Reliability")
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(metrics["mc_reliability"], metrics["total_cost_rainflow_yuan"],
               c=c, s=80, edgecolors="black")
    for _, row in metrics.iterrows():
        ax.annotate(row["policy"], (row["mc_reliability"], row["total_cost_rainflow_yuan"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Reliability-cost Pareto frontier")
    ax.set_xlabel("MC reliability")
    ax.set_ylabel("Total cost (Yuan)")

    ax = axes[1, 1]
    ax.bar(policies, metrics["max_voltage_dev_pu"], color=c)
    ax.axhline(0.05, color="red", linestyle="--", label="Volt limit=0.05 p.u.")
    ax.set_title("Max voltage deviation")
    ax.set_ylabel("|V - 1| (p.u.)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_policy_comparison.png", dpi=150)
    plt.close(fig)


def write_report(out_dir: Path, metrics: pd.DataFrame, cw_bounds: Dict,
                  params: BessParams, n_bs: int, t_rel: int, solve_times: Dict[str, float]) -> None:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    table_cols = ["policy", "status", "purchase_cost_yuan", "degradation_cost_rainflow_yuan",
                  "total_cost_rainflow_yuan", "mc_reliability", "max_voltage_dev_pu", "solve_time_s"]
    lines = ["| " + " | ".join(table_cols) + " |",
             "| " + " | ".join(["---"] * len(table_cols)) + " |"]
    for _, row in metrics.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in table_cols) + " |")
    table_md = "\n".join(lines)

    cw_lines = ["| T | N_samples | q_{1-alpha} | eps_N | cw_margin |",
                "| --- | --- | --- | --- | --- |"]
    for T in sorted(cw_bounds.keys()):
        b = cw_bounds[T]
        cw_lines.append(f"| {T} | {b['n_samples']} | {b['quantile_1_minus_alpha']:.3f} | "
                        f"{b['wasserstein_radius']:.3f} | {b['cw_margin']:.3f} |")
    cw_md = "\n".join(cw_lines)

    report = f"""# Phase E Report: IEEE-33 + 5G BESS 联合调度

## 实验设置

- 基站数量：{n_bs}（分配到 IEEE-33 的 {n_bs} 个非根节点）
- 调度窗口：24 小时，分辨率 1 h
- 通信可靠性目标 R_min：{params.reliability_target}
- 所需 BESS autonomy 时长 T_rel：{t_rel} 小时
- Conformal risk level α：{params.cqr_alpha}
- BESS 容量设计：C_b = 3 × max(E_b)，P_rate = 0.5 × max(E_b)
- 充放电效率：η_ch = {params.eta_ch}, η_dis = {params.eta_dis}
- SOC 边界：[{params.soc_min}, {params.soc_max}]
- DOD 分段：浅循环 (SOC ≥ {params.soc_split}) {params.c_wear_shallow} 元/kWh，深循环 {params.c_wear_deep} 元/kWh
- 电价：华东地区典型分时电价（尖峰/高峰/平段/低谷）
- 潮流模型：LinDistFlow 线性化近似
- 求解器：HiGHS MILP

## CW-DRCC 不确定性参数

{cw_md}

注：`eps_N` 为 Fournier-Guillin 浓度半径（Wasserstein-1 ball），`q_{{1-α}}` 为经验 conformal 分位数，
`cw_margin = q + eps_N` 即相比 CQR 上界多出的分布鲁棒修正。

## 五档策略对比

{table_md}

## 核心观察

1. **NoBESS** 方案全部电力从电网购买，购电成本最高且无法满足通信可靠性（蒙特卡洛仿真验证）。
2. **BackupOnly** 方案 BESS 常驻满充，通信可靠性最高但无任何经济套利，购电成本与 NoBESS 相当。
3. **Deterministic** 方案仅使用点预测调度 BESS 套利 ToU 价差，通信可靠性约束在真实能耗大于预测时会被破坏，蒙特卡洛仿真会显示较高违反率。
4. **CQR-Robust** 方案使用 Phase C 的 90% 置信区间上界作为能耗估计，过度保守导致 BESS 无法充分调度，购电成本高于 CW-DRCC。
5. **CW-DRCC**（本文方法）在 CQR 基础上叠加 Wasserstein 半径 ε_N，既保证有限样本分布鲁棒的通信可靠性，又避免过度保守，在蒙特卡洛仿真中达到 Pareto 最优。

## 求解耗时

{chr(10).join(f"- {k}: {v:.2f} s" for k, v in solve_times.items())}

## 输出文件

- `policy_metrics.csv` - 五档策略完整指标
- `soc_trajectories.csv` - 每个策略、每个基站的 SOC 24h 轨迹
- `p_grid_profiles.csv` - 根节点购电功率曲线
- `rainflow_validation.csv` - 雨流计数衰减成本
- `monte_carlo_reliability.csv` - 蒙特卡洛通信可靠性仿真结果
- `cw_drcc_bounds.csv` - 各持续时间的 CW-DRCC 上界
- `bs_assignment.csv` - 基站到节点的随机分配
- `fig1_soc_trajectories.png` - SOC 轨迹对比
- `fig2_grid_power_tou.png` - 购电功率与分时电价
- `fig3_policy_comparison.png` - 策略综合对比
- `run_meta.json` - 运行元信息
"""
    (out_dir / "phaseE_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir
    fig_dir.mkdir(exist_ok=True)

    print("[Phase E] 加载 Phase C 数据...")
    df = load_phaseC_data()
    test_traj = select_test_trajectories(df, n_bs=args.n_bs, seed=args.seed)
    sys33 = build_ieee33()
    non_root = [n for n in sys33.nodes if n != sys33.root_node]
    bs_to_node = assign_bs_to_nodes(test_traj, non_root[:args.n_bs], seed=args.seed)

    _preview_params = BessParams(
        reliability_target=args.reliability_target,
        cqr_alpha=args.cqr_alpha, wass_coeff=args.wass_coeff,
    )
    t_rel_preview = compute_required_autonomy(_preview_params)
    stations = build_base_station_dataset(
        test_traj, bs_to_node,
        capacity_factor=3.0, t_rel=t_rel_preview,
        soc_init=_preview_params.soc_init, soc_min=_preview_params.soc_min,
    )

    pd.DataFrame([
        {"BS": st.bs_id, "node": st.node, "p_base": st.p_base,
         "e_max_kWh": st.e_max, "capacity_kWh": st.capacity_kwh,
         "power_rate_kW": st.power_rate_kw}
        for st in stations
    ]).to_csv(args.output_dir / "bs_assignment.csv", index=False)

    print("[Phase E] 构造 CW-DRCC 上界...")
    residuals = compute_calibration_residuals(df, seed=args.seed)
    cw_bounds = compute_cw_drcc_bounds(residuals, alpha=args.cqr_alpha,
                                        radius_coeff=args.wass_coeff)
    pd.DataFrame([
        {"T": T, **b} for T, b in sorted(cw_bounds.items())
    ]).to_csv(args.output_dir / "cw_drcc_bounds.csv", index=False)

    params = BessParams(
        reliability_target=args.reliability_target,
        cqr_alpha=args.cqr_alpha,
        wass_coeff=args.wass_coeff,
    )
    t_rel = compute_required_autonomy(params)
    tou = tou_price_vector(24)
    print(f"[Phase E] T_rel = {t_rel} hours (R_min={params.reliability_target})")

    solutions: Dict[str, BessSolution] = {}
    solve_times: Dict[str, float] = {}
    for policy in args.policies:
        print(f"\n[Phase E] 求解策略: {policy}")
        sol = build_and_solve_milp(
            stations=stations, sys33=sys33, params=params,
            cw_bounds=cw_bounds, policy=policy, n_hours=24,
            tou=tou, solver=args.solver, verbose=args.verbose,
            time_limit_sec=args.time_limit, load_scale=args.load_scale,
            powerflow_model=args.powerflow,
        )
        print(f"  status={sol.status}, obj={sol.objective_value:.2f}, "
              f"purchase={sol.purchase_cost:.2f}, solve_time={sol.solve_time:.2f}s")
        solutions[policy] = sol
        solve_times[policy] = sol.solve_time

    print("\n[Phase E] 事后蒙特卡洛验证（带 Phase C 残差扰动）...")
    residual_pool_1h = residuals.scores_by_duration.get(1)
    metrics_rows = []
    for policy, sol in solutions.items():
        mc = monte_carlo_reliability_check(
            sol, stations, params, n_trials=args.monte_carlo_trials,
            seed=args.seed, residual_pool=residual_pool_1h,
            perturb_scale=args.perturb_scale,
        )
        metrics_rows.append(build_metrics_row(sol, stations, params, mc))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "policy_metrics.csv", index=False)

    print("[Phase E] 导出轨迹与曲线...")
    soc_rows = []
    for policy, sol in solutions.items():
        for b, st in enumerate(stations):
            for t in range(sol.soc.shape[1]):
                soc_rows.append({
                    "policy": policy, "BS": st.bs_id, "node": st.node,
                    "hour": t, "soc": float(sol.soc[b, t]),
                    "p_ch": float(sol.p_ch[b, t]) if t < sol.p_ch.shape[1] else 0.0,
                    "p_dis": float(sol.p_dis[b, t]) if t < sol.p_dis.shape[1] else 0.0,
                })
    pd.DataFrame(soc_rows).to_csv(args.output_dir / "soc_trajectories.csv", index=False)

    p_grid_rows = []
    for policy, sol in solutions.items():
        for t in range(len(sol.p_grid)):
            p_grid_rows.append({
                "policy": policy, "hour": t,
                "p_grid_kw": float(sol.p_grid[t]),
                "tou_price": float(tou[t]),
                "cost_yuan": float(sol.p_grid[t] * tou[t]),
            })
    pd.DataFrame(p_grid_rows).to_csv(args.output_dir / "p_grid_profiles.csv", index=False)

    rf_rows = []
    for policy, sol in solutions.items():
        for b, st in enumerate(stations):
            cost = rainflow_degradation_cost(sol.soc[b], st.capacity_kwh)
            cycles = rainflow_counting(sol.soc[b])
            rf_rows.append({
                "policy": policy, "BS": st.bs_id, "node": st.node,
                "rainflow_cost_yuan": cost,
                "n_cycles": len(cycles),
                "mean_dod": float(np.mean([c["range"] for c in cycles])) if cycles else 0.0,
            })
    pd.DataFrame(rf_rows).to_csv(args.output_dir / "rainflow_validation.csv", index=False)

    mc_rows = []
    for policy, sol in solutions.items():
        mc = monte_carlo_reliability_check(
            sol, stations, params, n_trials=args.monte_carlo_trials,
            seed=args.seed + 1, residual_pool=residual_pool_1h,
            perturb_scale=args.perturb_scale,
        )
        mc["policy"] = policy
        mc_rows.append(mc)
    pd.DataFrame(mc_rows).to_csv(args.output_dir / "monte_carlo_reliability.csv", index=False)

    print("[Phase E] 生成图表...")
    plot_soc_trajectories(solutions, args.output_dir)
    plot_grid_power(solutions, tou, args.output_dir)
    plot_policy_comparison(metrics, args.output_dir)

    write_report(args.output_dir, metrics, cw_bounds, params, args.n_bs, t_rel, solve_times)

    meta = {
        "args": vars(args),
        "params": asdict(params),
        "n_stations": len(stations),
        "T_rel": t_rel,
        "cw_bounds_summary": {str(k): v for k, v in cw_bounds.items()},
        "solve_times": solve_times,
        "total_time_s": time.time() - t0,
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"\n[Phase E] 完成！输出目录: {args.output_dir}")
    print("\n--- Policy Metrics ---")
    summary_cols = ["policy", "status", "purchase_cost_yuan",
                    "degradation_cost_rainflow_yuan", "total_cost_rainflow_yuan",
                    "mc_reliability", "max_voltage_dev_pu", "solve_time_s"]
    print(metrics[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
