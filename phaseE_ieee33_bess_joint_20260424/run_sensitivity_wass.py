"""Phase E 敏感性扫描：Wasserstein 半径系数 C_epsilon。

CW-DRCC 的半径 epsilon_N = C_epsilon * sigma * N^{-1/3} * sqrt(log(1/alpha))
此系数体现"分布偏移允许程度"：C_epsilon 越大，鲁棒裕度越大，保守性越强。

本脚本扫描 C_epsilon ∈ {0.5, 1.0, 2.0, 3.0, 5.0}，固定 R_min=0.99，
展示 CW-DRCC 在经济性-鲁棒性权衡上独有的调节维度。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import (
    assign_bs_to_nodes, build_base_station_dataset,
    compute_calibration_residuals, compute_cw_drcc_bounds,
    load_phaseC_data, select_test_trajectories,
)
from ieee33_params import build_ieee33, tou_price_vector
from bess_model import BessParams, build_and_solve_milp, compute_required_autonomy
from run_phaseE_ieee33_bess_joint import (
    monte_carlo_reliability_check, rainflow_degradation_cost,
)


COEF_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]


def main() -> None:
    out_dir = BASE_DIR / "outputs_sensitivity_wass"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 20260424
    n_bs = 32
    load_scale = 0.5
    cqr_alpha = 0.10
    r_min = 0.99
    mc_trials = 500

    print("[Sens-Wass] 加载 Phase C 数据 & IEEE-33...")
    df = load_phaseC_data()
    test_traj = select_test_trajectories(df, n_bs=n_bs, seed=seed)
    sys33 = build_ieee33()
    non_root = [n for n in sys33.nodes if n != sys33.root_node]
    bs_to_node = assign_bs_to_nodes(test_traj, non_root[:n_bs], seed=seed)

    residuals = compute_calibration_residuals(df, seed=seed)
    params_preview = BessParams(reliability_target=r_min, cqr_alpha=cqr_alpha)
    t_rel = compute_required_autonomy(params_preview)
    stations = build_base_station_dataset(
        test_traj, bs_to_node, capacity_factor=3.0, t_rel=t_rel,
        soc_init=params_preview.soc_init, soc_min=params_preview.soc_min,
    )
    residual_pool_1h = residuals.scores_by_duration.get(1)
    tou = tou_price_vector(24)

    rows: List[Dict] = []
    for c_eps in COEF_GRID:
        cw_bounds = compute_cw_drcc_bounds(residuals, alpha=cqr_alpha, radius_coeff=c_eps)
        params = BessParams(reliability_target=r_min, cqr_alpha=cqr_alpha, wass_coeff=c_eps)
        cw_margin = cw_bounds.get(t_rel, {}).get("cw_margin", 0.0)
        q_only = cw_bounds.get(t_rel, {}).get("quantile_1_minus_alpha", 0.0)
        eps_N = cw_bounds.get(t_rel, {}).get("wasserstein_radius", 0.0)
        print(f"\n[Sens-Wass] C_eps={c_eps}  T_rel={t_rel}h  "
              f"q_0.9={q_only:.2f}  eps_N={eps_N:.2f}  margin={cw_margin:.2f}")

        t0 = time.time()
        sol = build_and_solve_milp(
            stations=stations, sys33=sys33, params=params,
            cw_bounds=cw_bounds, policy="CW-DRCC", n_hours=24,
            tou=tou, solver="GUROBI", verbose=False,
            time_limit_sec=180.0, load_scale=load_scale,
            powerflow_model="distflow_socp",
        )
        rainflow_cost = 0.0
        if sol.status == "optimal":
            for b, st in enumerate(stations):
                rainflow_cost += rainflow_degradation_cost(sol.soc[b], st.capacity_kwh)
        mc1 = monte_carlo_reliability_check(
            sol, stations, params, n_trials=mc_trials, seed=seed + 1,
            residual_pool=residual_pool_1h, perturb_scale=1.0,
        )
        mc3 = monte_carlo_reliability_check(
            sol, stations, params, n_trials=mc_trials, seed=seed + 2,
            residual_pool=residual_pool_1h, perturb_scale=3.0,
        )
        mc5 = monte_carlo_reliability_check(
            sol, stations, params, n_trials=mc_trials, seed=seed + 3,
            residual_pool=residual_pool_1h, perturb_scale=5.0,
        )
        rows.append({
            "c_epsilon": c_eps,
            "q_1_minus_alpha_Trel": q_only,
            "eps_N_Trel": eps_N,
            "cw_margin_Trel": cw_margin,
            "status": sol.status,
            "purchase_cost_yuan": sol.purchase_cost,
            "rainflow_cost_yuan": rainflow_cost,
            "total_cost_yuan": sol.purchase_cost + rainflow_cost,
            "mc_reliability_1x": mc1["mc_reliability"],
            "mc_reliability_3x": mc3["mc_reliability"],
            "mc_reliability_5x": mc5["mc_reliability"],
            "solve_time_s": time.time() - t0,
        })
        print(f"  obj={sol.objective_value:.2f}  cost_total={sol.purchase_cost + rainflow_cost:.2f}  "
              f"R_1x={mc1['mc_reliability']:.4f}  R_3x={mc3['mc_reliability']:.4f}  "
              f"R_5x={mc5['mc_reliability']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "sensitivity_wass.csv", index=False)
    print(f"\n[Sens-Wass] 保存到 {out_dir / 'sensitivity_wass.csv'}")
    print(df_out.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(df_out["c_epsilon"], df_out["total_cost_yuan"], marker="o",
            color="#D0021B", linewidth=2, label="CW-DRCC")
    ax.set_xlabel(r"Wasserstein coefficient $C_{\epsilon}$")
    ax.set_ylabel("Total daily cost (Yuan)")
    ax.set_title("Economic cost vs Wasserstein radius")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(df_out["c_epsilon"], df_out["mc_reliability_1x"], marker="o",
            linewidth=2, label="1x residual perturbation", color="#4A90E2")
    ax.plot(df_out["c_epsilon"], df_out["mc_reliability_3x"], marker="s",
            linewidth=2, label="3x residual perturbation", color="#F5A623")
    ax.plot(df_out["c_epsilon"], df_out["mc_reliability_5x"], marker="D",
            linewidth=2, label="5x residual perturbation", color="#D0021B")
    ax.set_xlabel(r"Wasserstein coefficient $C_{\epsilon}$")
    ax.set_ylabel("MC communication reliability")
    ax.set_title("Distributional robustness vs Wasserstein radius")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sensitivity_wass.png", dpi=150)
    plt.close(fig)
    print(f"[Sens-Wass] 图表保存到 {out_dir / 'fig_sensitivity_wass.png'}")


if __name__ == "__main__":
    main()
