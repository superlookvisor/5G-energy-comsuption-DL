"""Phase E 敏感性扫描：通信可靠性目标 R_min。

扫描 R_min ∈ {0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995} 下的三种策略
(Deterministic / CQR-Robust / CW-DRCC) 的经济性与实际可靠性。
"""
from __future__ import annotations

import json
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


R_MIN_GRID = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995]
POLICIES = ["Deterministic", "CQR-Robust", "CW-DRCC"]


def main() -> None:
    out_dir = BASE_DIR / "outputs_sensitivity_rmin"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 20260424
    n_bs = 32
    load_scale = 0.5
    wass_coeff = 2.0
    cqr_alpha = 0.10
    mc_trials = 500

    print("[Sens-Rmin] 加载 Phase C 数据 & IEEE-33...")
    df = load_phaseC_data()
    test_traj = select_test_trajectories(df, n_bs=n_bs, seed=seed)
    sys33 = build_ieee33()
    non_root = [n for n in sys33.nodes if n != sys33.root_node]
    bs_to_node = assign_bs_to_nodes(test_traj, non_root[:n_bs], seed=seed)

    residuals = compute_calibration_residuals(df, seed=seed)
    cw_bounds = compute_cw_drcc_bounds(residuals, alpha=cqr_alpha, radius_coeff=wass_coeff)
    residual_pool_1h = residuals.scores_by_duration.get(1)
    tou = tou_price_vector(24)

    rows: List[Dict] = []
    for r_min in R_MIN_GRID:
        params_preview = BessParams(reliability_target=r_min, cqr_alpha=cqr_alpha,
                                    wass_coeff=wass_coeff)
        t_rel = compute_required_autonomy(params_preview)
        stations = build_base_station_dataset(
            test_traj, bs_to_node, capacity_factor=3.0, t_rel=t_rel,
            soc_init=params_preview.soc_init, soc_min=params_preview.soc_min,
        )
        total_cap = float(sum(st.capacity_kwh for st in stations))

        print(f"\n[Sens-Rmin] R_min={r_min}  T_rel={t_rel}h  total_cap={total_cap:.1f}kWh")
        params = BessParams(reliability_target=r_min, cqr_alpha=cqr_alpha, wass_coeff=wass_coeff)

        for policy in POLICIES:
            t0 = time.time()
            sol = build_and_solve_milp(
                stations=stations, sys33=sys33, params=params,
                cw_bounds=cw_bounds, policy=policy, n_hours=24,
                tou=tou, solver="GUROBI", verbose=False,
                time_limit_sec=180.0, load_scale=load_scale,
                powerflow_model="distflow_socp",
            )
            rainflow_cost = 0.0
            if sol.status == "optimal":
                for b, st in enumerate(stations):
                    rainflow_cost += rainflow_degradation_cost(sol.soc[b], st.capacity_kwh)
            mc = monte_carlo_reliability_check(
                sol, stations, params, n_trials=mc_trials, seed=seed + 1,
                residual_pool=residual_pool_1h, perturb_scale=1.0,
            )
            mc_stress = monte_carlo_reliability_check(
                sol, stations, params, n_trials=mc_trials, seed=seed + 2,
                residual_pool=residual_pool_1h, perturb_scale=3.0,
            )
            cw_margin = cw_bounds.get(t_rel, {}).get("cw_margin", 0.0)
            row = {
                "r_min": r_min,
                "t_rel_h": t_rel,
                "policy": policy,
                "status": sol.status,
                "total_capacity_kwh": total_cap,
                "cw_margin_kwh_T_rel": cw_margin,
                "purchase_cost_yuan": sol.purchase_cost,
                "rainflow_cost_yuan": rainflow_cost,
                "total_cost_yuan": sol.purchase_cost + rainflow_cost,
                "mc_reliability_nominal": mc["mc_reliability"],
                "mc_reliability_stress3x": mc_stress["mc_reliability"],
                "mc_mean_interrupt_h_stress": mc_stress["mc_mean_interruption_h"],
                "solve_time_s": time.time() - t0,
            }
            rows.append(row)
            print(f"  [{policy:14s}] obj={sol.objective_value:.2f}, "
                  f"R_nominal={row['mc_reliability_nominal']:.4f}, "
                  f"R_stress3x={row['mc_reliability_stress3x']:.4f}, "
                  f"t={row['solve_time_s']:.1f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "sensitivity_rmin.csv", index=False)
    print(f"\n[Sens-Rmin] 保存到 {out_dir / 'sensitivity_rmin.csv'}")
    print(df_out.to_string(index=False))

    # 生成扫描图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"Deterministic": "#F5A623", "CQR-Robust": "#7ED321", "CW-DRCC": "#D0021B"}
    markers = {"Deterministic": "o", "CQR-Robust": "s", "CW-DRCC": "D"}
    for policy in POLICIES:
        sub = df_out[df_out["policy"] == policy].sort_values("r_min")
        axes[0].plot(sub["r_min"], sub["total_cost_yuan"], marker=markers[policy],
                     color=colors[policy], label=policy, linewidth=2)
        axes[1].plot(sub["r_min"], sub["mc_reliability_nominal"], marker=markers[policy],
                     color=colors[policy], label=policy, linewidth=2)
        axes[2].plot(sub["r_min"], sub["mc_reliability_stress3x"], marker=markers[policy],
                     color=colors[policy], label=policy, linewidth=2)
    axes[0].set_xlabel("R_min"); axes[0].set_ylabel("Total daily cost (Yuan)")
    axes[0].set_title("Total cost vs reliability target")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("R_min"); axes[1].set_ylabel("MC reliability (nominal)")
    axes[1].set_title("MC reliability under nominal residuals")
    axes[1].set_ylim([0.90, 1.0]); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].plot([R_MIN_GRID[0], R_MIN_GRID[-1]], [R_MIN_GRID[0], R_MIN_GRID[-1]],
                 color="gray", linestyle=":", label="target")
    axes[2].set_xlabel("R_min"); axes[2].set_ylabel("MC reliability (3x stress)")
    axes[2].set_title("MC reliability under 3x stress")
    axes[2].set_ylim([0.80, 1.0]); axes[2].grid(alpha=0.3); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sensitivity_rmin.png", dpi=150)
    plt.close(fig)
    print(f"[Sens-Rmin] 图表保存到 {out_dir / 'fig_sensitivity_rmin.png'}")


if __name__ == "__main__":
    main()
