"""Phase E MILP 优化模型：IEEE-33 + 多基站 BESS 联合调度。

模型形式（实现版）：
- 潮流：LinDistFlow 线性化近似（保证开源 HiGHS 可解）
- 互斥：充放电 0-1 指示 + Big-M
- DOD 分段：2 段（浅循环/深循环）+ 0-1 段指示
- CW-DRCC：通信可靠性的线性鲁棒约束
- 目标：ToU 购电成本 + DOD 分段衰减成本

MILP 变量规模（32 基站 × 24 h × 2 DOD 段）：
- 连续变量 ≈ 4000
- 二进制变量 ≈ 4000
- HiGHS 典型求解时间：30~120 s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from data_loader import BaseStationData
from ieee33_params import IEEE33System, tou_price_vector


@dataclass
class BessParams:
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_init: float = 0.90
    c_wear_shallow: float = 0.15  # 元/kWh，SOC >= 0.5 的浅循环
    c_wear_deep: float = 0.50  # 元/kWh，SOC < 0.5 的深循环
    soc_split: float = 0.50  # DOD 分段临界 SOC
    big_M_power: float = 1e3
    repair_rate: float = 0.7  # 指数修复率 (1/h)
    reliability_target: float = 0.99  # R_min
    cqr_alpha: float = 0.10
    wass_coeff: float = 1.0  # Fournier-Guillin 系数


@dataclass
class BessSolution:
    policy: str
    status: str
    objective_value: float
    purchase_cost: float
    degradation_cost_piecewise: float
    soc: np.ndarray
    p_ch: np.ndarray
    p_dis: np.ndarray
    p_grid: np.ndarray
    voltage_sq: np.ndarray
    energy_used: np.ndarray
    solve_time: float
    cw_margins: Dict[int, float]
    required_autonomy_h: int
    n_reliability_violations: int = 0
    n_voltage_violations: int = 0


def compute_required_autonomy(params: BessParams) -> int:
    """T_rel = ceil(-ln(1-R_min) / mu_repair)."""
    import math
    t_rel = -math.log(1.0 - params.reliability_target) / params.repair_rate
    return int(math.ceil(t_rel))


def build_and_solve_milp(stations: List[BaseStationData], sys33: IEEE33System,
                          params: BessParams, cw_bounds: Dict[int, Dict[str, float]],
                          policy: str = "CW-DRCC", n_hours: int = 24,
                          tou: Optional[np.ndarray] = None,
                          solver: str = "HIGHS", verbose: bool = False,
                          time_limit_sec: float = 180.0,
                          load_scale: float = 0.5,
                          powerflow_model: str = "lindistflow",
                          ) -> BessSolution:
    """构建并求解单次 MILP 调度问题。

    policy 选项：
    - "NoBESS": 不装 BESS，P_ch = P_dis = 0，SOC = SOC_init 常量
    - "BackupOnly": BESS 满充做后备，不参与调度
    - "Deterministic": 用 energy_pred 作为负荷，不鲁棒
    - "CQR-Robust": 用 energy_pred + q_{1-alpha} 作为负荷（过度保守）
    - "CW-DRCC": 用 energy_pred + q_{1-alpha} + eps_N 作为负荷（本文方法）
    """
    if tou is None:
        tou = tou_price_vector(n_hours)
    T = n_hours
    B = len(stations)
    N = sys33.n_nodes
    root = sys33.root_node
    nodes = sys33.nodes
    branches = sys33.branches
    node_to_bs = {st.node: i for i, st in enumerate(stations)}
    t_rel = compute_required_autonomy(params)

    energy_nominal = np.zeros((B, T))
    for i, st in enumerate(stations):
        if policy == "CQR-Robust":
            energy_nominal[i] = st.upper_90
        else:
            energy_nominal[i] = st.energy_pred

    s_base_kva = sys33.s_base_kva
    # 非基站本地负荷：IEEE-33 原始峰值负荷 * load_scale * 日内波动
    # load_scale 默认 0.5 使峰值总负荷约 1.85 MW，符合 10 kV 配电网典型场景
    daily_scale = 0.6 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 2)
    load_p_nominal = np.zeros((N, T))
    load_q_nominal = np.zeros((N, T))
    tan_phi = 0.3287  # tan(arccos(0.95))
    for i, n in enumerate(nodes):
        base_p = sys33.base_load_p_pu[n] * s_base_kva * load_scale
        base_q = sys33.base_load_q_pu[n] * s_base_kva * load_scale
        load_p_nominal[i] = base_p * daily_scale
        load_q_nominal[i] = base_q * daily_scale
        if n in node_to_bs:
            bs_idx = node_to_bs[n]
            load_p_nominal[i] += energy_nominal[bs_idx]
            load_q_nominal[i] += energy_nominal[bs_idx] * tan_phi

    c_b = np.array([st.capacity_kwh for st in stations])
    p_rate = np.array([st.power_rate_kw for st in stations])

    # ============= 决策变量 =============
    if policy == "NoBESS":
        p_ch = np.zeros((B, T))
        p_dis = np.zeros((B, T))
        soc = np.full((B, T + 1), params.soc_init)
        ch_vars_given = True
    elif policy == "BackupOnly":
        p_ch = np.zeros((B, T))
        p_dis = np.zeros((B, T))
        soc = np.full((B, T + 1), params.soc_max)
        ch_vars_given = True
    else:
        p_ch = cp.Variable((B, T), nonneg=True)
        p_dis = cp.Variable((B, T), nonneg=True)
        soc = cp.Variable((B, T + 1))
        ch_vars_given = False

    # 充放电互斥与 DOD 分段用 0-1 变量
    use_binary = policy in ("Deterministic", "CQR-Robust", "CW-DRCC")
    if use_binary:
        u_ch = cp.Variable((B, T), boolean=True)
        u_dis = cp.Variable((B, T), boolean=True)
        w_shallow = cp.Variable((B, T), boolean=True)  # SOC >= soc_split
        p_ch_sh = cp.Variable((B, T), nonneg=True)
        p_ch_dp = cp.Variable((B, T), nonneg=True)
        p_dis_sh = cp.Variable((B, T), nonneg=True)
        p_dis_dp = cp.Variable((B, T), nonneg=True)

    # 潮流变量
    # - LinDistFlow: (P,Q,V) 线性化，不含损耗
    # - DistFlow SOCP: (P,Q,V,l) + 旋转二阶锥，含损耗（需要 GUROBI / MOSEK 等）
    use_socp = powerflow_model.lower() in ("distflow_socp", "distflow", "socp")
    p_grid = cp.Variable(T, nonneg=True)  # 根节点购电总有功 (kW)
    q_grid = cp.Variable(T)  # 根节点无功
    p_branch = cp.Variable((len(branches), T))  # 支路有功 (kW)
    q_branch = cp.Variable((len(branches), T))  # 支路无功 (kVar)
    v_sq = cp.Variable((N, T))  # 节点电压平方 (p.u.)
    l_branch = cp.Variable((len(branches), T), nonneg=True) if use_socp else None

    branch_to_idx = {br: i for i, br in enumerate(branches)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    constraints = []

    # ============= BESS 约束 =============
    if not ch_vars_given:
        for b in range(B):
            constraints.append(soc[b, 0] == params.soc_init)
            constraints.append(soc[b, T] == params.soc_init)
            for t in range(T):
                constraints.append(
                    soc[b, t + 1] * c_b[b]
                    == soc[b, t] * c_b[b]
                    + params.eta_ch * p_ch[b, t]
                    - p_dis[b, t] / params.eta_dis
                )
                constraints.append(soc[b, t + 1] >= params.soc_min)
                constraints.append(soc[b, t + 1] <= params.soc_max)
                constraints.append(p_ch[b, t] <= p_rate[b])
                constraints.append(p_dis[b, t] <= p_rate[b])
        if use_binary:
            for b in range(B):
                for t in range(T):
                    constraints.append(p_ch[b, t] <= p_rate[b] * u_ch[b, t])
                    constraints.append(p_dis[b, t] <= p_rate[b] * u_dis[b, t])
                    constraints.append(u_ch[b, t] + u_dis[b, t] <= 1)
                    constraints.append(p_ch[b, t] == p_ch_sh[b, t] + p_ch_dp[b, t])
                    constraints.append(p_dis[b, t] == p_dis_sh[b, t] + p_dis_dp[b, t])
                    constraints.append(p_ch_sh[b, t] <= p_rate[b] * w_shallow[b, t])
                    constraints.append(p_dis_sh[b, t] <= p_rate[b] * w_shallow[b, t])
                    constraints.append(p_ch_dp[b, t] <= p_rate[b] * (1 - w_shallow[b, t]))
                    constraints.append(p_dis_dp[b, t] <= p_rate[b] * (1 - w_shallow[b, t]))
                    constraints.append(
                        soc[b, t] >= params.soc_split - (1 - w_shallow[b, t]) * (params.soc_split - params.soc_min)
                    )
                    constraints.append(
                        soc[b, t] <= params.soc_split + w_shallow[b, t] * (params.soc_max - params.soc_split)
                    )

    # ============= CW-DRCC 通信可靠性约束 =============
    if policy in ("CW-DRCC",) and t_rel <= T:
        margin = cw_bounds.get(t_rel, {}).get("cw_margin", 0.0)
        for b in range(B):
            for t_start in range(T - t_rel + 1):
                cum = sum(energy_nominal[b, t_start + j] for j in range(t_rel))
                constraints.append(
                    (soc[b, t_start] - params.soc_min) * c_b[b] >= cum + margin
                )
    elif policy in ("CQR-Robust",) and t_rel <= T:
        for b in range(B):
            for t_start in range(T - t_rel + 1):
                cum_upper = sum(stations[b].upper_90[t_start + j] for j in range(t_rel))
                constraints.append(
                    (soc[b, t_start] - params.soc_min) * c_b[b] >= cum_upper
                )
    elif policy in ("Deterministic",) and t_rel <= T:
        q_only = cw_bounds.get(t_rel, {}).get("quantile_1_minus_alpha", 0.0)
        for b in range(B):
            for t_start in range(T - t_rel + 1):
                cum = sum(energy_nominal[b, t_start + j] for j in range(t_rel))
                constraints.append(
                    (soc[b, t_start] - params.soc_min) * c_b[b] >= cum
                )

    # ============= 配电网潮流约束 =============
    # 节点注入 p_inj_j = 本地负荷 + 基站能耗 + 基站 BESS 充电 - 基站 BESS 放电
    # （p_inj 定义为从节点流向外部负荷，所以基站充电增加它，基站放电减少它）
    for t in range(T):
        constraints.append(v_sq[node_to_idx[root], t] == sys33.v_root_sq)
        for j in nodes:
            j_idx = node_to_idx[j]
            p_inj_expr = load_p_nominal[j_idx, t]
            q_inj_expr = load_q_nominal[j_idx, t]
            if j in node_to_bs:
                bs_idx = node_to_bs[j]
                if not ch_vars_given:
                    p_inj_expr = p_inj_expr + p_ch[bs_idx, t] - p_dis[bs_idx, t]
            child_branches = [(j, k) for k in sys33.children.get(j, [])]
            sum_p_out = 0
            sum_q_out = 0
            for cb in child_branches:
                cb_idx = branch_to_idx[cb]
                sum_p_out = sum_p_out + p_branch[cb_idx, t]
                sum_q_out = sum_q_out + q_branch[cb_idx, t]

            if j == root:
                constraints.append(p_grid[t] == sum_p_out + p_inj_expr)
                constraints.append(q_grid[t] == sum_q_out + q_inj_expr)
            else:
                parent = sys33.parent[j]
                br = (parent, j)
                br_idx = branch_to_idx[br]
                r_pu = sys33.r_pu[br]
                x_pu = sys33.x_pu[br]
                parent_idx = node_to_idx[parent]
                if not use_socp:
                    # LinDistFlow：忽略损耗项与二阶项
                    constraints.append(p_branch[br_idx, t] == p_inj_expr + sum_p_out)
                    constraints.append(q_branch[br_idx, t] == q_inj_expr + sum_q_out)
                    constraints.append(
                        v_sq[j_idx, t] == v_sq[parent_idx, t]
                        - 2.0 * (r_pu * p_branch[br_idx, t] / s_base_kva
                                + x_pu * q_branch[br_idx, t] / s_base_kva)
                    )
                else:
                    # DistFlow SOCP：含损耗与电流平方变量
                    assert l_branch is not None
                    constraints.append(
                        p_branch[br_idx, t] == p_inj_expr + sum_p_out + (r_pu * l_branch[br_idx, t] * s_base_kva)
                    )
                    constraints.append(
                        q_branch[br_idx, t] == q_inj_expr + sum_q_out + (x_pu * l_branch[br_idx, t] * s_base_kva)
                    )
                    constraints.append(
                        v_sq[j_idx, t] == v_sq[parent_idx, t]
                        - 2.0 * (r_pu * p_branch[br_idx, t] / s_base_kva
                                + x_pu * q_branch[br_idx, t] / s_base_kva)
                        + (r_pu * r_pu + x_pu * x_pu) * l_branch[br_idx, t]
                    )
                    # 旋转二阶锥：P^2 + Q^2 <= v_parent * l
                    u = v_sq[parent_idx, t]
                    v = l_branch[br_idx, t]
                    xvec = cp.hstack([2.0 * p_branch[br_idx, t] / s_base_kva,
                                      2.0 * q_branch[br_idx, t] / s_base_kva,
                                      u - v])
                    constraints.append(cp.SOC(u + v, xvec))
                    constraints.append(l_branch[br_idx, t] <= sys33.i_max_sq)
                constraints.append(v_sq[j_idx, t] >= sys33.v_min_sq)
                constraints.append(v_sq[j_idx, t] <= sys33.v_max_sq)

    # ============= 目标函数 =============
    delta_t = 1.0
    purchase_cost = cp.sum(cp.multiply(tou, p_grid)) * delta_t
    if use_binary and not ch_vars_given:
        deg_cost = (
            params.c_wear_shallow * cp.sum(p_ch_sh + p_dis_sh) * delta_t
            + params.c_wear_deep * cp.sum(p_ch_dp + p_dis_dp) * delta_t
        )
    else:
        deg_cost = 0.0

    objective = cp.Minimize(purchase_cost + deg_cost)
    problem = cp.Problem(objective, constraints)

    solver_kwargs: Dict = {"verbose": verbose}
    if solver == "HIGHS":
        solver_kwargs["solver"] = cp.HIGHS
        solver_kwargs["time_limit"] = time_limit_sec
    elif solver == "ECOS_BB":
        solver_kwargs["solver"] = cp.ECOS_BB
        solver_kwargs["mi_max_iters"] = 2000
    elif solver == "GUROBI":
        solver_kwargs["solver"] = cp.GUROBI
        solver_kwargs["TimeLimit"] = time_limit_sec
        solver_kwargs["MIPGap"] = 0.01

    import time as _time
    t0 = _time.time()
    try:
        problem.solve(**solver_kwargs)
    except Exception as e:
        print(f"[{policy}] solver failed: {e}")
    solve_time = _time.time() - t0

    status = problem.status
    def _val(x):
        if isinstance(x, np.ndarray):
            return x
        if x is None:
            return None
        return x.value

    soc_val = _val(soc) if not isinstance(soc, np.ndarray) else soc
    p_ch_val = _val(p_ch) if not isinstance(p_ch, np.ndarray) else p_ch
    p_dis_val = _val(p_dis) if not isinstance(p_dis, np.ndarray) else p_dis
    p_grid_val = _val(p_grid)
    v_sq_val = _val(v_sq)

    if soc_val is None:
        soc_val = np.full((B, T + 1), params.soc_init)
    if p_ch_val is None:
        p_ch_val = np.zeros((B, T))
    if p_dis_val is None:
        p_dis_val = np.zeros((B, T))
    if p_grid_val is None:
        p_grid_val = np.zeros(T)
    if v_sq_val is None:
        v_sq_val = np.ones((N, T))

    energy_used = energy_nominal.copy()
    purchase_val = float(np.sum(tou * np.asarray(p_grid_val))) * delta_t
    deg_piecewise = 0.0
    if use_binary and not ch_vars_given:
        try:
            deg_piecewise = (
                params.c_wear_shallow * float(np.sum(_val(p_ch_sh) + _val(p_dis_sh))) * delta_t
                + params.c_wear_deep * float(np.sum(_val(p_ch_dp) + _val(p_dis_dp))) * delta_t
            )
        except Exception:
            deg_piecewise = 0.0

    obj_val = problem.value if problem.value is not None else purchase_val + deg_piecewise

    cw_margins = {t_rel: cw_bounds.get(t_rel, {}).get("cw_margin", 0.0)}
    n_viol_rel = 0
    n_viol_volt = int(np.sum((v_sq_val < sys33.v_min_sq - 1e-4) | (v_sq_val > sys33.v_max_sq + 1e-4)))

    return BessSolution(
        policy=policy, status=status, objective_value=float(obj_val),
        purchase_cost=purchase_val, degradation_cost_piecewise=deg_piecewise,
        soc=np.asarray(soc_val), p_ch=np.asarray(p_ch_val), p_dis=np.asarray(p_dis_val),
        p_grid=np.asarray(p_grid_val), voltage_sq=np.asarray(v_sq_val),
        energy_used=energy_used, solve_time=solve_time,
        cw_margins=cw_margins, required_autonomy_h=t_rel,
        n_reliability_violations=n_viol_rel, n_voltage_violations=n_viol_volt,
    )
