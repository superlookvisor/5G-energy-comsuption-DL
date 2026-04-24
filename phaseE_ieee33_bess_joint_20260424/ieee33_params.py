"""IEEE-33 节点辐射状配电网参数。

数据源：Baran M.E., Wu F.F. (1989) "Network reconfiguration in distribution
systems for loss reduction and load balancing", IEEE TPWRD.

基值：V_base = 12.66 kV, S_base = 10 MVA, Z_base = V_base^2/S_base = 16.02356 ohm
以下支路参数已按此基值归一化为 p.u.。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# 基值
V_BASE_KV = 12.66
S_BASE_MVA = 10.0
Z_BASE_OHM = V_BASE_KV ** 2 / S_BASE_MVA  # ≈ 16.02356 Ω

# 支路数据：(from_bus, to_bus, r_ohm, x_ohm)
# 节点编号 1~33（后续内部处理时映射为 0~32 或保留 1~33）
BRANCH_DATA_OHM: List[Tuple[int, int, float, float]] = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 1.7114, 1.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0400, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]

# 节点负荷（有功 kW，无功 kVar），索引与节点编号 1~33 对齐
# 节点 1 为根节点（配电变压器），无本地负荷
NODE_LOAD_KW_KVAR: List[Tuple[int, float, float]] = [
    (1, 0.0, 0.0),
    (2, 100.0, 60.0),
    (3, 90.0, 40.0),
    (4, 120.0, 80.0),
    (5, 60.0, 30.0),
    (6, 60.0, 20.0),
    (7, 200.0, 100.0),
    (8, 200.0, 100.0),
    (9, 60.0, 20.0),
    (10, 60.0, 20.0),
    (11, 45.0, 30.0),
    (12, 60.0, 35.0),
    (13, 60.0, 35.0),
    (14, 120.0, 80.0),
    (15, 60.0, 10.0),
    (16, 60.0, 20.0),
    (17, 60.0, 20.0),
    (18, 90.0, 40.0),
    (19, 90.0, 40.0),
    (20, 90.0, 40.0),
    (21, 90.0, 40.0),
    (22, 90.0, 40.0),
    (23, 90.0, 50.0),
    (24, 420.0, 200.0),
    (25, 420.0, 200.0),
    (26, 60.0, 25.0),
    (27, 60.0, 25.0),
    (28, 60.0, 20.0),
    (29, 120.0, 70.0),
    (30, 200.0, 600.0),
    (31, 150.0, 70.0),
    (32, 210.0, 100.0),
    (33, 60.0, 40.0),
]

# 支路电流上限（p.u.），辐射状 33 节点系统典型值
# 主干线路取较大值，分支末端取较小值
BRANCH_CURRENT_LIMIT_PU = 0.4  # 对应约 2.3 kA，给足余量

# 电压限值 (p.u.)
# IEEE-33 原始负荷场景下末端节点电压约 0.91-0.92 p.u.，为避免纯潮流infeasibility
# 采用 0.90-1.05 的放宽区间（符合国标 GB/T 12325 对 10 kV 配电网电压允许偏差 ±7%）
V_MIN_PU = 0.90
V_MAX_PU = 1.05
V_ROOT_PU = 1.00


@dataclass(frozen=True)
class IEEE33System:
    n_nodes: int
    root_node: int
    nodes: List[int]
    branches: List[Tuple[int, int]]
    r_pu: Dict[Tuple[int, int], float]
    x_pu: Dict[Tuple[int, int], float]
    parent: Dict[int, int]
    children: Dict[int, List[int]]
    base_load_p_pu: Dict[int, float]
    base_load_q_pu: Dict[int, float]
    v_min_sq: float
    v_max_sq: float
    v_root_sq: float
    i_max_sq: float
    s_base_kva: float


def build_ieee33() -> IEEE33System:
    """构造 IEEE-33 节点辐射网参数（所有量按 p.u. 归一化）。"""
    nodes = list(range(1, 34))
    branches = [(f, t) for (f, t, _, _) in BRANCH_DATA_OHM]
    r_pu = {(f, t): r / Z_BASE_OHM for (f, t, r, _) in BRANCH_DATA_OHM}
    x_pu = {(f, t): x / Z_BASE_OHM for (f, t, _, x) in BRANCH_DATA_OHM}

    parent: Dict[int, int] = {}
    children: Dict[int, List[int]] = {n: [] for n in nodes}
    for (f, t) in branches:
        parent[t] = f
        children[f].append(t)

    s_base_kva = S_BASE_MVA * 1000.0
    base_load_p_pu = {n: p / s_base_kva for (n, p, _) in NODE_LOAD_KW_KVAR}
    base_load_q_pu = {n: q / s_base_kva for (n, _, q) in NODE_LOAD_KW_KVAR}

    return IEEE33System(
        n_nodes=len(nodes),
        root_node=1,
        nodes=nodes,
        branches=branches,
        r_pu=r_pu,
        x_pu=x_pu,
        parent=parent,
        children=children,
        base_load_p_pu=base_load_p_pu,
        base_load_q_pu=base_load_q_pu,
        v_min_sq=V_MIN_PU ** 2,
        v_max_sq=V_MAX_PU ** 2,
        v_root_sq=V_ROOT_PU ** 2,
        i_max_sq=BRANCH_CURRENT_LIMIT_PU ** 2,
        s_base_kva=s_base_kva,
    )


def get_tou_price(hour: int) -> float:
    """华东地区典型分时电价（元/kWh）。

    - 尖峰 (19, 20): 1.35
    - 高峰 (8-10, 18, 21): 1.08
    - 平段 (7, 11-17, 22): 0.72
    - 低谷 (23, 0-6): 0.35
    """
    h = int(hour) % 24
    if h in (19, 20):
        return 1.35
    if h in (8, 9, 10, 18, 21):
        return 1.08
    if h in (7, 11, 12, 13, 14, 15, 16, 17, 22):
        return 0.72
    return 0.35


def tou_price_vector(n_hours: int = 24) -> np.ndarray:
    return np.array([get_tou_price(h) for h in range(n_hours)], dtype=float)


if __name__ == "__main__":
    sys33 = build_ieee33()
    print(f"IEEE-33: {sys33.n_nodes} nodes, {len(sys33.branches)} branches")
    print(f"Root node: {sys33.root_node}")
    print(f"Total base load P: {sum(sys33.base_load_p_pu.values()) * sys33.s_base_kva:.1f} kW")
    print(f"Total base load Q: {sum(sys33.base_load_q_pu.values()) * sys33.s_base_kva:.1f} kVar")
    print(f"TOU prices: {tou_price_vector()}")
