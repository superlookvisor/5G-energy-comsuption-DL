"""Phase E 数据加载：从 Phase C 输出加载基站能耗预测和校准残差。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PHASEC_OUTPUT_DIR = REPO_ROOT / "phaseC_pg_rnp_cqr_20260416" / "outputs"


@dataclass
class BaseStationData:
    """单个基站的日前预测数据。"""
    bs_id: str
    node: int
    trajectory_id: str
    energy_pred: np.ndarray
    energy_true: np.ndarray
    lower_90: np.ndarray
    upper_90: np.ndarray
    p_base: float
    e_max: float
    capacity_kwh: float
    power_rate_kw: float


@dataclass
class CalibrationResiduals:
    """校准集残差：支持任意聚合持续时间 T 的累计误差分数。"""
    horizons: np.ndarray
    scores_by_duration: Dict[int, np.ndarray]


def load_phaseC_data(phasec_dir: Path = PHASEC_OUTPUT_DIR) -> pd.DataFrame:
    """加载 Phase C 的完整预测表。"""
    csv_path = phasec_dir / "pg_rnp_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase C predictions not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["origin_time", "target_time"])
    required = ["BS", "trajectory_id", "horizon", "energy_true", "energy_pred",
                "lower_90", "upper_90", "p_base", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Phase C data missing columns: {missing}")
    return df


def select_test_trajectories(df: pd.DataFrame, n_bs: int = 32, seed: int = 20260424,
                              min_coverage: int = 6) -> pd.DataFrame:
    """为 IEEE-33 的 32 个非根节点随机挑选 n_bs 个基站并合成 24 小时能耗曲线。

    由于 Phase C 日前预测数据稀疏（完整 24 小时轨迹稀缺），本函数按
    `target_hour`（0..23）对每个基站的全部 split 数据聚合，得到该基站的
    小时级典型日负荷曲线：
    - energy_pred: 该 BS 在每个小时的点预测均值
    - energy_true: 该 BS 在每个小时的真实能耗均值
    - lower_90 / upper_90: 对应 CQR 区间的小时级均值

    要求每个基站至少有 `min_coverage` 个不同小时的数据。
    """
    test_df = df[df["split"] == "test"].copy()
    if "target_hour" not in test_df.columns and "target_time" in test_df.columns:
        test_df["target_hour"] = pd.to_datetime(test_df["target_time"]).dt.hour
    coverage = test_df.groupby("BS")["target_hour"].nunique()
    eligible = coverage[coverage >= min_coverage].index.tolist()
    if len(eligible) < n_bs:
        extra = df[df["BS"].isin(df["BS"].unique()) & ~df["BS"].isin(eligible)].copy()
        if "target_hour" not in extra.columns and "target_time" in extra.columns:
            extra["target_hour"] = pd.to_datetime(extra["target_time"]).dt.hour
        extra_cov = extra.groupby("BS")["target_hour"].nunique()
        extra_eligible = extra_cov[extra_cov >= min_coverage].index.tolist()
        eligible = list(dict.fromkeys(eligible + extra_eligible))
        work_df = df.copy()
        if "target_hour" not in work_df.columns and "target_time" in work_df.columns:
            work_df["target_hour"] = pd.to_datetime(work_df["target_time"]).dt.hour
    else:
        work_df = test_df

    if len(eligible) < n_bs:
        raise ValueError(f"Only {len(eligible)} base stations with >= {min_coverage}h coverage, need {n_bs}")

    rng = np.random.default_rng(seed)
    chosen_bs = rng.choice(eligible, size=n_bs, replace=False)

    agg_cols = ["energy_pred", "energy_true", "lower_90", "upper_90", "p_base"]
    aggregated_rows: List[Dict] = []
    for bs in chosen_bs:
        bs_data = work_df[work_df["BS"] == bs]
        if len(bs_data) == 0:
            continue
        hourly = bs_data.groupby("target_hour")[agg_cols].mean().reset_index()
        overall_mean = bs_data[agg_cols].mean()
        traj_id = str(bs_data["trajectory_id"].iloc[0])
        for h in range(24):
            if h in hourly["target_hour"].values:
                row = hourly[hourly["target_hour"] == h].iloc[0]
                aggregated_rows.append({
                    "BS": bs, "trajectory_id": traj_id, "horizon": h + 1, "target_hour": h,
                    "energy_pred": float(row["energy_pred"]),
                    "energy_true": float(row["energy_true"]),
                    "lower_90": float(row["lower_90"]),
                    "upper_90": float(row["upper_90"]),
                    "p_base": float(row["p_base"]),
                })
            else:
                aggregated_rows.append({
                    "BS": bs, "trajectory_id": traj_id, "horizon": h + 1, "target_hour": h,
                    "energy_pred": float(overall_mean["energy_pred"]),
                    "energy_true": float(overall_mean["energy_true"]),
                    "lower_90": float(overall_mean["lower_90"]),
                    "upper_90": float(overall_mean["upper_90"]),
                    "p_base": float(overall_mean["p_base"]),
                })
    combined = pd.DataFrame(aggregated_rows)
    return combined


def assign_bs_to_nodes(test_trajectories: pd.DataFrame, non_root_nodes: List[int],
                       seed: int = 20260424) -> Dict[str, int]:
    """将基站随机分配到 IEEE-33 的非根节点。"""
    bs_list = test_trajectories["BS"].unique().tolist()
    if len(bs_list) != len(non_root_nodes):
        raise ValueError(f"BS count {len(bs_list)} != non-root node count {len(non_root_nodes)}")
    rng = np.random.default_rng(seed + 1)
    node_perm = list(non_root_nodes)
    rng.shuffle(node_perm)
    return {bs: node for bs, node in zip(bs_list, node_perm)}


def build_base_station_dataset(test_trajectories: pd.DataFrame,
                                bs_to_node: Dict[str, int],
                                capacity_factor: float = 3.0,
                                t_rel: int = 3,
                                soc_init: float = 0.90,
                                soc_min: float = 0.05,
                                feasibility_margin: float = 1.15,
                                feasibility_margin_energy_upper: bool = True
                                ) -> List[BaseStationData]:
    """构造每个基站的 24 小时日前数据结构。

    BESS 容量设计：
    - 基础容量 C_b^(0) = capacity_factor * max(energy_pred)
    - 可行性下界：C_b^(feas) = feasibility_margin * max_t (sum_{t..t+T_rel-1} upper_90)
                              / (soc_init - soc_min)
      （确保起末 SOC=soc_init 的约束下，任意时刻可用能量 >= T_rel 累计 CQR 上界）
    - 最终 C_b = max(C_b^(0), C_b^(feas))
    这样既保持用户设定的 "3 倍最大能耗" 原则，又在该值不足时微调至可行。
    """
    stations: List[BaseStationData] = []
    for bs_id, group in test_trajectories.groupby("BS", sort=False):
        g = group.sort_values("horizon")
        if len(g) != 24:
            raise ValueError(f"BS {bs_id} has {len(g)} hours, expected 24")
        energy_pred = g["energy_pred"].to_numpy(dtype=float)
        energy_true = g["energy_true"].to_numpy(dtype=float)
        lower_90 = g["lower_90"].to_numpy(dtype=float)
        upper_90 = g["upper_90"].to_numpy(dtype=float)
        p_base = float(g["p_base"].iloc[0])
        e_max = float(np.max(energy_pred))

        base_cap = capacity_factor * e_max
        cum_profile = upper_90 if feasibility_margin_energy_upper else energy_pred
        max_cum = 0.0
        for t_start in range(max(1, 24 - t_rel + 1)):
            max_cum = max(max_cum, float(np.sum(cum_profile[t_start:t_start + t_rel])))
        feas_cap = feasibility_margin * max_cum / max(soc_init - soc_min, 1e-6)
        capacity_kwh = max(base_cap, feas_cap)

        power_rate_kw = 0.5 * e_max
        traj_id = str(g["trajectory_id"].iloc[0])
        stations.append(BaseStationData(
            bs_id=bs_id,
            node=bs_to_node[bs_id],
            trajectory_id=traj_id,
            energy_pred=energy_pred,
            energy_true=energy_true,
            lower_90=lower_90,
            upper_90=upper_90,
            p_base=p_base,
            e_max=e_max,
            capacity_kwh=capacity_kwh,
            power_rate_kw=power_rate_kw,
        ))
    return stations


def compute_calibration_residuals(df: pd.DataFrame, max_duration: int = 24,
                                    bootstrap_samples: int = 500, seed: int = 20260424
                                    ) -> CalibrationResiduals:
    """从 calibration split 构造任意持续时间 T 的累计误差分数。

    由于完整 24 小时校准轨迹稀缺，本函数采用 block-bootstrap 方式：
    1. 对每个 calibration 基站，取其所有小时级 (y_true - y_pred) 残差样本
    2. 对每个 T ∈ [1, max_duration]，做 bootstrap_samples 次 T-size 连续抽样
    3. 每次抽样的累加和作为一个 s^{(T)}_i 分数

    这样得到的经验分布用于构造 Wasserstein 模糊集。
    """
    cal_df = df[df["split"] == "calibration"].copy()
    rng = np.random.default_rng(seed)
    scores: Dict[int, List[float]] = {T: [] for T in range(1, max_duration + 1)}

    for bs, bs_data in cal_df.groupby("BS"):
        err = (bs_data["energy_true"].to_numpy(dtype=float)
               - bs_data["energy_pred"].to_numpy(dtype=float))
        if len(err) < 2:
            continue
        for T in range(1, max_duration + 1):
            n_draw = min(bootstrap_samples, max(50, len(err) * 5))
            for _ in range(n_draw):
                indices = rng.integers(0, len(err), size=T)
                scores[T].append(float(np.sum(err[indices])))

    horizons = np.arange(1, max_duration + 1)
    scores_by_duration = {T: np.asarray(scores[T], dtype=float) for T in scores}
    return CalibrationResiduals(horizons=horizons, scores_by_duration=scores_by_duration)


def fournier_guillin_radius(scores: np.ndarray, alpha: float = 0.10, C: float = 1.0) -> float:
    """Wasserstein 半径的 Fournier-Guillin 浓度系数（1 维情况）。

    epsilon_N = C * std(scores) * N^{-1/3} * sqrt(log(1/alpha))
    其中 alpha 是置信失败概率。
    """
    N = len(scores)
    if N < 2:
        return 0.0
    sigma = float(np.std(scores, ddof=1))
    return float(C * sigma * max(N, 1) ** (-1.0 / 3.0) * np.sqrt(np.log(1.0 / max(alpha, 1e-6))))


def empirical_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """(1-alpha) 分位数，conformal 校准用。"""
    if len(scores) == 0:
        return 0.0
    N = len(scores)
    idx = int(np.ceil((N + 1) * (1.0 - alpha))) - 1
    idx = min(max(idx, 0), N - 1)
    return float(np.sort(scores)[idx])


def compute_cw_drcc_bounds(residuals: CalibrationResiduals, alpha: float = 0.10,
                            radius_coeff: float = 1.0) -> Dict[int, Dict[str, float]]:
    """计算 CW-DRCC 累计能耗上界所需的 (q_{1-alpha}, epsilon_N) 参数。

    上界修正项 = q_{1-alpha} + epsilon_N
    """
    out: Dict[int, Dict[str, float]] = {}
    for T, scores in residuals.scores_by_duration.items():
        q = empirical_quantile(scores, alpha)
        eps = fournier_guillin_radius(scores, alpha, radius_coeff)
        out[T] = {
            "quantile_1_minus_alpha": q,
            "wasserstein_radius": eps,
            "cw_margin": q + eps,
            "n_samples": int(len(scores)),
            "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        }
    return out


if __name__ == "__main__":
    df = load_phaseC_data()
    print(f"Loaded {len(df)} rows from Phase C")
    traj = select_test_trajectories(df, n_bs=32)
    print(f"Selected {traj['BS'].nunique()} base stations, {len(traj)} rows")
    residuals = compute_calibration_residuals(df)
    print(f"Calibration residuals built for durations 1..{len(residuals.horizons)}")
    for T in [1, 3, 6, 12, 24]:
        s = residuals.scores_by_duration[T]
        print(f"  T={T:2d}: N={len(s):4d}, mean={np.mean(s):8.3f}, std={np.std(s):8.3f}, q90={np.quantile(s, 0.9):8.3f}")
    bounds = compute_cw_drcc_bounds(residuals, alpha=0.10)
    print("\nCW-DRCC bounds (alpha=0.10):")
    for T in [1, 3, 6, 12, 24]:
        b = bounds[T]
        print(f"  T={T:2d}: q_0.9={b['quantile_1_minus_alpha']:8.3f}, "
              f"eps_N={b['wasserstein_radius']:8.3f}, cw_margin={b['cw_margin']:8.3f}")
