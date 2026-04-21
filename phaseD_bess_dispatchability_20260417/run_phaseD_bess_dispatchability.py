"""Run Phase D reliability-aware BESS dispatchability analysis."""
from __future__ import annotations

import argparse, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PHASEC_DEFAULT_OUTPUT_DIR = REPO_ROOT / "phaseC_pg_rnp_cqr_20260416" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
KEY_COLS = ["BS", "trajectory_id", "origin_time", "target_time", "horizon"]
EPS = 1e-9

RELIABILITY_POLICIES: Mapping[str, str] = {
    "Oracle Reliability": "oracle",
    "Phase C Point Reliability": "phaseC_point",
    "Reliability-Aware CQR": "reliability_cqr",
    "Hourly Upper Reliability": "hourly_upper",
    "Full BESS Backup": "full_bess",
}

@dataclass(frozen=True)
class PhaseDConfig:
    phasec_output_dir: str
    output_dir: str
    phaseb_strategy: str
    phaseb_model: str
    backup_duration: int
    epsilon: float
    capacity_hours: float
    soc: float
    soc_min: float
    dispatch_price: float
    shortfall_penalty: float
    evaluation_split: str
    repair_distribution: str
    repair_rate: float
    weibull_shape: float
    weibull_scale: float
    reliability_target: float
    energy_risk_epsilon: float
    outage_rate: float
    interruption_penalty: float
    unserved_traffic_penalty: float
    traffic_column: Optional[str]
    max_support_hours: int

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D reliability-aware BESS dispatchability model")
    p.add_argument("--phasec-output-dir", type=Path, default=PHASEC_DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--phaseb-strategy", default="two_stage_proxy")
    p.add_argument("--phaseb-model", default="RandomForest")
    p.add_argument("--backup-duration", type=int, default=4, help="Compatibility baseline duration.")
    p.add_argument("--epsilon", type=float, default=0.10, help="Compatibility fixed-window risk level.")
    p.add_argument("--capacity-hours", type=float, default=8.0, help="C_b = capacity_hours * p_base.")
    p.add_argument("--soc", type=float, default=1.0)
    p.add_argument("--soc-min", type=float, default=0.0)
    p.add_argument("--dispatch-price", type=float, default=1.0)
    p.add_argument("--shortfall-penalty", type=float, default=10.0)
    p.add_argument("--evaluation-split", choices=["test", "calibration", "train", "all"], default="test")
    p.add_argument("--repair-distribution", choices=["exponential", "weibull"], default="exponential")
    p.add_argument("--repair-rate", type=float, default=0.7, help="Exponential repair rate per hour.")
    p.add_argument("--weibull-shape", type=float, default=1.5)
    p.add_argument("--weibull-scale", type=float, default=4.0)
    p.add_argument("--reliability-target", type=float, default=0.99)
    p.add_argument("--energy-risk-epsilon", type=float, default=0.10)
    p.add_argument("--outage-rate", type=float, default=0.01)
    p.add_argument("--interruption-penalty", type=float, default=10.0)
    p.add_argument("--unserved-traffic-penalty", type=float, default=1.0)
    p.add_argument("--traffic-column", default=None)
    p.add_argument("--max-support-hours", type=int, default=24)
    return p.parse_args()

def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def validate_config(c: PhaseDConfig) -> None:
    if c.capacity_hours <= 0:
        raise ValueError("--capacity-hours must be positive")
    if c.soc < 0 or c.soc_min < 0 or c.soc < c.soc_min:
        raise ValueError("SOC parameters must satisfy soc >= soc_min >= 0")
    if c.max_support_hours < 1 or c.max_support_hours > 24:
        raise ValueError("--max-support-hours must be in [1, 24]")
    if not (0 < c.reliability_target < 1) or not (0 < c.energy_risk_epsilon < 1):
        raise ValueError("Reliability and energy risk levels must be in (0, 1)")
    if c.repair_rate <= 0 or c.weibull_shape <= 0 or c.weibull_scale <= 0:
        raise ValueError("Repair distribution parameters must be positive")
    if c.outage_rate < 0:
        raise ValueError("--outage-rate must be non-negative")

def load_hourly_predictions(phasec_output_dir: Path, phaseb_strategy: str, phaseb_model: str, traffic_column: Optional[str]) -> pd.DataFrame:
    phasec_path = phasec_output_dir / "pg_rnp_predictions.csv"
    phaseb_path = phasec_output_dir / "phaseb_baseline_predictions.csv"
    if not phasec_path.exists():
        raise FileNotFoundError(str(phasec_path))
    if not phaseb_path.exists():
        raise FileNotFoundError(str(phaseb_path))
    phasec = pd.read_csv(phasec_path, parse_dates=["origin_time", "target_time"])
    phaseb_all = pd.read_csv(phaseb_path, parse_dates=["origin_time", "target_time"])
    require_columns(phasec, KEY_COLS + ["split", "energy_true", "energy_pred", "lower_90", "upper_90", "p_base"], "pg_rnp_predictions.csv")
    require_columns(phaseb_all, KEY_COLS + ["strategy", "model", "energy_pred"], "phaseb_baseline_predictions.csv")
    phaseb = phaseb_all[(phaseb_all["strategy"].astype(str) == phaseb_strategy) & (phaseb_all["model"].astype(str) == phaseb_model)].copy()
    if phaseb.empty:
        available = phaseb_all[["strategy", "model"]].drop_duplicates().sort_values(["strategy", "model"]).to_dict(orient="records")
        raise ValueError(f"No Phase B rows for strategy={phaseb_strategy!r}, model={phaseb_model!r}. Available: {available}")
    if phasec.duplicated(KEY_COLS).any() or phaseb.duplicated(KEY_COLS).any():
        raise ValueError("Duplicated hourly keys found in Phase B or Phase C predictions")
    optional = [c for c in ["load_mean_hat", "load_pmax_hat", "load_std_hat", traffic_column] if c and c in phasec.columns]
    c_cols = list(dict.fromkeys(KEY_COLS + ["split", "energy_true", "energy_pred", "lower_90", "upper_90", "p_base"] + optional))
    hourly = phasec[c_cols].rename(columns={"energy_pred": "energy_pred_phaseC"})
    phaseb = phaseb[KEY_COLS + ["energy_pred"]].rename(columns={"energy_pred": "energy_pred_phaseB"})
    hourly = hourly.merge(phaseb, on=KEY_COLS, how="left", validate="one_to_one")
    if hourly["energy_pred_phaseB"].isna().any():
        raise ValueError("Selected Phase B predictions are missing for some Phase C rows")
    numeric = ["horizon", "energy_true", "energy_pred_phaseB", "energy_pred_phaseC", "lower_90", "upper_90", "p_base"] + optional
    for col in numeric:
        if col in hourly.columns:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    before = len(hourly)
    hourly = hourly.dropna(subset=["horizon", "energy_true", "energy_pred_phaseB", "energy_pred_phaseC", "upper_90", "p_base"]).copy()
    if len(hourly) != before:
        print(f"Dropped {before - len(hourly)} hourly rows with missing numeric values")
    if traffic_column and traffic_column in hourly.columns:
        hourly["traffic_proxy"] = pd.to_numeric(hourly[traffic_column], errors="coerce")
        source = traffic_column
    elif "load_mean_hat" in hourly.columns:
        hourly["traffic_proxy"] = pd.to_numeric(hourly["load_mean_hat"], errors="coerce")
        source = "load_mean_hat"
    else:
        hourly["traffic_proxy"] = pd.to_numeric(hourly["energy_true"], errors="coerce")
        source = "energy_true"
    hourly["traffic_proxy"] = hourly["traffic_proxy"].fillna(hourly["energy_true"])
    hourly["traffic_proxy_source"] = source
    hourly["horizon"] = hourly["horizon"].astype(int)
    hourly["hourly_error_phaseC"] = hourly["energy_pred_phaseC"] - hourly["energy_true"]
    return hourly

def split_label(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    return str(modes.iloc[0]) if not modes.empty else "unknown"

def safe_p_base(group: pd.DataFrame) -> float:
    pbase = pd.to_numeric(group["p_base"], errors="coerce").dropna()
    if not pbase.empty and float(pbase.median()) > 0:
        return float(pbase.median())
    fallback = pd.to_numeric(group["energy_true"], errors="coerce").dropna()
    return float(max(fallback.median(), EPS)) if not fallback.empty else 1.0

def duration_col(prefix: str, duration: int) -> str:
    return f"{prefix}_T{duration:02d}"

def build_reliability_trajectory_dataset(hourly: pd.DataFrame, max_support_hours: int, capacity_hours: float, soc: float, soc_min: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (bs, trajectory_id, origin_time), group in hourly.groupby(["BS", "trajectory_id", "origin_time"], sort=False):
        group = group.sort_values("horizon").drop_duplicates("horizon").set_index("horizon", drop=False)
        horizons = sorted(int(h) for h in group.index)
        if not horizons:
            continue
        base = safe_p_base(group)
        c_bess = capacity_hours * base
        usable = max(0.0, (soc - soc_min) * c_bess)
        split = split_label(group["split"])
        max_h = min(max(horizons), 24)
        for start_h in horizons:
            if start_h > max_h:
                continue
            row0 = group.loc[start_h]
            t_available = min(max_support_hours, max_h - start_h + 1)
            if t_available < 1:
                continue
            out: Dict[str, object] = {
                "BS": bs, "trajectory_id": trajectory_id, "origin_time": origin_time,
                "outage_start_time": row0["target_time"], "outage_start_horizon": int(start_h),
                "split": split, "T_available": int(t_available), "C_bess": c_bess,
                "usable_bess_energy": usable, "p_base": base, "soc": float(soc), "soc_min": float(soc_min),
                "traffic_proxy_source": str(row0.get("traffic_proxy_source", "energy_true")),
            }
            running_true = running_phaseb = running_phasec = running_upper = 0.0
            for duration in range(1, max_support_hours + 1):
                h = start_h + duration - 1
                if h in group.index and duration <= t_available:
                    r = group.loc[h]
                    running_true += float(r["energy_true"])
                    running_phaseb += float(r["energy_pred_phaseB"])
                    running_phasec += float(r["energy_pred_phaseC"])
                    running_upper += float(r["upper_90"])
                    out[duration_col("R_true", duration)] = running_true
                    out[duration_col("R_pred_phaseB", duration)] = running_phaseb
                    out[duration_col("R_pred_phaseC", duration)] = running_phasec
                    out[duration_col("R_hourly_upper", duration)] = running_upper
                    out[duration_col("traffic", duration)] = float(r["traffic_proxy"])
                else:
                    for prefix in ["R_true", "R_pred_phaseB", "R_pred_phaseC", "R_hourly_upper", "traffic"]:
                        out[duration_col(prefix, duration)] = np.nan
            rows.append(out)
    if not rows:
        raise ValueError("No reliability trajectory rows could be built")
    df = pd.DataFrame(rows)
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    df["outage_start_time"] = pd.to_datetime(df["outage_start_time"])
    return df.sort_values(["split", "BS", "origin_time", "outage_start_horizon"]).reset_index(drop=True)

def conformal_upper_quantile(scores: pd.Series, epsilon: float) -> float:
    clean = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("Empty calibration score set")
    ordered = np.sort(clean)
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - epsilon)))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])

def calibrate_energy_quantiles_by_duration(traj: pd.DataFrame, max_support_hours: int, epsilon: float) -> Dict[int, float]:
    cal = traj[traj["split"].astype(str) == "calibration"].copy()
    if cal.empty:
        raise ValueError("No calibration rows are available for Aggregate-CQR")
    out: Dict[int, float] = {}
    for duration in range(1, max_support_hours + 1):
        scores = cal[duration_col("R_true", duration)] - cal[duration_col("R_pred_phaseC", duration)]
        out[duration] = conformal_upper_quantile(scores, epsilon)
    return out

def repair_cdf(duration: float, c: PhaseDConfig) -> float:
    d = max(0.0, float(duration))
    if c.repair_distribution == "exponential":
        return float(1.0 - math.exp(-c.repair_rate * d))
    return float(1.0 - math.exp(-((d / c.weibull_scale) ** c.weibull_shape)))

def repair_survival(duration: float, c: PhaseDConfig) -> float:
    return float(max(0.0, min(1.0, 1.0 - repair_cdf(duration, c))))

def repair_quantile(prob: float, c: PhaseDConfig) -> float:
    p = min(max(float(prob), EPS), 1.0 - EPS)
    if c.repair_distribution == "exponential":
        return float(-math.log(1.0 - p) / c.repair_rate)
    return float(c.weibull_scale * (-math.log(1.0 - p)) ** (1.0 / c.weibull_shape))

def expected_excess_repair_time(tau: float, c: PhaseDConfig) -> float:
    tau = max(0.0, float(tau))
    if c.repair_distribution == "exponential":
        return float(math.exp(-c.repair_rate * tau) / c.repair_rate)
    upper = max(tau + 10.0 * c.weibull_scale, c.max_support_hours + 10.0 * c.weibull_scale)
    xs = np.linspace(tau, upper, 512)
    survival = np.exp(-((xs / c.weibull_scale) ** c.weibull_shape))
    return float(np.trapz(survival, xs))

def autonomy_time_from_row(row: pd.Series, backup_energy: float, max_support_hours: int) -> float:
    tau = 0.0
    for duration in range(1, max_support_hours + 1):
        val = row.get(duration_col("R_true", duration), np.nan)
        if pd.isna(val):
            break
        if float(val) <= backup_energy + EPS:
            tau = float(duration)
        else:
            break
    return tau

def expected_unserved_traffic(row: pd.Series, tau: float, c: PhaseDConfig) -> float:
    total = 0.0
    for duration in range(1, c.max_support_hours + 1):
        traffic = row.get(duration_col("traffic", duration), np.nan)
        if pd.isna(traffic):
            break
        if duration > tau:
            total += float(traffic) * repair_survival(float(duration), c)
    return float(total)

def apply_reliability_policies(traj: pd.DataFrame, q_by_duration: Dict[int, float], c: PhaseDConfig) -> Tuple[pd.DataFrame, int, float]:
    t_rel_cont = repair_quantile(c.reliability_target, c)
    t_rel = int(math.ceil(t_rel_cont))
    out = traj.copy()
    out["T_rel_continuous"] = t_rel_cont
    out["T_rel_discrete"] = t_rel
    out["required_duration_available"] = out["T_available"].astype(int) >= t_rel
    out["horizon_infeasible"] = ~out["required_duration_available"]
    metric_suffixes = [
        "reliable_energy_requirement", "C_disp", "backup_energy_retained", "autonomy_time_true",
        "comm_reliability", "expected_interruption_duration", "saidi_bs",
        "expected_unserved_traffic", "net_dispatch_value",
    ]
    for suffix in RELIABILITY_POLICIES.values():
        for name in metric_suffixes:
            out[f"{name}_{suffix}"] = np.nan
    if t_rel > c.max_support_hours:
        return out, t_rel, t_rel_cont
    pred_col = duration_col("R_pred_phaseC", t_rel)
    true_col = duration_col("R_true", t_rel)
    upper_col = duration_col("R_hourly_upper", t_rel)
    q_rel = q_by_duration[t_rel]
    for idx, row in out.iterrows():
        if not bool(row["required_duration_available"]):
            continue
        usable = float(row["usable_bess_energy"])
        requirements = {
            "oracle": max(0.0, float(row[true_col])),
            "phaseC_point": max(0.0, float(row[pred_col])),
            "reliability_cqr": max(0.0, float(row[pred_col]) + q_rel),
            "hourly_upper": max(0.0, float(row[upper_col])),
            "full_bess": usable,
        }
        for suffix in RELIABILITY_POLICIES.values():
            req = requirements[suffix]
            c_disp = max(0.0, usable - req)
            backup = max(0.0, usable - c_disp)
            tau = autonomy_time_from_row(row, backup, c.max_support_hours)
            rel = repair_cdf(tau, c)
            ecid = expected_excess_repair_time(tau, c)
            eut = expected_unserved_traffic(row, tau, c)
            net = c.dispatch_price * c_disp - c.interruption_penalty * ecid - c.unserved_traffic_penalty * eut
            out.at[idx, f"reliable_energy_requirement_{suffix}"] = req
            out.at[idx, f"C_disp_{suffix}"] = c_disp
            out.at[idx, f"backup_energy_retained_{suffix}"] = backup
            out.at[idx, f"autonomy_time_true_{suffix}"] = tau
            out.at[idx, f"comm_reliability_{suffix}"] = rel
            out.at[idx, f"expected_interruption_duration_{suffix}"] = ecid
            out.at[idx, f"saidi_bs_{suffix}"] = c.outage_rate * ecid
            out.at[idx, f"expected_unserved_traffic_{suffix}"] = eut
            out.at[idx, f"net_dispatch_value_{suffix}"] = net
    return out, t_rel, t_rel_cont

def evaluation_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return df.copy()
    sub = df[df["split"].astype(str) == split].copy()
    if sub.empty:
        raise ValueError(f"No rows for evaluation split {split!r}")
    return sub

def reliability_metric_rows(decisions_eval: pd.DataFrame, c: PhaseDConfig, t_rel: int, t_rel_cont: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total_capacity = float(decisions_eval["C_bess"].sum())
    horizon_infeas = float(decisions_eval["horizon_infeasible"].mean())
    for policy, suffix in RELIABILITY_POLICIES.items():
        valid = decisions_eval[decisions_eval[f"C_disp_{suffix}"].notna()].copy()
        if valid.empty:
            rows.append({
                "policy": policy, "evaluation_split": c.evaluation_split, "n_decisions": 0, "n_bs": 0,
                "repair_distribution": c.repair_distribution, "reliability_target": c.reliability_target,
                "energy_risk_epsilon": c.energy_risk_epsilon, "T_rel_continuous": t_rel_cont,
                "T_rel_discrete": t_rel, "mean_dispatchable_capacity": np.nan,
                "dispatchable_capacity_ratio": np.nan, "mean_comm_reliability": np.nan,
                "reliability_violation_rate": np.nan, "mean_expected_interruption_duration": np.nan,
                "mean_saidi_bs": np.nan, "mean_expected_unserved_traffic": np.nan,
                "mean_reliable_energy_requirement": np.nan, "mean_autonomy_time_true": np.nan,
                "net_dispatch_value": np.nan, "capacity_infeasibility_rate": np.nan,
                "horizon_infeasibility_rate": horizon_infeas,
            })
            continue
        disp = valid[f"C_disp_{suffix}"].astype(float)
        rel = valid[f"comm_reliability_{suffix}"].astype(float)
        req = valid[f"reliable_energy_requirement_{suffix}"].astype(float)
        rows.append({
            "policy": policy, "evaluation_split": c.evaluation_split, "n_decisions": int(len(valid)),
            "n_bs": int(valid["BS"].nunique()), "repair_distribution": c.repair_distribution,
            "reliability_target": c.reliability_target, "energy_risk_epsilon": c.energy_risk_epsilon,
            "T_rel_continuous": t_rel_cont, "T_rel_discrete": t_rel,
            "mean_dispatchable_capacity": float(disp.mean()),
            "dispatchable_capacity_ratio": float(disp.sum() / max(total_capacity, EPS)),
            "mean_comm_reliability": float(rel.mean()),
            "reliability_violation_rate": float((rel + EPS < c.reliability_target).mean()),
            "mean_expected_interruption_duration": float(valid[f"expected_interruption_duration_{suffix}"].mean()),
            "mean_saidi_bs": float(valid[f"saidi_bs_{suffix}"].mean()),
            "mean_expected_unserved_traffic": float(valid[f"expected_unserved_traffic_{suffix}"].mean()),
            "mean_reliable_energy_requirement": float(req.mean()),
            "mean_autonomy_time_true": float(valid[f"autonomy_time_true_{suffix}"].mean()),
            "net_dispatch_value": float(valid[f"net_dispatch_value_{suffix}"].mean()),
            "capacity_infeasibility_rate": float((req > valid["usable_bess_energy"].astype(float) + EPS).mean()),
            "horizon_infeasibility_rate": horizon_infeas,
        })
    return pd.DataFrame(rows)

def reliability_by_horizon(decisions_eval: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for horizon, group in decisions_eval.groupby("outage_start_horizon"):
        row: Dict[str, object] = {"outage_start_horizon": int(horizon), "n_decisions": int(len(group))}
        for _, suffix in RELIABILITY_POLICIES.items():
            row[f"{suffix}_mean_dispatchable_capacity"] = float(group[f"C_disp_{suffix}"].mean())
            row[f"{suffix}_mean_comm_reliability"] = float(group[f"comm_reliability_{suffix}"].mean())
            row[f"{suffix}_mean_autonomy_time"] = float(group[f"autonomy_time_true_{suffix}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("outage_start_horizon")

def setup_plot_style() -> None:
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

def save_figures(output_dir: Path, decisions_eval: pd.DataFrame, metrics: pd.DataFrame, q_by_duration: Dict[int, float], c: PhaseDConfig) -> None:
    setup_plot_style()
    xs = np.linspace(0, c.max_support_hours, 200)
    ys = [repair_cdf(float(x), c) for x in xs]
    t_rel = repair_quantile(c.reliability_target, c)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys, color="#276FBF", linewidth=2)
    ax.axhline(c.reliability_target, color="#B23A48", linestyle="--", label="target")
    ax.axvline(t_rel, color="#2A9D8F", linestyle="--", label=f"T_rel={t_rel:.2f}h")
    ax.set_title("Repair-time CDF and communication reliability")
    ax.set_xlabel("BESS autonomy time (h)"); ax.set_ylabel("Reliability"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "fig1_repair_distribution_reliability_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data, labels = [], []
    for policy, suffix in RELIABILITY_POLICIES.items():
        vals = decisions_eval[f"autonomy_time_true_{suffix}"].dropna().to_numpy(dtype=float)
        if len(vals):
            data.append(vals); labels.append(policy)
    ax.boxplot(data, tick_labels=labels, showfliers=False); ax.tick_params(axis="x", rotation=25)
    ax.set_title("BESS autonomy time by policy"); ax.set_ylabel("Autonomy time (h)")
    fig.tight_layout(); fig.savefig(output_dir / "fig2_autonomy_time_distribution.png"); plt.close(fig)

    targets = np.array([0.90, 0.95, 0.99, 0.995, 0.999])
    caps: List[float] = []
    for target in targets:
        tr = int(math.ceil(repair_quantile(float(target), c)))
        if tr < 1 or tr > c.max_support_hours:
            caps.append(np.nan); continue
        pred_col = duration_col("R_pred_phaseC", tr)
        req = decisions_eval[pred_col].astype(float) + q_by_duration[tr]
        feasible = decisions_eval["T_available"].astype(int) >= tr
        disp = (decisions_eval.loc[feasible, "usable_bess_energy"].astype(float) - req.loc[feasible]).clip(lower=0.0)
        caps.append(float(disp.mean()) if len(disp) else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.plot(targets, caps, marker="o", color="#B23A48")
    ax.set_title("Dispatchable capacity vs reliability target"); ax.set_xlabel("Reliability target"); ax.set_ylabel("Mean dispatchable capacity")
    fig.tight_layout(); fig.savefig(output_dir / "fig3_dispatchable_capacity_vs_reliability_target.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, row in metrics.dropna(subset=["mean_comm_reliability"]).iterrows():
        ax.scatter(row["mean_comm_reliability"], row["mean_dispatchable_capacity"], s=60)
        ax.annotate(str(row["policy"]), (row["mean_comm_reliability"], row["mean_dispatchable_capacity"]), xytext=(4, 3), textcoords="offset points")
    ax.set_title("Reliability-dispatchability trade-off"); ax.set_xlabel("Mean reliability"); ax.set_ylabel("Mean dispatchable capacity")
    fig.tight_layout(); fig.savefig(output_dir / "fig4_comm_reliability_dispatch_tradeoff.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0)); sub = metrics.dropna(subset=["mean_expected_interruption_duration"])
    ax.bar(sub["policy"], sub["mean_expected_interruption_duration"], color="#276FBF")
    ax.set_title("Expected communication interruption duration"); ax.set_ylabel("Mean ECID (h)"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output_dir / "fig5_expected_interruption_duration.png"); plt.close(fig)

def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if math.isnan(float(value)) else f"{float(value):.{digits}f}"
    return str(value)

def markdown_table(df: pd.DataFrame, cols: List[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)

def write_outputs(output_dir: Path, hourly: pd.DataFrame, trajectories: pd.DataFrame, decisions: pd.DataFrame, metrics: pd.DataFrame, by_horizon: pd.DataFrame, q_by_duration: Dict[int, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_dir / "aligned_hourly_predictions.csv", index=False)
    trajectories.to_csv(output_dir / "reliability_trajectory_dataset.csv", index=False)
    decisions.to_csv(output_dir / "reliability_policy_decisions.csv", index=False)
    metrics.to_csv(output_dir / "reliability_policy_metrics.csv", index=False)
    by_horizon.to_csv(output_dir / "reliability_by_horizon.csv", index=False)
    pd.DataFrame([{"duration_hours": k, "aggregate_cqr_q": v} for k, v in sorted(q_by_duration.items())]).to_csv(output_dir / "energy_cqr_quantiles_by_duration.csv", index=False)

def write_report(output_dir: Path, c: PhaseDConfig, hourly: pd.DataFrame, trajectories: pd.DataFrame, decisions_eval: pd.DataFrame, metrics: pd.DataFrame, t_rel: int, t_rel_cont: float, q_by_duration: Dict[int, float]) -> None:
    cols = ["policy", "mean_dispatchable_capacity", "mean_comm_reliability", "reliability_violation_rate", "mean_expected_interruption_duration", "mean_expected_unserved_traffic", "net_dispatch_value", "capacity_infeasibility_rate"]
    q_text = "nan" if t_rel not in q_by_duration else f"{q_by_duration[t_rel]:.6f}"
    report = f"""# Phase D Reliability-Aware Dispatchability Report

## Research Design

This run models communication reliability through outage repair time and BESS autonomy time. The key event is `D_b <= tau_b,t`, meaning the outage is repaired before backup energy is exhausted. Phase C uncertainty builds an Aggregate-CQR upper bound for cumulative energy over the required autonomy duration.

## Inputs And Assumptions

- Phase C output directory: `{c.phasec_output_dir}`
- BESS scenario: `C_b = {c.capacity_hours:.2f} * p_base`, `SOC={c.soc:.2f}`, `SOC_min={c.soc_min:.2f}`
- Repair distribution: `{c.repair_distribution}`
- Reliability target `R_min`: `{c.reliability_target:.6f}`
- Required autonomy time: `{t_rel_cont:.4f}` hours, discretized to `{t_rel}` hours
- Energy uncertainty risk epsilon: `{c.energy_risk_epsilon:.4f}`
- Aggregate-CQR margin at required duration: `{q_text}`
- Evaluation split: `{c.evaluation_split}`

## Data Coverage

- Aligned hourly rows: `{len(hourly)}`
- Reliability trajectory rows: `{len(trajectories)}`
- Evaluation decisions: `{len(decisions_eval)}`
- Evaluation base stations: `{decisions_eval['BS'].nunique()}`
- Horizon infeasibility rate: `{decisions_eval['horizon_infeasible'].mean():.4f}`

## Main Metrics

{markdown_table(metrics, cols)}

## Interpretation

The reliability-aware CQR policy reserves enough BESS energy to cover the Phase C cumulative energy upper bound over the repair-time quantile. Increasing the reliability target increases the required autonomy time, which increases retained communication-backup energy and reduces dispatchable capacity. Full BESS Backup gives the maximum communication reliability under installed capacity, while Oracle Reliability is an infeasible benchmark that uses true future energy.

## Outputs

- `aligned_hourly_predictions.csv`
- `reliability_trajectory_dataset.csv`
- `reliability_policy_decisions.csv`
- `reliability_policy_metrics.csv`
- `reliability_by_horizon.csv`
- `energy_cqr_quantiles_by_duration.csv`
- `reliability_aware_dispatchability_report.md`
- `run_meta.json`
- `fig1_repair_distribution_reliability_curve.png`
- `fig2_autonomy_time_distribution.png`
- `fig3_dispatchable_capacity_vs_reliability_target.png`
- `fig4_comm_reliability_dispatch_tradeoff.png`
- `fig5_expected_interruption_duration.png`
"""
    (output_dir / "reliability_aware_dispatchability_report.md").write_text(report, encoding="utf-8")

def write_meta(output_dir: Path, c: PhaseDConfig, hourly: pd.DataFrame, trajectories: pd.DataFrame, decisions: pd.DataFrame, q_by_duration: Dict[int, float], t_rel: int, t_rel_cont: float) -> None:
    meta = {
        "config": asdict(c),
        "required_autonomy_hours_continuous": t_rel_cont,
        "required_autonomy_hours_discrete": t_rel,
        "energy_cqr_quantiles_by_duration": {str(k): v for k, v in sorted(q_by_duration.items())},
        "hourly_rows": int(len(hourly)),
        "hourly_base_stations": int(hourly["BS"].nunique()),
        "trajectory_rows": int(len(trajectories)),
        "decision_rows": int(len(decisions)),
        "decision_base_stations": int(decisions["BS"].nunique()),
        "decisions_by_split": decisions["split"].value_counts().to_dict(),
        "horizon_infeasible_rows": int(decisions["horizon_infeasible"].sum()),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def main() -> None:
    args = parse_args()
    c = PhaseDConfig(
        phasec_output_dir=str(args.phasec_output_dir), output_dir=str(args.output_dir),
        phaseb_strategy=args.phaseb_strategy, phaseb_model=args.phaseb_model,
        backup_duration=args.backup_duration, epsilon=args.epsilon, capacity_hours=args.capacity_hours,
        soc=args.soc, soc_min=args.soc_min, dispatch_price=args.dispatch_price,
        shortfall_penalty=args.shortfall_penalty, evaluation_split=args.evaluation_split,
        repair_distribution=args.repair_distribution, repair_rate=args.repair_rate,
        weibull_shape=args.weibull_shape, weibull_scale=args.weibull_scale,
        reliability_target=args.reliability_target, energy_risk_epsilon=args.energy_risk_epsilon,
        outage_rate=args.outage_rate, interruption_penalty=args.interruption_penalty,
        unserved_traffic_penalty=args.unserved_traffic_penalty, traffic_column=args.traffic_column,
        max_support_hours=args.max_support_hours,
    )
    validate_config(c)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_predictions(args.phasec_output_dir, args.phaseb_strategy, args.phaseb_model, args.traffic_column)
    trajectories = build_reliability_trajectory_dataset(hourly, args.max_support_hours, args.capacity_hours, args.soc, args.soc_min)
    q_by_duration = calibrate_energy_quantiles_by_duration(trajectories, args.max_support_hours, args.energy_risk_epsilon)
    decisions, t_rel, t_rel_cont = apply_reliability_policies(trajectories, q_by_duration, c)
    decisions_eval = evaluation_frame(decisions, args.evaluation_split)
    metrics = reliability_metric_rows(decisions_eval, c, t_rel, t_rel_cont)
    by_horizon = reliability_by_horizon(decisions_eval)
    write_outputs(args.output_dir, hourly, trajectories, decisions, metrics, by_horizon, q_by_duration)
    save_figures(args.output_dir, decisions_eval, metrics, q_by_duration, c)
    write_report(args.output_dir, c, hourly, trajectories, decisions_eval, metrics, t_rel, t_rel_cont, q_by_duration)
    write_meta(args.output_dir, c, hourly, trajectories, decisions, q_by_duration, t_rel, t_rel_cont)
    print(f"Phase D reliability-aware analysis finished. Outputs: {args.output_dir.resolve()}")
    print(metrics[["policy", "mean_dispatchable_capacity", "mean_comm_reliability", "reliability_violation_rate", "mean_expected_interruption_duration"]].to_string(index=False))

if __name__ == "__main__":
    main()

