"""
学习日前多源代理权重（全局常数）。

约束：w >= 0, sum(w)=1。实现采用 softmax 参数化：w=softmax(a)，天然落在 simplex 上。
监督信号：目标时刻真实 covariates（不使用未来 Energy）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _softmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a - np.max(a)
    ea = np.exp(a)
    return ea / np.sum(ea)


def _prepare_xy(
    candidates: List[pd.Series],
    y_true: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """对齐并丢掉缺失，返回 (X, y)"""
    xs = [pd.to_numeric(s, errors="coerce") for s in candidates]
    y = pd.to_numeric(y_true, errors="coerce")
    df = pd.concat(xs + [y], axis=1)
    df = df.dropna()
    if df.empty:
        return np.empty((0, len(candidates))), np.empty((0,))
    X = df.iloc[:, : len(candidates)].to_numpy(dtype=float)
    yv = df.iloc[:, len(candidates)].to_numpy(dtype=float)
    return X, yv


def _fit_simplex_ls(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> Dict[str, Any]:
    """
    最小化 ||y - Xw||^2 + l2*||w||^2, s.t. w=softmax(a) => w>=0,sum=1
    """
    k = X.shape[1]
    if X.shape[0] == 0:
        return {"weights": [1.0 / k] * k, "n": 0, "rmse": None, "mae": None}

    def obj(a: np.ndarray) -> float:
        w = _softmax(a)
        pred = X @ w
        err = pred - y
        return float(np.mean(err**2) + l2 * np.sum(w**2))

    a0 = np.zeros(k, dtype=float)
    res = minimize(obj, a0, method="L-BFGS-B")
    w = _softmax(res.x)
    pred = X @ w
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    return {"weights": [float(v) for v in w], "n": int(len(y)), "rmse": rmse, "mae": mae, "success": bool(res.success)}


@dataclass(frozen=True)
class ProxyWeights:
    load_mean: List[float]  # prevday, roll24, current
    load_pmax: List[float]  # prevday, roll24, current
    load_std: List[float]  # prevday, roll24
    s_hat: List[float]  # prevday, hour_prior


def learn_global_proxy_weights(
    panel: pd.DataFrame,
    horizons: Iterable[int],
    s_cols: List[str],
    rng_seed: int = 42,
    train_bs_frac: float = 0.8,
) -> Tuple[ProxyWeights, Dict[str, Any]]:
    """
    在 panel 上学习全局权重。做法：
    - 按 BS 进行 train/valid 切分（避免同站泄漏）
    - 对每个 horizon 构造目标时刻真值（y_true covariates）
    - 汇总所有 horizon 的样本拟合一组 w（全局常数）
    """
    work = panel.sort_values(["BS", "Time"]).copy()
    work_idx = work.set_index(["BS", "Time"], drop=False)

    # 目标时刻真值（covariates）
    # 注意：与 build_dayahead_dataset 保持一致：按 (BS, Time) 精确对齐，而非按行位移 shift。
    bs_list = work["BS"].astype(str).dropna().unique().tolist()
    rs = np.random.RandomState(rng_seed)
    rs.shuffle(bs_list)
    cut = int(len(bs_list) * train_bs_frac)
    train_bs = set(bs_list[:cut])
    valid_bs = set(bs_list[cut:])

    rows: Dict[str, List[pd.Series]] = {
        "load_mean_prevday": [],
        "load_mean_roll24": [],
        "load_mean_current": [],
        "load_mean_true": [],
        "load_pmax_prevday": [],
        "load_pmax_roll24": [],
        "load_pmax_current": [],
        "load_pmax_true": [],
        "load_std_prevday": [],
        "load_std_roll24": [],
        "load_std_current": [],
        "load_std_true": [],
    }
    # ES（跨模式共享一组权重）
    s_prevday_all: List[pd.Series] = []
    s_prior_all: List[pd.Series] = []
    s_true_all: List[pd.Series] = []

    for h in horizons:
        origin = work[work["hour"] == 0].copy()
        origin = origin[origin["BS"].astype(str).isin(train_bs)].sort_values(["BS", "Time"])

        origin_time = pd.to_datetime(origin["Time"], errors="coerce")
        target_time = origin_time + pd.Timedelta(hours=int(h))
        target_key = pd.MultiIndex.from_arrays([origin["BS"], target_time])
        # valid: target row exists in panel
        valid = work_idx["Time"].reindex(target_key).notna().to_numpy()
        origin = origin.loc[valid].copy()
        origin_time = pd.to_datetime(origin["Time"], errors="coerce")
        target_time = origin_time + pd.Timedelta(hours=int(h))
        target_key = pd.MultiIndex.from_arrays([origin["BS"], target_time])
        if origin.empty:
            continue

        # 目标真值：取目标时刻 covariates（按 (BS, Time)）
        load_mean_true = work_idx["load_mean"].reindex(target_key)
        load_pmax_true = work_idx["load_pmax_weighted"].reindex(target_key)
        load_std_true = work_idx["load_std"].reindex(target_key)

        # prevday 同小时：相对目标时刻往前 24h
        prevday_time = target_time - pd.Timedelta(hours=24)
        prevday_key = pd.MultiIndex.from_arrays([origin["BS"], prevday_time])
        load_mean_prevday = work_idx["load_mean"].reindex(prevday_key)
        load_pmax_prevday = work_idx["load_pmax_weighted"].reindex(prevday_key)
        load_std_prevday = work_idx["load_std"].reindex(prevday_key)

        # roll24/current from origin row (already computed in panel)
        rows["load_mean_prevday"].append(load_mean_prevday)
        rows["load_mean_roll24"].append(origin["load_mean_roll24"])
        rows["load_mean_current"].append(origin["load_mean"])
        rows["load_mean_true"].append(load_mean_true)

        rows["load_pmax_prevday"].append(load_pmax_prevday)
        rows["load_pmax_roll24"].append(origin["load_pmax_roll24"])
        rows["load_pmax_current"].append(origin["load_pmax_weighted"])
        rows["load_pmax_true"].append(load_pmax_true)

        rows["load_std_prevday"].append(load_std_prevday)
        rows["load_std_roll24"].append(origin["load_std_roll24"])
        rows["load_std_current"].append(origin["load_std"])
        rows["load_std_true"].append(load_std_true)

        for col in s_cols:
            s_prevday = work_idx[col].reindex(prevday_key)
            s_prior = origin[f"{col}_hour_prior"]
            s_true = work_idx[col].reindex(target_key)
            s_prevday_all.append(s_prevday)
            s_prior_all.append(s_prior)
            s_true_all.append(s_true)

    def cat(series_list: List[pd.Series]) -> pd.Series:
        return pd.concat(series_list, ignore_index=True) if series_list else pd.Series(dtype=float)

    load_mean_fit = _fit_simplex_ls(*_prepare_xy([cat(rows["load_mean_prevday"]), cat(rows["load_mean_roll24"]), cat(rows["load_mean_current"])], cat(rows["load_mean_true"])))
    load_pmax_fit = _fit_simplex_ls(*_prepare_xy([cat(rows["load_pmax_prevday"]), cat(rows["load_pmax_roll24"]), cat(rows["load_pmax_current"])], cat(rows["load_pmax_true"])))
    load_std_fit = _fit_simplex_ls(*_prepare_xy([cat(rows["load_std_prevday"]), cat(rows["load_std_roll24"])], cat(rows["load_std_true"])))
    s_fit = _fit_simplex_ls(*_prepare_xy([cat(s_prevday_all), cat(s_prior_all)], cat(s_true_all)))

    weights = ProxyWeights(
        load_mean=load_mean_fit["weights"],
        load_pmax=load_pmax_fit["weights"],
        load_std=load_std_fit["weights"],
        s_hat=s_fit["weights"],
    )

    meta: Dict[str, Any] = {
        "rng_seed": rng_seed,
        "train_bs_frac": train_bs_frac,
        "n_bs_total": int(len(bs_list)),
        "n_bs_train": int(len(train_bs)),
        "n_bs_valid": int(len(valid_bs)),
        "horizons_used": [int(h) for h in horizons],
        "fit_load_mean": load_mean_fit,
        "fit_load_pmax": load_pmax_fit,
        "fit_load_std": load_std_fit,
        "fit_s_hat": s_fit,
        "note": "weights learned to predict target-time covariates; Energy not used.",
    }
    return weights, meta


def save_weights(out_dir: Path, weights: ProxyWeights, meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proxy_weights.json").write_text(json.dumps(weights.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "proxy_weights_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

