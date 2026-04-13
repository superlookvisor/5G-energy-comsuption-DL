"""
05_pi_anp_evaluation.py - Conformal Calibration + Evaluation + Visualization
基站能耗预测建模项目

功能：
1. 对 PI-ANP 预测结果进行 CQR（Conformalized Quantile Regression）校准
2. EnCQR（Ensemble CQR）通过 MC-Dropout 多次前向传播实现
3. 多场景评估：标准 / Day-Ahead / Intra-Day / Few-Shot
4. IEEE TSG 出版级可视化
5. DER 接口适配（兼容 soc_min_scheduler）

参考文献：
  - Romano et al., "Conformalized Quantile Regression", NeurIPS 2019
  - Jensen et al., "Ensemble Conformalized Quantile Regression", IEEE TNNLS 2024
  - Xie et al., "Boosted Conformal Prediction Intervals", NeurIPS 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# 复用 06 中的模块
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pi_anp_imports import (
    AttentiveNeuralProcess, PhysicsConstraint, HardwareConfigMapper,
    ANP_FEATURES, ANP_DAY_AHEAD_FEATURES, PHYSICS_RAW_FEATURES,
    load_and_prepare_data, TrainingConfig, predict_all,
)

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "pi_anp"
RESULT_DIR = PROJECT_ROOT / "results" / "pi_anp_cqr"
PLOT_DIR = PROJECT_ROOT / "plots"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CQR Calibrator
# ============================================================================
class CQRCalibrator:
    """
    Conformalized Quantile Regression 校准器

    将 ANP 的 N(μ, σ) 预测转换为有覆盖率保证的预测区间。
    """

    def __init__(self, alpha: float = 0.10):
        """
        alpha: 错误率。alpha=0.10 → 90% 覆盖率保证。
        """
        self.alpha = alpha
        self.q_hat = None

    def fit(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        z_nominal: float = 1.645,
    ):
        """
        在校准集上拟合。

        参数：
            y_true: 真实值
            mu: ANP 预测均值
            sigma: ANP 预测标准差
            z_nominal: 名义分位数对应的 z 值（1.645 for 90%）
        """
        q_lo = mu - z_nominal * sigma
        q_hi = mu + z_nominal * sigma

        # Nonconformity score (CQR score)
        scores = np.maximum(q_lo - y_true, y_true - q_hi)

        # Conformal quantile
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))

        print(f"[CQR] 校准样本: {n}, α={self.alpha}, Q_hat={self.q_hat:.4f}")

    def predict(
        self, mu: np.ndarray, sigma: np.ndarray, z_nominal: float = 1.645
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回校准后的预测区间。

        返回：
            lower, upper: 校准后的预测下界和上界
        """
        if self.q_hat is None:
            raise ValueError("请先调用 fit()")

        q_lo = mu - z_nominal * sigma
        q_hi = mu + z_nominal * sigma

        lower = q_lo - self.q_hat
        upper = q_hi + self.q_hat
        return lower, upper


class EnCQRCalibrator:
    """
    Ensemble CQR：使用 MC-Dropout 构建 ensemble，
    对每次 forward pass 分别计算 nonconformity scores 后聚合。
    """

    def __init__(self, alpha: float = 0.10, n_ensemble: int = 5):
        self.alpha = alpha
        self.n_ensemble = n_ensemble
        self.q_hat = None

    def fit(
        self,
        y_true: np.ndarray,
        mu_ensemble: np.ndarray,
        sigma_ensemble: np.ndarray,
        z_nominal: float = 1.645,
    ):
        """
        mu_ensemble: (n_ensemble, N) 每个 ensemble 成员的预测均值
        sigma_ensemble: (n_ensemble, N) 每个 ensemble 成员的预测标准差
        """
        all_scores = []
        for i in range(self.n_ensemble):
            q_lo = mu_ensemble[i] - z_nominal * sigma_ensemble[i]
            q_hi = mu_ensemble[i] + z_nominal * sigma_ensemble[i]
            scores = np.maximum(q_lo - y_true, y_true - q_hi)
            all_scores.append(scores)

        # 对每个样本取 ensemble 中的最大 score
        all_scores = np.array(all_scores)  # (n_ensemble, N)
        agg_scores = np.mean(all_scores, axis=0)  # 取 mean

        n = len(agg_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(agg_scores, level))

        print(f"[EnCQR] 校准样本: {n}, ensemble={self.n_ensemble}, α={self.alpha}, Q_hat={self.q_hat:.4f}")

    def predict(
        self, mu: np.ndarray, sigma: np.ndarray, z_nominal: float = 1.645
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise ValueError("请先调用 fit()")

        q_lo = mu - z_nominal * sigma
        q_hi = mu + z_nominal * sigma
        lower = q_lo - self.q_hat
        upper = q_hi + self.q_hat
        return lower, upper


# ============================================================================
# MC-Dropout 预测
# ============================================================================
def mc_dropout_predict(
    model: 'AttentiveNeuralProcess',
    df: pd.DataFrame,
    features: List[str],
    hw_mapper: 'HardwareConfigMapper',
    feature_scaler,
    target_mean: float,
    target_std: float,
    device: torch.device,
    n_mc: int = 5,
    context_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MC-Dropout 预测：每次 forward 保留 dropout，得到 ensemble 输出。

    返回：
        mu_mean: (N,) ensemble 均值
        sigma_mean: (N,) ensemble 平均 aleatoric std
        mu_ensemble: (n_mc, N) 每次的预测均值
        sigma_ensemble: (n_mc, N) 每次的预测标准差
    """
    from sklearn.preprocessing import StandardScaler

    X_raw = df[features].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)
    y_raw = df['energy'].values
    hw_ids_all = hw_mapper.transform_fast(df)

    N = len(df)
    mu_ensemble = np.full((n_mc, N), np.nan)
    sigma_ensemble = np.full((n_mc, N), np.nan)

    for mc_i in range(n_mc):
        # 启用 dropout（train mode）
        model.train()

        for bs_id, group in df.groupby('bs_id'):
            idx = group.index.values
            n = len(idx)
            if n < 5:
                continue

            n_ctx = max(3, int(n * context_ratio))
            ctx_idx = idx[:n_ctx]
            all_idx = idx

            hw = torch.tensor(hw_ids_all[idx[0]], dtype=torch.long).unsqueeze(0).to(device)
            ctx_x = torch.tensor(X_scaled[ctx_idx], dtype=torch.float32).unsqueeze(0).to(device)
            ctx_y_s = torch.tensor(
                (y_raw[ctx_idx] - target_mean) / target_std, dtype=torch.float32
            ).unsqueeze(0).to(device)
            tgt_x = torch.tensor(X_scaled[all_idx], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(ctx_x, ctx_y_s, tgt_x, hw, target_y=None)
                mu = out['mu'].cpu().numpy().flatten() * target_std + target_mean
                sigma = out['sigma'].cpu().numpy().flatten() * target_std

            mu_ensemble[mc_i, all_idx] = mu
            sigma_ensemble[mc_i, all_idx] = sigma

    model.eval()

    # 聚合
    mu_mean = np.nanmean(mu_ensemble, axis=0)
    sigma_mean = np.nanmean(sigma_ensemble, axis=0)

    return mu_mean, sigma_mean, mu_ensemble, sigma_ensemble


# ============================================================================
# 评估函数
# ============================================================================
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    confidence: float = 0.9,
) -> dict:
    """评估预测结果（复用现有指标定义）"""
    metrics = {}

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['MAPE'] = mape
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    metrics['RMSE'] = rmse
    mae = np.mean(np.abs(y_true - y_pred))
    metrics['MAE'] = mae
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    metrics['R2'] = r2

    if y_lower is not None and y_upper is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100
        metrics[f'Coverage_{int(confidence*100)}'] = coverage
        avg_width = np.mean(y_upper - y_lower)
        metrics[f'IntervalWidth_{int(confidence*100)}'] = avg_width

        alpha = 1 - confidence
        width = y_upper - y_lower
        w_score = np.mean(
            width
            + (2 / alpha) * np.maximum(y_lower - y_true, 0)
            + (2 / alpha) * np.maximum(y_true - y_upper, 0)
        )
        metrics[f'WinklerScore_{int(confidence*100)}'] = w_score

    return metrics


# ============================================================================
# DER 接口适配器
# ============================================================================
def anp_to_der_results(
    mu: np.ndarray,
    sigma: np.ndarray,
    mu_ensemble: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    将 ANP 输出转换为 soc_min_scheduler 兼容的 DER 格式。

    兼容: compute_soc_min_from_der_results(load_forecast, der_results, ...)
    """
    aleatoric_var = sigma ** 2
    if mu_ensemble is not None and mu_ensemble.shape[0] > 1:
        epistemic_var = np.nanvar(mu_ensemble, axis=0)
    else:
        epistemic_var = np.zeros_like(aleatoric_var)

    return {
        'gamma': mu,
        'aleatoric': aleatoric_var,
        'epistemic': epistemic_var,
        'total': aleatoric_var + epistemic_var,
    }


# ============================================================================
# IEEE TSG 出版级可视化
# ============================================================================

# IEEE 风格颜色
IEEE_COLORS = {
    'blue': '#0070C0',
    'orange': '#FF6B35',
    'green': '#00B050',
    'red': '#C00000',
    'purple': '#7030A0',
}


def setup_ieee_style():
    """设置 IEEE TSG 出版级样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
    })


def plot_cqr_comparison(
    metrics_before: dict,
    metrics_after: dict,
    save_path: Path,
):
    """CQR 校准前后对比图"""
    setup_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Coverage
    ax = axes[0]
    vals = [metrics_before.get('Coverage_90', 0), metrics_after.get('Coverage_90', 0)]
    bars = ax.bar(['Before CQR', 'After CQR'], vals,
                  color=[IEEE_COLORS['blue'], IEEE_COLORS['green']], edgecolor='black', linewidth=0.5)
    ax.axhline(y=90, color=IEEE_COLORS['red'], linestyle='--', linewidth=0.8, label='Target 90%')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('(a) Coverage Rate')
    ax.legend(fontsize=7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

    # Interval Width
    ax = axes[1]
    vals = [metrics_before.get('IntervalWidth_90', 0), metrics_after.get('IntervalWidth_90', 0)]
    bars = ax.bar(['Before CQR', 'After CQR'], vals,
                  color=[IEEE_COLORS['blue'], IEEE_COLORS['green']], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Interval Width (kWh)')
    ax.set_title('(b) Interval Width')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # Winkler Score
    ax = axes[2]
    vals = [metrics_before.get('WinklerScore_90', 0), metrics_after.get('WinklerScore_90', 0)]
    bars = ax.bar(['Before CQR', 'After CQR'], vals,
                  color=[IEEE_COLORS['blue'], IEEE_COLORS['green']], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Winkler Score')
    ax.set_title('(c) Winkler Score')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[保存] CQR 对比图: {save_path}")


def plot_method_comparison(
    pi_anp_metrics: dict,
    lgbm_metrics: Optional[dict],
    save_path: Path,
):
    """PI-ANP-CQR vs LightGBM 对比图"""
    setup_ieee_style()
    metrics_keys = ['MAPE', 'RMSE', 'MAE', 'Coverage_90']
    labels = ['MAPE (%)', 'RMSE (kWh)', 'MAE (kWh)', 'Coverage (%)']

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.2))

    for ax, key, label in zip(axes, metrics_keys, labels):
        v1 = pi_anp_metrics.get(key, 0)
        methods = ['PI-ANP-CQR']
        values = [v1]
        colors = [IEEE_COLORS['blue']]

        if lgbm_metrics is not None:
            v2 = lgbm_metrics.get(key, 0)
            methods.append('LightGBM-QR')
            values.append(v2)
            colors.append(IEEE_COLORS['orange'])

        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis='x', rotation=15, labelsize=6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[保存] 方法对比图: {save_path}")


def plot_uncertainty_decomposition(
    mu: np.ndarray,
    sigma: np.ndarray,
    mu_ensemble: np.ndarray,
    y_true: np.ndarray,
    save_path: Path,
    n_show: int = 200,
):
    """不确定性分解图：aleatoric vs epistemic"""
    setup_ieee_style()

    aleatoric = sigma ** 2
    epistemic = np.nanvar(mu_ensemble, axis=0)
    total = aleatoric + epistemic

    # 取前 n_show 个样本展示
    idx = np.arange(min(n_show, len(mu)))

    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    # 上图: 预测 vs 真实
    ax = axes[0]
    ax.plot(idx, y_true[idx], color=IEEE_COLORS['blue'], linewidth=0.8, label='Actual')
    ax.plot(idx, mu[idx], color=IEEE_COLORS['red'], linewidth=0.8, linestyle='--', label='Predicted')
    ax.fill_between(idx, mu[idx] - 1.645 * sigma[idx], mu[idx] + 1.645 * sigma[idx],
                    alpha=0.2, color=IEEE_COLORS['orange'], label='90% PI')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('(a) Prediction with 90% Prediction Interval')
    ax.legend(fontsize=7, ncol=3)

    # 下图: 不确定性分解
    ax = axes[1]
    ax.fill_between(idx, 0, aleatoric[idx], alpha=0.6, color=IEEE_COLORS['blue'], label='Aleatoric')
    ax.fill_between(idx, aleatoric[idx], total[idx], alpha=0.6, color=IEEE_COLORS['orange'], label='Epistemic')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Sample Index')
    ax.set_title('(b) Uncertainty Decomposition')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[保存] 不确定性分解图: {save_path}")


def plot_few_shot_curve(
    few_shot_results: Dict[int, dict],
    save_path: Path,
):
    """Few-shot 性能曲线"""
    setup_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    ctx_sizes = sorted(few_shot_results.keys())
    mapes = [few_shot_results[k]['MAPE'] for k in ctx_sizes]
    coverages = [few_shot_results[k].get('Coverage_90', 0) for k in ctx_sizes]

    ax = axes[0]
    ax.plot(ctx_sizes, mapes, 'o-', color=IEEE_COLORS['blue'], markersize=5, linewidth=1.2)
    ax.set_xlabel('Context Size (# samples)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('(a) MAPE vs Context Size')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ctx_sizes, coverages, 's-', color=IEEE_COLORS['green'], markersize=5, linewidth=1.2)
    ax.axhline(y=90, color=IEEE_COLORS['red'], linestyle='--', linewidth=0.8, label='Target 90%')
    ax.set_xlabel('Context Size (# samples)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('(b) Coverage vs Context Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[保存] Few-shot 曲线: {save_path}")


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
):
    """残差分布直方图 + 正态拟合 + 统计标注"""
    setup_ieee_style()
    residuals = y_true - y_pred

    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    skew_r = float(sp_stats.skew(residuals))
    kurt_r = float(sp_stats.kurtosis(residuals))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # 左图：直方图 + 正态拟合
    ax = axes[0]
    ax.hist(residuals, bins=100, density=True, alpha=0.7,
            color=IEEE_COLORS['blue'], edgecolor='white', linewidth=0.3)
    x_range = np.linspace(mean_r - 4*std_r, mean_r + 4*std_r, 200)
    ax.plot(x_range, sp_stats.norm.pdf(x_range, mean_r, std_r),
            color=IEEE_COLORS['red'], linewidth=1.2, label='Normal fit')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.axvline(x=mean_r, color=IEEE_COLORS['orange'], linestyle='-', linewidth=0.8,
              label=f'Mean={mean_r:.2f}')
    ax.set_xlabel('Residual (kWh)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Residual Distribution')
    ax.legend(fontsize=7)

    # 右图：Q-Q 图
    ax = axes[1]
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, s=2, alpha=0.3, color=IEEE_COLORS['blue'])
    ax.plot(osm, slope * np.array(osm) + intercept,
            color=IEEE_COLORS['red'], linewidth=1.0, label=f'R\u00b2={r_val**2:.4f}')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('(b) Q-Q Plot')
    ax.legend(fontsize=7)

    # 统计标注
    stats_text = (f'Mean={mean_r:.3f}\nStd={std_r:.3f}\n'
                  f'Skew={skew_r:.2f}\nKurt={kurt_r:.1f}')
    fig.text(0.98, 0.02, stats_text, fontsize=7, ha='right', va='bottom',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[保存] 残差分布图: {save_path}")


# ============================================================================
# Few-Shot 评估
# ============================================================================
@torch.no_grad()
def evaluate_few_shot(
    model,
    df: pd.DataFrame,
    test_bs: List[str],
    features: List[str],
    hw_mapper,
    feature_scaler,
    target_mean: float,
    target_std: float,
    device: torch.device,
    context_sizes: List[int] = [5, 10, 20, 50],
) -> Dict[int, dict]:
    """
    Few-shot 评估：对 test BS，使用不同大小的 context set 做预测。
    """
    from sklearn.preprocessing import StandardScaler

    X_raw = df[features].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)
    y_raw = df['energy'].values
    hw_ids_all = hw_mapper.transform_fast(df)

    results = {}
    model.eval()

    for n_ctx in context_sizes:
        all_true = []
        all_pred = []
        all_sigma = []

        for bs_id in test_bs:
            mask = df['bs_id'] == bs_id
            idx = df.index[mask].values
            n = len(idx)
            if n < n_ctx + 5:
                continue

            ctx_idx = idx[:n_ctx]
            tgt_idx = idx[n_ctx:]

            hw = torch.tensor(hw_ids_all[idx[0]], dtype=torch.long).unsqueeze(0).to(device)
            ctx_x = torch.tensor(X_scaled[ctx_idx], dtype=torch.float32).unsqueeze(0).to(device)
            ctx_y_s = torch.tensor(
                (y_raw[ctx_idx] - target_mean) / target_std, dtype=torch.float32
            ).unsqueeze(0).to(device)
            tgt_x = torch.tensor(X_scaled[tgt_idx], dtype=torch.float32).unsqueeze(0).to(device)

            mu, sigma = model.predict(ctx_x, ctx_y_s, tgt_x, hw, n_samples=5)
            mu_np = mu.cpu().numpy().flatten() * target_std + target_mean
            sigma_np = sigma.cpu().numpy().flatten() * target_std

            all_true.extend(y_raw[tgt_idx])
            all_pred.extend(mu_np)
            all_sigma.extend(sigma_np)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_sigma = np.array(all_sigma)

        if len(all_true) > 0:
            lower = all_pred - 1.645 * all_sigma
            upper = all_pred + 1.645 * all_sigma
            metrics = evaluate_predictions(all_true, all_pred, lower, upper)
            results[n_ctx] = metrics
            print(f"  Context={n_ctx:3d}: MAPE={metrics['MAPE']:.2f}%, "
                  f"Coverage={metrics.get('Coverage_90', 0):.1f}%, "
                  f"n={len(all_true)}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("PI-ANP-CQR: Conformal Calibration + Evaluation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 1. 加载模型和数据 ==========
    print("\n>>> 加载模型和数据...")

    # 训练配置
    config_path = MODEL_DIR / "training_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)

    features = saved_config['features']
    train_bs = saved_config['train_bs']
    val_bs = saved_config['val_bs']
    test_bs = saved_config['test_bs']

    # 数据
    df, _ = load_and_prepare_data()

    # Scaler
    with open(MODEL_DIR / "scaler.pkl", 'rb') as f:
        scaler_data = pickle.load(f)
    feature_scaler = scaler_data['feature_scaler']
    target_mean = scaler_data['target_mean']
    target_std = scaler_data['target_std']

    # 硬件映射
    with open(MODEL_DIR / "hw_config_map.pkl", 'rb') as f:
        hw_mapper = pickle.load(f)

    # 模型
    cfg = saved_config['config']
    model = AttentiveNeuralProcess(
        d_in=len(features),
        n_hw_configs=saved_config['n_hw_configs'],
        d_hw=cfg['d_hw'],
        d_hidden=cfg['d_hidden'],
        d_z=cfg['d_z'],
        n_heads=cfg['n_heads'],
        dropout=cfg['dropout'],
    )
    model.load_state_dict(torch.load(MODEL_DIR / "anp_model.pt", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"  模型已加载，参数量: {saved_config['n_params']:,}")

    # ========== 2. MC-Dropout 预测（EnCQR 需要） ==========
    print("\n>>> MC-Dropout 预测 (5 passes)...")
    mu_mean, sigma_mean, mu_ensemble, sigma_ensemble = mc_dropout_predict(
        model, df, features, hw_mapper, feature_scaler,
        target_mean, target_std, device, n_mc=5,
    )

    # ========== 3. CQR 校准 ==========
    print("\n>>> CQR 校准...")

    # 使用 val_bs 作为校准集
    cal_mask = df['bs_id'].isin(val_bs)
    valid_cal = cal_mask & ~np.isnan(mu_mean)

    y_cal = df.loc[valid_cal, 'energy'].values
    mu_cal = mu_mean[valid_cal]
    sigma_cal = sigma_mean[valid_cal]

    # CQR
    cqr = CQRCalibrator(alpha=0.10)
    cqr.fit(y_cal, mu_cal, sigma_cal)

    # EnCQR
    encqr = EnCQRCalibrator(alpha=0.10, n_ensemble=5)
    valid_cal_idx = np.where(valid_cal)[0]
    mu_ens_cal = mu_ensemble[:, valid_cal_idx]
    sigma_ens_cal = sigma_ensemble[:, valid_cal_idx]
    encqr.fit(y_cal, mu_ens_cal, sigma_ens_cal)

    # ========== 4. 在 test_bs 上评估 ==========
    print("\n>>> 评估 (Test BS)...")

    test_mask = df['bs_id'].isin(test_bs)
    valid_test = test_mask & ~np.isnan(mu_mean)

    y_test = df.loc[valid_test, 'energy'].values
    mu_test = mu_mean[valid_test]
    sigma_test = sigma_mean[valid_test]

    # 校准前
    lower_before = mu_test - 1.645 * sigma_test
    upper_before = mu_test + 1.645 * sigma_test
    metrics_before = evaluate_predictions(y_test, mu_test, lower_before, upper_before)

    # CQR 校准后
    lower_cqr, upper_cqr = cqr.predict(mu_test, sigma_test)
    metrics_cqr = evaluate_predictions(y_test, mu_test, lower_cqr, upper_cqr)

    # EnCQR 校准后
    lower_encqr, upper_encqr = encqr.predict(mu_test, sigma_test)
    metrics_encqr = evaluate_predictions(y_test, mu_test, lower_encqr, upper_encqr)

    print(f"\n{'='*60}")
    print("Test BS 评估结果")
    print(f"{'='*60}")
    print(f"{'指标':<25} {'校准前':<12} {'CQR':<12} {'EnCQR':<12}")
    print("-" * 60)
    for key in ['MAPE', 'RMSE', 'MAE', 'R2', 'Coverage_90', 'IntervalWidth_90', 'WinklerScore_90']:
        v0 = metrics_before.get(key, 0)
        v1 = metrics_cqr.get(key, 0)
        v2 = metrics_encqr.get(key, 0)
        print(f"  {key:<23} {v0:<12.4f} {v1:<12.4f} {v2:<12.4f}")
    print(f"{'='*60}")

    # ========== 5. Few-Shot 评估 ==========
    print("\n>>> Few-Shot 评估...")
    few_shot_results = evaluate_few_shot(
        model, df, test_bs, features, hw_mapper,
        feature_scaler, target_mean, target_std, device,
        context_sizes=[5, 10, 20, 50, 100],
    )

    # ========== 6. 保存结果 ==========
    print("\n>>> 保存结果...")

    # 预测结果 CSV
    result_df = df[['time', 'bs_id', 'energy']].copy()
    result_df['predicted'] = mu_mean
    result_df['sigma'] = sigma_mean
    result_df['lower_90'] = np.nan
    result_df['upper_90'] = np.nan

    # 全量 EnCQR 校准
    valid_all = ~np.isnan(mu_mean)
    lower_all, upper_all = encqr.predict(mu_mean[valid_all], sigma_mean[valid_all])
    result_df.loc[result_df.index[valid_all], 'lower_90'] = lower_all
    result_df.loc[result_df.index[valid_all], 'upper_90'] = upper_all

    pred_path = RESULT_DIR / "predictions.csv"
    result_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"  预测结果: {pred_path}")

    # Metrics JSON
    all_metrics = {
        'before_cqr': {k: float(v) for k, v in metrics_before.items()},
        'cqr': {k: float(v) for k, v in metrics_cqr.items()},
        'encqr': {k: float(v) for k, v in metrics_encqr.items()},
        'few_shot': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in few_shot_results.items()},
    }
    metrics_path = RESULT_DIR / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"  评估指标: {metrics_path}")

    # DER 接口演示
    der_results = anp_to_der_results(mu_mean[valid_all], sigma_mean[valid_all], mu_ensemble[:, valid_all])
    der_path = RESULT_DIR / "der_interface_demo.json"
    with open(der_path, 'w') as f:
        json.dump({
            'gamma_sample': der_results['gamma'][:24].tolist(),
            'aleatoric_sample': der_results['aleatoric'][:24].tolist(),
            'epistemic_sample': der_results['epistemic'][:24].tolist(),
            'total_sample': der_results['total'][:24].tolist(),
            'note': '完整数据请使用 anp_to_der_results() 函数',
        }, f, indent=2)
    print(f"  DER 接口演示: {der_path}")

    # ========== 7. 可视化 ==========
    print("\n>>> 生成可视化...")

    # CQR 前后对比
    plot_cqr_comparison(
        metrics_before, metrics_encqr,
        PLOT_DIR / "ieee_pi_anp_cqr_comparison",
    )

    # 不确定性分解
    valid_test_idx = np.where(valid_test)[0]
    plot_uncertainty_decomposition(
        mu_test, sigma_test, mu_ensemble[:, valid_test_idx],
        y_test,
        PLOT_DIR / "ieee_uncertainty_decomposition",
    )

    # Few-shot 曲线
    if few_shot_results:
        plot_few_shot_curve(
            few_shot_results,
            PLOT_DIR / "ieee_few_shot_curve",
        )

    # 方法对比（尝试加载 LightGBM 结果）
    lgbm_metrics = None
    lgbm_path = PROJECT_ROOT / "results" / "quantile_v2" / "metrics_v2.json"
    if lgbm_path.exists():
        with open(lgbm_path, 'r') as f:
            lgbm_metrics = json.load(f)
        print(f"  已加载 LightGBM 基线指标")

    plot_method_comparison(
        metrics_encqr, lgbm_metrics,
        PLOT_DIR / "ieee_pi_anp_vs_lgbm",
    )

    # 残差分布诊断（验证训练后偏差是否改善）
    print("\n>>> 残差分布诊断 (Test BS)...")
    test_residuals = y_test - mu_test
    r_mean = float(np.mean(test_residuals))
    r_std = float(np.std(test_residuals))
    r_skew = float(sp_stats.skew(test_residuals))
    r_kurt = float(sp_stats.kurtosis(test_residuals))
    print(f"  残差均值:   {r_mean:.4f} kWh  (物理基线偏差为 2.91, 期望已缩小)")
    print(f"  残差标准差: {r_std:.4f} kWh")
    print(f"  偏度:       {r_skew:.4f}  (物理基线为 -8.49, 期望已改善)")
    print(f"  峰度:       {r_kurt:.2f}  (物理基线为 221.6, 期望已降低)")

    plot_residual_distribution(
        y_test, mu_test,
        PLOT_DIR / "ieee_residual_distribution",
    )

    # ========== 8. 汇总 ==========
    print(f"\n{'='*60}")
    print("PI-ANP-CQR 评估完成！")
    print(f"{'='*60}")
    print(f"\n结果目录: {RESULT_DIR}")
    print(f"图表目录: {PLOT_DIR}")
    print(f"\n关键指标 (EnCQR, Test BS):")
    for key in ['MAPE', 'R2', 'Coverage_90', 'WinklerScore_90']:
        print(f"  {key}: {metrics_encqr.get(key, 0):.4f}")

    if few_shot_results:
        print(f"\nFew-Shot 性能:")
        for n_ctx, m in sorted(few_shot_results.items()):
            print(f"  Context={n_ctx}: MAPE={m['MAPE']:.2f}%")

    print(f"\n{'='*60}")

    return all_metrics


if __name__ == "__main__":
    metrics = main()
