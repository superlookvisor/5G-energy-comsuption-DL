"""
04_pi_anp_model.py - Physics-Informed Attentive Neural Process (PI-ANP)
基站能耗预测建模项目

功能：
1. 基于 Attentive Neural Process 的元学习架构
2. 每个基站视为一个 task，学习跨基站的预测分布
3. 物理约束 loss 嵌入 3GPP 功率模型先验
4. 输出预测分布 N(μ, σ²)，支持不确定性量化

架构：
  Layer 1: Physics-Informed Loss (物理一致性约束)
  Layer 2: Attentive Neural Process (小样本元学习 + 概率预测)

参考文献：
  - Kim et al., "Attentive Neural Processes", ICLR 2019
  - Amini et al., "Deep Evidential Regression", NeurIPS 2020
  - 3GPP TR 38.864, "Study on network energy savings for NR"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# 路径配置
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "features"
MODEL_DIR = PROJECT_ROOT / "models" / "pi_anp"
RESULT_DIR = PROJECT_ROOT / "results" / "pi_anp_cqr"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 特征配置
# ============================================================================

# ANP 使用的动态特征（Intra-day 全特征）
# 注: 列名已与 feature_data_nogroup.csv 实际列名对齐
ANP_FEATURES = [
    'load_sum', 'load_sum_sq',
    'sum_es1', 'sum_es2', 'sum_es3', 'sum_es4', 'sum_es5', 'sum_es6',
    'current_active_cells',
    'load_sum_lag_1h', 'load_sum_lag_2h', 'load_sum_lag_3h', 'load_sum_lag_24h',
    'load_rolling_24h_mean', 'load_rolling_24h_std', 'load_sum_diff_1h',
    'load_sum_diff_sg', 'load_sum_diff2_sg',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'load_x_pmax', 'load_x_hw_scale', 'bs_power_density', 'freq_normalized',
]

# Day-ahead 特征（不含当前 load 的子集）
ANP_DAY_AHEAD_FEATURES = [
    'load_sum_lag_1h', 'load_sum_lag_2h', 'load_sum_lag_3h', 'load_sum_lag_24h',
    'load_rolling_24h_mean', 'load_rolling_24h_std',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'freq_normalized',
]

# 物理模型预计算基准值（P_base，来自 residual_data.csv）
PHYSICS_RAW_FEATURES = ['P_base']


# ============================================================================
# 硬件配置映射
# ============================================================================
class HardwareConfigMapper:
    """
    将 (ru_type, frequency, bandwidth, antennas) 硬件组合映射为整数 ID。
    用于 nn.Embedding 的索引查找。
    """

    def __init__(self):
        self.config_to_id: Dict[tuple, int] = {}
        self.id_to_config: Dict[int, tuple] = {}
        self.n_configs: int = 0

    def fit(self, df: pd.DataFrame) -> 'HardwareConfigMapper':
        configs = df.groupby(
            ['ru_type', 'bs_frequency_mean', 'bs_bandwidth_mean', 'bs_antennas_sum']
        ).ngroups
        unique_configs = (
            df.groupby(['ru_type', 'bs_frequency_mean', 'bs_bandwidth_mean', 'bs_antennas_sum'])
            .size()
            .reset_index()
            .drop(columns=0)
        )
        for idx, row in unique_configs.iterrows():
            key = (row['ru_type'], row['bs_frequency_mean'], row['bs_bandwidth_mean'], row['bs_antennas_sum'])
            self.config_to_id[key] = idx
            self.id_to_config[idx] = key
        self.n_configs = len(self.config_to_id)
        print(f"[HardwareConfigMapper] {self.n_configs} 种硬件配置")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        ids = np.zeros(len(df), dtype=np.int64)
        for i, (_, row) in enumerate(df.iterrows()):
            key = (row['ru_type'], row['bs_frequency_mean'], row['bs_bandwidth_mean'], row['bs_antennas_sum'])
            ids[i] = self.config_to_id.get(key, 0)
        return ids

    def transform_fast(self, df: pd.DataFrame) -> np.ndarray:
        """向量化版本，避免逐行迭代"""
        keys = list(zip(
            df['ru_type'].values,
            df['bs_frequency_mean'].values,
            df['bs_bandwidth_mean'].values,
            df['bs_antennas_sum'].values,
        ))
        ids = np.array([self.config_to_id.get(k, 0) for k in keys], dtype=np.int64)
        return ids


# ============================================================================
# Episode Dataset
# ============================================================================
class BSEpisodeDataset(Dataset):
    """
    基站 Episode 数据集。
    每个 episode = 一个基站的所有数据，随机划分为 context / target。
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        hw_ids: np.ndarray,
        bs_indices: Dict[str, np.ndarray],
        physics_features: np.ndarray,
        context_size_range: Tuple[int, int] = (10, 50),
        mode: str = 'random',
        hours: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
    ):
        """
        参数：
            features: (N, d_in) 归一化后的特征
            targets: (N,) 归一化后的目标
            hw_ids: (N,) 硬件配置 ID
            bs_indices: {bs_id: array of row indices}
            physics_features: (N, 1) 物理基准预测值 [P_base]
            context_size_range: context set 大小范围
            mode: 'random' | 'day_ahead' | 'intra_day'
            hours: (N,) 小时值（mode != 'random' 时需要）
            dates: (N,) 日期值（mode == 'day_ahead' 时需要）
        """
        self.features = features
        self.targets = targets
        self.hw_ids = hw_ids
        self.bs_list = list(bs_indices.keys())
        self.bs_indices = bs_indices
        self.physics_features = physics_features
        self.context_min, self.context_max = context_size_range
        self.mode = mode
        self.hours = hours
        self.dates = dates

    def __len__(self):
        return len(self.bs_list)

    def __getitem__(self, idx):
        bs_id = self.bs_list[idx]
        indices = self.bs_indices[bs_id]
        n = len(indices)

        if self.mode == 'random':
            ctx_max = min(self.context_max, n - 5)
            ctx_min = min(self.context_min, ctx_max)
            if ctx_min < 3:
                ctx_min = 3
            if ctx_max < ctx_min:
                ctx_max = ctx_min
            n_ctx = np.random.randint(ctx_min, ctx_max + 1)
            perm = np.random.permutation(n)
            ctx_idx = indices[perm[:n_ctx]]
            tgt_idx = indices[perm[n_ctx:]]

        elif self.mode == 'intra_day':
            hours = self.hours[indices]
            ctx_mask = hours <= 18
            tgt_mask = hours > 18
            ctx_idx = indices[ctx_mask]
            tgt_idx = indices[tgt_mask]
            if len(tgt_idx) == 0:
                # fallback: random split
                perm = np.random.permutation(n)
                split = max(3, n // 2)
                ctx_idx = indices[perm[:split]]
                tgt_idx = indices[perm[split:]]

        elif self.mode == 'day_ahead':
            dates = self.dates[indices]
            unique_dates = np.unique(dates)
            if len(unique_dates) >= 2:
                last_date = unique_dates[-1]
                ctx_mask = dates != last_date
                tgt_mask = dates == last_date
                ctx_idx = indices[ctx_mask]
                tgt_idx = indices[tgt_mask]
            else:
                perm = np.random.permutation(n)
                split = max(3, n // 2)
                ctx_idx = indices[perm[:split]]
                tgt_idx = indices[perm[split:]]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 确保 target 不为空
        if len(tgt_idx) == 0:
            perm = np.random.permutation(n)
            split = max(3, n // 2)
            ctx_idx = indices[perm[:split]]
            tgt_idx = indices[perm[split:]]

        hw_id = self.hw_ids[indices[0]]

        return {
            'context_x': torch.tensor(self.features[ctx_idx], dtype=torch.float32),
            'context_y': torch.tensor(self.targets[ctx_idx], dtype=torch.float32),
            'target_x': torch.tensor(self.features[tgt_idx], dtype=torch.float32),
            'target_y': torch.tensor(self.targets[tgt_idx], dtype=torch.float32),
            'hw_id': torch.tensor(hw_id, dtype=torch.long),
            'physics_ctx': torch.tensor(self.physics_features[ctx_idx], dtype=torch.float32),
            'physics_tgt': torch.tensor(self.physics_features[tgt_idx], dtype=torch.float32),
        }


def collate_episodes(batch: List[dict]) -> dict:
    """
    自定义 collate：将变长 episodes pad 到 batch 内最大长度。
    返回 padded tensors + masks。
    """
    max_ctx = max(b['context_x'].shape[0] for b in batch)
    max_tgt = max(b['target_x'].shape[0] for b in batch)
    d_in = batch[0]['context_x'].shape[1]
    d_phys = batch[0]['physics_ctx'].shape[1]
    B = len(batch)

    ctx_x = torch.zeros(B, max_ctx, d_in)
    ctx_y = torch.zeros(B, max_ctx)
    tgt_x = torch.zeros(B, max_tgt, d_in)
    tgt_y = torch.zeros(B, max_tgt)
    hw_ids = torch.zeros(B, dtype=torch.long)
    ctx_mask = torch.zeros(B, max_ctx, dtype=torch.bool)
    tgt_mask = torch.zeros(B, max_tgt, dtype=torch.bool)
    phys_ctx = torch.zeros(B, max_ctx, d_phys)
    phys_tgt = torch.zeros(B, max_tgt, d_phys)

    for i, b in enumerate(batch):
        nc = b['context_x'].shape[0]
        nt = b['target_x'].shape[0]
        ctx_x[i, :nc] = b['context_x']
        ctx_y[i, :nc] = b['context_y']
        tgt_x[i, :nt] = b['target_x']
        tgt_y[i, :nt] = b['target_y']
        hw_ids[i] = b['hw_id']
        ctx_mask[i, :nc] = True
        tgt_mask[i, :nt] = True
        phys_ctx[i, :nc] = b['physics_ctx']
        phys_tgt[i, :nt] = b['physics_tgt']

    return {
        'context_x': ctx_x,
        'context_y': ctx_y,
        'target_x': tgt_x,
        'target_y': tgt_y,
        'hw_ids': hw_ids,
        'ctx_mask': ctx_mask,
        'tgt_mask': tgt_mask,
        'physics_ctx': phys_ctx,
        'physics_tgt': phys_tgt,
    }


# ============================================================================
# ANP 模型组件
# ============================================================================

class HardwareEmbedding(nn.Module):
    """硬件配置嵌入层"""
    def __init__(self, n_configs: int, d_hw: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(n_configs, d_hw)

    def forward(self, hw_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(hw_ids)


class DeterministicEncoder(nn.Module):
    """
    确定性编码器：将 context (x, y) 编码为 key-value 对。
    """
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        # 输入: (x, y) -> d_in + 1
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.key_proj = nn.Linear(d_hidden, d_hidden)
        self.value_proj = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N_ctx, d_in)
        y: (B, N_ctx)
        返回: keys (B, N_ctx, d_hidden), values (B, N_ctx, d_hidden)
        """
        xy = torch.cat([x, y.unsqueeze(-1)], dim=-1)
        h = self.mlp(xy)
        return self.key_proj(h), self.value_proj(h)


class QueryEncoder(nn.Module):
    """查询编码器：将 target x 编码为 query 向量"""
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力"""
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        queries: (B, N_tgt, d_model)
        keys: (B, N_ctx, d_model)
        values: (B, N_ctx, d_model)
        mask: (B, N_ctx) bool — True = valid
        返回: (B, N_tgt, d_model)
        """
        B, N_tgt, d = queries.shape
        N_ctx = keys.shape[1]

        # 多头拆分: (B, N, d) -> (B, n_heads, N, d_k)
        Q = queries.view(B, N_tgt, self.n_heads, self.d_k).transpose(1, 2)
        K = keys.view(B, N_ctx, self.n_heads, self.d_k).transpose(1, 2)
        V = values.view(B, N_ctx, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力得分: (B, n_heads, N_tgt, N_ctx)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (B, N_ctx) -> (B, 1, 1, N_ctx)
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, V)  # (B, n_heads, N_tgt, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N_tgt, d)
        return out


class LatentEncoder(nn.Module):
    """
    潜变量编码器：将 context (或 context+target) 聚合为全局潜变量 z。
    """
    def __init__(self, d_in: int, d_hidden: int = 128, d_z: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.mu_proj = nn.Linear(d_hidden, d_z)
        self.logvar_proj = nn.Linear(d_hidden, d_z)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, d_in)
        y: (B, N)
        mask: (B, N) bool
        返回: mu (B, d_z), logvar (B, d_z)
        """
        xy = torch.cat([x, y.unsqueeze(-1)], dim=-1)
        h = self.mlp(xy)  # (B, N, d_hidden)

        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
            counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            h_agg = h.sum(dim=1) / counts
        else:
            h_agg = h.mean(dim=1)

        mu = self.mu_proj(h_agg)
        logvar = self.logvar_proj(h_agg)
        return mu, logvar


class Decoder(nn.Module):
    """
    解码器：从 (target_x, r_det, z) 预测 N(μ, σ²)。
    包含可学习偏置修正，弥补物理模型的系统性低估（残差均值=2.91 kWh）。
    """
    def __init__(self, d_in: int, d_det: int = 128, d_z: int = 64, d_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in + d_det + d_z, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 64),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(64, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        # 可学习偏置修正（弥补物理模型 2.91 kWh 系统性低估）
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, target_x: torch.Tensor, r_det: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        target_x: (B, N_tgt, d_in)
        r_det: (B, N_tgt, d_det)
        z: (B, d_z) -> broadcast to (B, N_tgt, d_z)
        返回: mu (B, N_tgt), sigma (B, N_tgt)
        """
        N_tgt = target_x.shape[1]
        z_expanded = z.unsqueeze(1).expand(-1, N_tgt, -1)

        inp = torch.cat([target_x, r_det, z_expanded], dim=-1)
        h = self.mlp(inp)
        mu = self.mu_head(h).squeeze(-1) + self.bias     # 偏置修正
        sigma = self.sigma_head(h).squeeze(-1) + 1e-4  # 最小标准差
        return mu, sigma


class AttentiveNeuralProcess(nn.Module):
    """
    Attentive Neural Process (ANP) 完整模型

    包含：
    - 硬件嵌入层
    - 确定性路径（交叉注意力）
    - 潜变量路径（全局 z）
    - 解码器
    """

    def __init__(
        self,
        d_in: int,
        n_hw_configs: int,
        d_hw: int = 16,
        d_hidden: int = 128,
        d_z: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_total = d_in + d_hw  # 特征 + 硬件嵌入

        # 硬件嵌入
        self.hw_embed = HardwareEmbedding(n_hw_configs, d_hw)

        # 确定性路径
        self.det_encoder = DeterministicEncoder(self.d_total, d_hidden)
        self.query_encoder = QueryEncoder(self.d_total, d_hidden)
        self.cross_attn = MultiHeadCrossAttention(d_hidden, n_heads, dropout)

        # 潜变量路径
        self.latent_encoder = LatentEncoder(self.d_total, d_hidden, d_z)
        self.prior_encoder = LatentEncoder(self.d_total, d_hidden, d_z)

        # 解码器
        self.decoder = Decoder(self.d_total, d_hidden, d_z, d_hidden)

    def _prepend_hw(
        self, x: torch.Tensor, hw_ids: torch.Tensor
    ) -> torch.Tensor:
        """在特征前拼接硬件嵌入"""
        B, N, _ = x.shape
        hw = self.hw_embed(hw_ids)  # (B, d_hw)
        hw_expanded = hw.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_hw)
        return torch.cat([x, hw_expanded], dim=-1)

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        hw_ids: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        target_y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        训练时: 提供 target_y 用于计算 posterior latent
        推理时: 不提供 target_y，使用 prior latent

        返回:
            mu: (B, N_tgt) 预测均值
            sigma: (B, N_tgt) 预测标准差
            kl: scalar KL 散度
            prior_mu, prior_logvar: 先验参数
            posterior_mu, posterior_logvar: 后验参数（仅训练时）
        """
        # 拼接硬件嵌入
        ctx_x_hw = self._prepend_hw(context_x, hw_ids)
        tgt_x_hw = self._prepend_hw(target_x, hw_ids)

        # ====== 确定性路径 ======
        keys, values = self.det_encoder(ctx_x_hw, context_y)
        queries = self.query_encoder(tgt_x_hw)
        r_det = self.cross_attn(queries, keys, values, mask=ctx_mask)

        # ====== 潜变量路径 ======
        # Prior: 仅使用 context
        prior_mu, prior_logvar = self.prior_encoder(ctx_x_hw, context_y, ctx_mask)

        if target_y is not None:
            # Posterior: 使用 context + target
            all_x = torch.cat([ctx_x_hw, tgt_x_hw], dim=1)
            all_y = torch.cat([context_y, target_y], dim=1)
            if ctx_mask is not None and tgt_mask is not None:
                all_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
            else:
                all_mask = None
            post_mu, post_logvar = self.latent_encoder(all_x, all_y, all_mask)

            # 重参数化采样 z ~ q(z | C, T)
            std = torch.exp(0.5 * post_logvar)
            eps = torch.randn_like(std)
            z = post_mu + eps * std

            # KL divergence
            kl = -0.5 * torch.sum(
                1 + post_logvar - prior_logvar
                - (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp(),
                dim=-1,
            ).mean()
        else:
            # 推理时使用 prior
            std = torch.exp(0.5 * prior_logvar)
            eps = torch.randn_like(std)
            z = prior_mu + eps * std
            kl = torch.tensor(0.0, device=context_x.device)
            post_mu, post_logvar = prior_mu, prior_logvar

        # ====== 解码 ======
        mu, sigma = self.decoder(tgt_x_hw, r_det, z)

        return {
            'mu': mu,
            'sigma': sigma,
            'kl': kl,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
            'posterior_mu': post_mu,
            'posterior_logvar': post_logvar,
        }

    @torch.no_grad()
    def predict(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        hw_ids: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理：返回预测均值和标准差。
        n_samples > 1 时，对潜变量采样多次并聚合。
        """
        self.eval()
        mus = []
        sigmas = []

        for _ in range(n_samples):
            out = self.forward(
                context_x, context_y, target_x, hw_ids,
                ctx_mask, tgt_mask, target_y=None,
            )
            mus.append(out['mu'])
            sigmas.append(out['sigma'])

        mu_stack = torch.stack(mus, dim=0)
        sigma_stack = torch.stack(sigmas, dim=0)

        # 混合高斯的均值和方差
        mu_mean = mu_stack.mean(dim=0)
        # Var = E[sigma^2] + Var[mu]
        aleatoric = (sigma_stack ** 2).mean(dim=0)
        epistemic = mu_stack.var(dim=0)

        total_sigma = torch.sqrt(aleatoric + epistemic)
        return mu_mean, total_sigma


# ============================================================================
# Physics Constraint
# ============================================================================
class PhysicsConstraint(nn.Module):
    """
    物理约束模块（BS 级物理代理版）

    使用 03_physics_baseline.py 预计算的 P_base 值作为物理先验。
    P_base 通过 load_and_prepare_data() 合并自 residual_data.csv。

    物理语义：P_base = P_static + P_dynamic - P_save
        - P_static: 硬件规模决定的静态功耗（α_ant=2.26, α_cells=10.72）
        - P_dynamic: 负载相关的动态功耗（β=1.69）
        - P_save: 节能模式削减的功耗（ES6 最显著 γ=1.12, ES3 次之 γ=0.88）
    """

    def __init__(self, physics_pkl_path: Optional[Path] = None):
        super().__init__()
        self.available = False

        if physics_pkl_path is not None and physics_pkl_path.exists():
            with open(physics_pkl_path, 'rb') as f:
                model_data = pickle.load(f)

            # 记录物理模型参数用于日志（参数本身不参与训练）
            alpha = model_data['alpha']
            beta = model_data['beta']
            gamma = model_data['gamma']
            es_cols = model_data['es_col_names']

            self.available = True
            print(f"[PhysicsConstraint] 已加载物理基准模型参数")
            print(f"  静态: intercept={alpha['intercept']:.2f}, "
                  f"antennas={alpha['total_antennas']:.4f}, "
                  f"num_cells={alpha['num_cells']:.4f}")
            print(f"  动态: β={beta}")
            # 按 gamma 排序显示节能权重
            es_ranking = sorted(zip(es_cols, gamma), key=lambda x: -x[1])
            print(f"  节能权重（降序）:")
            for col, g in es_ranking:
                print(f"    {col}: γ={g:.4f}")

    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        """
        physics_features: (B, N, 1) = [P_base]
        返回: P_physics (B, N) — 物理基准模型的预测值（原始尺度 kWh）
        """
        if not self.available:
            return torch.zeros(physics_features.shape[0], physics_features.shape[1],
                             device=physics_features.device)

        # 直接提取预计算的 P_base
        return physics_features[..., 0]


# ============================================================================
# 训练器
# ============================================================================
@dataclass
class TrainingConfig:
    """训练超参数"""
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    patience: int = 20

    # Lambda 调度
    kl_anneal_epochs: int = 50
    kl_lambda_min: float = 0.01
    kl_lambda_max: float = 1.0
    physics_lambda_init: float = 1.0
    physics_lambda_min: float = 0.1

    # 模型架构
    d_hw: int = 16
    d_hidden: int = 128
    d_z: int = 64
    n_heads: int = 4
    dropout: float = 0.1

    # 数据
    context_size_range: Tuple[int, int] = (10, 50)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


class PIANPTrainer:
    """Physics-Informed ANP 训练器"""

    def __init__(
        self,
        model: AttentiveNeuralProcess,
        physics: PhysicsConstraint,
        config: TrainingConfig,
        device: torch.device,
        target_scaler_mean: float = 0.0,
        target_scaler_std: float = 1.0,
    ):
        self.model = model.to(device)
        self.physics = physics.to(device)
        self.config = config
        self.device = device
        self.target_mean = target_scaler_mean
        self.target_std = target_scaler_std

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _get_kl_lambda(self, epoch: int) -> float:
        cfg = self.config
        if epoch < cfg.kl_anneal_epochs:
            t = epoch / cfg.kl_anneal_epochs
            return cfg.kl_lambda_min + t * (cfg.kl_lambda_max - cfg.kl_lambda_min)
        return cfg.kl_lambda_max

    def _get_physics_lambda(self, epoch: int) -> float:
        cfg = self.config
        t = epoch / cfg.epochs
        # Cosine decay
        return cfg.physics_lambda_min + 0.5 * (cfg.physics_lambda_init - cfg.physics_lambda_min) * (
            1 + np.cos(np.pi * t)
        )

    def compute_loss(
        self, batch: dict, epoch: int
    ) -> Tuple[torch.Tensor, dict]:
        """计算总 loss = NLL + λ_kl * KL + λ_physics * Physics"""
        ctx_x = batch['context_x'].to(self.device)
        ctx_y = batch['context_y'].to(self.device)
        tgt_x = batch['target_x'].to(self.device)
        tgt_y = batch['target_y'].to(self.device)
        hw_ids = batch['hw_ids'].to(self.device)
        ctx_mask = batch['ctx_mask'].to(self.device)
        tgt_mask = batch['tgt_mask'].to(self.device)
        phys_tgt = batch['physics_tgt'].to(self.device)

        # 前向传播
        out = self.model(ctx_x, ctx_y, tgt_x, hw_ids, ctx_mask, tgt_mask, tgt_y)
        mu = out['mu']
        sigma = out['sigma']
        kl = out['kl']

        # ====== Huber NLL Loss ======
        # 残差分析发现极端离群值（偏度=-8.49, 峰度=221.6）
        # Huber化 NLL：标准化残差超过 δ 时用线性惩罚替代平方，抑制尾部主导
        residual = tgt_y - mu
        abs_r = residual.abs()
        delta_huber = 3.0  # Huber 阈值（标准化单位）
        huber_sq = torch.where(
            abs_r <= delta_huber,
            residual ** 2,
            2 * delta_huber * abs_r - delta_huber ** 2,
        )
        nll = 0.5 * (torch.log(sigma ** 2) + huber_sq / (sigma ** 2))
        # 仅对有效 target 计算
        nll = (nll * tgt_mask.float()).sum() / tgt_mask.float().sum().clamp(min=1)

        # ====== Physics Loss ======
        physics_loss = torch.tensor(0.0, device=self.device)
        if self.physics.available:
            P_phys = self.physics(phys_tgt)  # (B, N_tgt) 原始尺度 P_base
            # 将 mu 反归一化到原始尺度
            mu_orig = mu * self.target_std + self.target_mean
            phys_err = (mu_orig - P_phys) ** 2
            physics_loss = (phys_err * tgt_mask.float()).sum() / tgt_mask.float().sum().clamp(min=1)
            # 除以 target_std^2 使其与 NLL 量级匹配
            physics_loss = physics_loss / (self.target_std ** 2 + 1e-8)

        # ====== Lambda 调度 ======
        lam_kl = self._get_kl_lambda(epoch)
        lam_phys = self._get_physics_lambda(epoch)

        total = nll + lam_kl * kl + lam_phys * physics_loss

        info = {
            'total': total.item(),
            'nll': nll.item(),
            'kl': kl.item(),
            'physics': physics_loss.item(),
            'lam_kl': lam_kl,
            'lam_phys': lam_phys,
        }
        return total, info

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, _ = self.compute_loss(batch, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss, _ = self.compute_loss(batch, epoch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict:
        """完整训练循环"""
        print(f"\n{'='*60}")
        print("PI-ANP 训练")
        print(f"{'='*60}")
        print(f"  设备: {self.device}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  学习率: {self.config.lr}")

        best_state = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                lam_kl = self._get_kl_lambda(epoch)
                lam_phys = self._get_physics_lambda(epoch)
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Best: {self.best_val_loss:.4f} | "
                    f"λ_kl={lam_kl:.3f} λ_phys={lam_phys:.3f} lr={lr:.2e}"
                )

            if self.patience_counter >= self.config.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        # 恢复最优权重
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n  已恢复最优模型 (val_loss={self.best_val_loss:.4f})")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
        }


# ============================================================================
# 数据准备辅助函数
# ============================================================================
def load_and_prepare_data(
    feature_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """加载并准备数据，合并物理基准 P_base"""
    if feature_path is None:
        feature_path = FEATURE_DIR / "feature_data_nogroup.csv"

    df = pd.read_csv(feature_path, encoding='utf-8-sig')
    df['time'] = pd.to_datetime(df['time'])
    print(f"[加载数据] {len(df):,} 条记录, {df['bs_id'].nunique()} 个基站")

    # 合并物理基准 P_base（来自 03_physics_baseline.py 的残差分析）
    residual_path = FEATURE_DIR / "residual_data.csv"
    if residual_path.exists():
        df_res = pd.read_csv(residual_path, encoding='utf-8-sig')
        df_res['time'] = pd.to_datetime(df_res['time'])
        df_res = df_res[['time', 'bs_id', 'P_base']].copy()
        n_before = len(df)
        df = df.merge(df_res, on=['time', 'bs_id'], how='left')
        matched = df['P_base'].notna().sum()
        print(f"  [P_base 合并] 匹配 {matched:,}/{len(df):,} 条 (merge前={n_before:,})")
        # 未匹配部分用能耗中位数填充（合理 fallback）
        if df['P_base'].isna().any():
            df['P_base'] = df['P_base'].fillna(df['energy'].median())
    else:
        print(f"  [警告] 未找到 residual_data.csv，P_base 将使用 energy 均值替代")
        df['P_base'] = df['energy'].mean()

    # 确定可用特征
    available_features = [f for f in ANP_FEATURES if f in df.columns]
    missing = [f for f in ANP_FEATURES if f not in df.columns]
    if missing:
        print(f"  [警告] 缺失特征: {missing}")
    print(f"  可用特征: {len(available_features)}")

    return df, available_features


def split_base_stations(
    df: pd.DataFrame, config: TrainingConfig
) -> Tuple[List[str], List[str], List[str]]:
    """按基站划分 train/val/test"""
    np.random.seed(config.seed)
    bs_ids = df['bs_id'].unique()
    np.random.shuffle(bs_ids)

    n = len(bs_ids)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_bs = bs_ids[:n_train].tolist()
    val_bs = bs_ids[n_train:n_train + n_val].tolist()
    test_bs = bs_ids[n_train + n_val:].tolist()

    print(f"  基站划分: train={len(train_bs)}, val={len(val_bs)}, test={len(test_bs)}")
    return train_bs, val_bs, test_bs


def build_bs_indices(df: pd.DataFrame, bs_list: List[str]) -> Dict[str, np.ndarray]:
    """构建基站 -> 行索引的映射"""
    mask = df['bs_id'].isin(bs_list)
    sub = df[mask]
    indices = {}
    for bs_id, group in sub.groupby('bs_id'):
        indices[bs_id] = group.index.values
    return indices


def create_datasets(
    df: pd.DataFrame,
    features: List[str],
    train_bs: List[str],
    val_bs: List[str],
    test_bs: List[str],
    hw_mapper: HardwareConfigMapper,
    config: TrainingConfig,
    mode: str = 'random',
) -> Tuple[BSEpisodeDataset, BSEpisodeDataset, BSEpisodeDataset,
           StandardScaler, float, float]:
    """创建数据集"""
    # 提取特征矩阵
    X_raw = df[features].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = df['energy'].values.astype(np.float32)
    hw_ids = hw_mapper.transform_fast(df)

    # 物理特征（P_base，保留原始尺度用于 Physics Loss）
    phys_cols = [c for c in PHYSICS_RAW_FEATURES if c in df.columns]
    physics_raw = df[phys_cols].values.astype(np.float32)
    # 确保至少有 1 列（P_base）
    if physics_raw.shape[1] == 0:
        physics_raw = np.zeros((len(df), 1), dtype=np.float32)

    # Scaler: 仅在 train 上 fit
    train_mask = df['bs_id'].isin(train_bs)
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_raw[train_mask])
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)

    target_mean = float(y_raw[train_mask].mean())
    target_std = float(y_raw[train_mask].std())
    y_scaled = ((y_raw - target_mean) / target_std).astype(np.float32)

    # 离群值裁剪（残差分析发现 kurtosis=221.6，极端值影响训练稳定性）
    clip_threshold = 4.0  # ±4σ
    y_clipped = np.clip(y_scaled, -clip_threshold, clip_threshold)
    n_clipped = int(np.sum(y_scaled != y_clipped))
    if n_clipped > 0:
        print(f"  [离群值裁剪] {n_clipped:,} 个样本被裁剪至 ±{clip_threshold}σ 范围")
    y_scaled = y_clipped

    # 时间信息
    hours = df['time'].dt.hour.values
    dates = df['time'].dt.date.values

    # 构建索引
    train_indices = build_bs_indices(df, train_bs)
    val_indices = build_bs_indices(df, val_bs)
    test_indices = build_bs_indices(df, test_bs)

    kwargs = dict(
        features=X_scaled,
        targets=y_scaled,
        hw_ids=hw_ids,
        physics_features=physics_raw,
        context_size_range=config.context_size_range,
        mode=mode,
        hours=hours,
        dates=dates,
    )

    train_ds = BSEpisodeDataset(bs_indices=train_indices, **kwargs)
    val_ds = BSEpisodeDataset(bs_indices=val_indices, **kwargs)
    test_ds = BSEpisodeDataset(bs_indices=test_indices, **kwargs)

    return train_ds, val_ds, test_ds, feature_scaler, target_mean, target_std


# ============================================================================
# 全量预测（用于评估和 CQR 校准）
# ============================================================================
@torch.no_grad()
def predict_all(
    model: AttentiveNeuralProcess,
    df: pd.DataFrame,
    features: List[str],
    hw_mapper: HardwareConfigMapper,
    feature_scaler: StandardScaler,
    target_mean: float,
    target_std: float,
    device: torch.device,
    n_latent_samples: int = 5,
    context_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    对所有基站做预测：每个 BS 的前 context_ratio 数据作为 context，
    其余作为 target。返回每条记录的预测。
    """
    model.eval()

    X_raw = df[features].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)
    y_raw = df['energy'].values
    hw_ids = hw_mapper.transform_fast(df)

    mu_all = np.full(len(df), np.nan)
    sigma_all = np.full(len(df), np.nan)

    for bs_id, group in df.groupby('bs_id'):
        idx = group.index.values
        n = len(idx)
        if n < 5:
            continue

        n_ctx = max(3, int(n * context_ratio))
        ctx_idx = idx[:n_ctx]
        # 预测所有点（包括 context，用于 CQR 校准）
        all_idx = idx

        hw = torch.tensor(hw_ids[idx[0]], dtype=torch.long).unsqueeze(0).to(device)
        ctx_x = torch.tensor(X_scaled[ctx_idx], dtype=torch.float32).unsqueeze(0).to(device)
        ctx_y_scaled = torch.tensor(
            (y_raw[ctx_idx] - target_mean) / target_std,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        tgt_x = torch.tensor(X_scaled[all_idx], dtype=torch.float32).unsqueeze(0).to(device)

        mu, sigma = model.predict(ctx_x, ctx_y_scaled, tgt_x, hw, n_samples=n_latent_samples)

        # 反归一化
        mu_orig = mu.cpu().numpy().flatten() * target_std + target_mean
        sigma_orig = sigma.cpu().numpy().flatten() * target_std

        mu_all[all_idx] = mu_orig
        sigma_all[all_idx] = sigma_orig

    result = df[['time', 'bs_id', 'energy']].copy()
    result['predicted'] = mu_all
    result['sigma'] = sigma_all
    result['lower_90'] = mu_all - 1.645 * sigma_all
    result['upper_90'] = mu_all + 1.645 * sigma_all

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("PI-ANP: Physics-Informed Attentive Neural Process")
    print("=" * 60)

    # 设置随机种子
    config = TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 1. 加载数据
    df, features = load_and_prepare_data()

    # 2. 硬件配置映射
    hw_mapper = HardwareConfigMapper()
    hw_mapper.fit(df)

    # 3. 划分基站
    train_bs, val_bs, test_bs = split_base_stations(df, config)

    # 4. 创建数据集
    train_ds, val_ds, test_ds, feature_scaler, target_mean, target_std = create_datasets(
        df, features, train_bs, val_bs, test_bs, hw_mapper, config, mode='random'
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_episodes, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_episodes, num_workers=0,
    )

    print(f"\n  训练 episodes: {len(train_ds)}")
    print(f"  验证 episodes: {len(val_ds)}")
    print(f"  测试 episodes: {len(test_ds)}")
    print(f"  特征维度: {len(features)}")

    # 5. 创建模型
    model = AttentiveNeuralProcess(
        d_in=len(features),
        n_hw_configs=hw_mapper.n_configs,
        d_hw=config.d_hw,
        d_hidden=config.d_hidden,
        d_z=config.d_z,
        n_heads=config.n_heads,
        dropout=config.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {n_params:,}")

    # 6. 物理约束
    physics_path = PROJECT_ROOT / "models" / "physics_baseline.pkl"
    physics = PhysicsConstraint(physics_path)

    # 7. 训练
    trainer = PIANPTrainer(
        model, physics, config, device,
        target_scaler_mean=target_mean,
        target_scaler_std=target_std,
    )
    training_log = trainer.train(train_loader, val_loader)

    # 8. 全量预测
    print("\n>>> 生成全量预测...")
    result_df = predict_all(
        model, df, features, hw_mapper, feature_scaler,
        target_mean, target_std, device, n_latent_samples=5,
    )

    # 过滤有效预测
    valid_mask = ~result_df['predicted'].isna()
    y_true = result_df.loc[valid_mask, 'energy'].values
    y_pred = result_df.loc[valid_mask, 'predicted'].values
    y_lower = result_df.loc[valid_mask, 'lower_90'].values
    y_upper = result_df.loc[valid_mask, 'upper_90'].values

    # 简单评估
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100
    width = np.mean(y_upper - y_lower)

    print(f"\n{'='*60}")
    print("PI-ANP 初始评估（CQR校准前）")
    print(f"{'='*60}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  Coverage_90: {coverage:.2f}%")
    print(f"  IntervalWidth_90: {width:.2f}")
    print(f"{'='*60}")

    # 9. 保存
    # 模型
    model_path = MODEL_DIR / "anp_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[保存] 模型: {model_path}")

    # Scaler
    scaler_path = MODEL_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_mean': target_mean,
            'target_std': target_std,
        }, f)
    print(f"[保存] Scaler: {scaler_path}")

    # 硬件映射
    hw_path = MODEL_DIR / "hw_config_map.pkl"
    with open(hw_path, 'wb') as f:
        pickle.dump(hw_mapper, f)
    print(f"[保存] 硬件映射: {hw_path}")

    # 训练配置和日志
    config_path = MODEL_DIR / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config.to_dict(),
            'features': features,
            'n_params': n_params,
            'n_hw_configs': hw_mapper.n_configs,
            'training_log': {
                'epochs_trained': training_log['epochs_trained'],
                'best_val_loss': training_log['best_val_loss'],
            },
            'initial_metrics': {
                'MAPE': mape, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'Coverage_90': coverage, 'IntervalWidth_90': width,
            },
            'train_bs': train_bs,
            'val_bs': val_bs,
            'test_bs': test_bs,
        }, f, indent=2, ensure_ascii=False)
    print(f"[保存] 配置: {config_path}")

    # 预测结果
    pred_path = RESULT_DIR / "predictions_raw.csv"
    result_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"[保存] 原始预测: {pred_path}")

    print(f"\n{'='*60}")
    print("PI-ANP 训练完成！")
    print("下一步：运行 07_pi_anp_cqr_evaluation.py 进行 CQR 校准和评估")
    print(f"{'='*60}")

    return model, result_df


if __name__ == "__main__":
    model, result_df = main()
