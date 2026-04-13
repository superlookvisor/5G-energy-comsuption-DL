"""
02_feature_engineering.py - 特征工程（PI-ANP 物理机理版）
基站能耗预测建模项目

输入：01_preprocess.py 输出的 BS 级已聚合数据（merged_data.csv）

功能：
1. 静态特征：对应 P_common 和 Σ P_static 的"规模底座"
   - total_p_max, total_antennas, num_cells → 基站硬件规模
   - dominant_ru one-hot 编码 → 设备类型
2. 动态物理特征：对应 Σ P_dynamic 的时序驱动项
   - dyn_proxy_sum → 已加权的总动态功耗代理（关键输入）
   - es{k}_proxy_sum → 各节能模式的绝对节能规模代理
3. 时序统计特征：基于 dyn_proxy_sum 的滞后与滚动统计
   （不再使用原始 load，因为 dyn_proxy_sum 是更准确的代理变量）
4. PI-ANP 专用输出：区分
   - context_features：静态硬件特征 → 送入 ANP Context Encoder
   - dynamic_features：时序驱动特征 → 送入 ANP Target 序列输入
   - pi_constraint_features：标注哪些特征应施加梯度方向约束

物理约束（在后续训练脚本中使用）：
   ∂E/∂dyn_proxy_sum > 0    （动态功耗越高，总能耗越高）
   ∂E/∂es{k}_proxy_sum ≤ 0  （节能模式使用越多，总能耗越低）
   E ≥ f(num_cells, total_p_max)  （最低基础功耗约束）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from scipy.signal import savgol_filter

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
FEATURE_DIR = PROJECT_ROOT / "features"
FEATURE_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 数据加载
# ===========================================================================

def load_processed_data() -> pd.DataFrame:
    """
    加载预处理后的 BS 级数据。
    此时数据已完成物理聚合，每行 = 一个基站 × 一个时间步。
    """
    filepath = PROCESSED_DIR / "merged_data.csv"
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['bs_id', 'time']).reset_index(drop=True)

    print(f"[加载数据] {len(df):,} 条记录（BS 级）, {df['bs_id'].nunique()} 个基站")
    print(f"  - 列: {df.columns.tolist()}")
    return df


# ===========================================================================
# 特征组 1：静态硬件特征（对应 P_common + Σ P_static）
# ===========================================================================

def create_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建静态硬件特征，对应物理模型中的 P_common 和 Σ P_static。

    这些特征对同一基站的所有时间步取值相同，
    在 ANP 中作为 Context（基站身份标识）使用。

    数值型：
        total_p_max     : Σ_i p_max_i，静态功耗上界代理
        total_antennas  : Σ_i antennas_i，天线规模
        num_cells       : 该 BS 的总小区数
        p_max_per_cell  : 单小区平均最大发射功率（规模归一化）
    分类型（one-hot）：
        dominant_ru     : 代表性 RUType（设备类型）
        dominant_freq   : 代表性频段
    """
    print("\n>>> 构建静态特征（P_common / ΣP_static 代理）...")

    # 单小区平均最大发射功率（反映设备档次）
    df['p_max_per_cell'] = df['total_p_max'] / df['num_cells']

    # dominant_ru one-hot 编码
    ru_dummies = pd.get_dummies(df['dominant_ru'], prefix='ru')
    df = pd.concat([df, ru_dummies], axis=1)

    # dominant_freq one-hot 编码
    freq_dummies = pd.get_dummies(df['dominant_freq'], prefix='freq')
    df = pd.concat([df, freq_dummies], axis=1)

    ru_cols = [c for c in df.columns if c.startswith('ru_')]
    freq_cols = [c for c in df.columns if c.startswith('freq_')]

    print(f"  - 数值型静态特征: total_p_max, total_antennas, num_cells, p_max_per_cell")
    print(f"  - RUType one-hot: {len(ru_cols)} 维")
    print(f"  - 频段 one-hot: {len(freq_cols)} 维")

    return df


# ===========================================================================
# 特征组 2：动态物理特征（对应 Σ P_dynamic 及节能模式的调制）
# ===========================================================================

def create_dynamic_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建动态物理特征，对应物理模型中的 Σ P_dynamic_i(ρ_i) 和节能调制项。

    核心特征（已在 01_preprocess.py 中完成物理加权聚合）：
        dyn_proxy_sum    : Σ_i (ρ_i × p_max_i × ant_i)
                           物理含义：全站总射频动态功耗代理，量纲一致，可直接叠加
        es{k}_proxy_sum  : Σ_i (ESMode_k_i × p_max_i × ant_i)
                           物理含义：第 k 种节能模式节省的绝对功耗规模代理
                           PI 约束：∂E/∂es{k}_proxy_sum ≤ 0

    派生特征：
        total_es_proxy_sum : 各模式节能代理之和，反映当前总节能力度
        net_dyn_proxy      : dyn_proxy_sum - total_es_proxy_sum
                             物理含义：动态功耗代理扣除节能抵消后的净值
        dyn_proxy_per_cell : 单小区平均动态功耗代理（负载强度归一化）
    """
    print("\n>>> 构建动态物理特征（ΣP_dynamic 代理）...")

    es_proxy_cols = [f'es{k}_proxy_sum' for k in range(1, 7)]
    existing_es = [c for c in es_proxy_cols if c in df.columns]

    # 各节能模式代理之和
    if existing_es:
        df['total_es_proxy_sum'] = df[existing_es].sum(axis=1)
    else:
        df['total_es_proxy_sum'] = 0.0

    # 净动态功耗代理（物理含义：动态功耗减去节能抵消）
    df['net_dyn_proxy'] = df['dyn_proxy_sum'] - df['total_es_proxy_sum']

    # 单小区平均动态功耗代理（负载强度，不受小区数量影响）
    df['dyn_proxy_per_cell'] = df['dyn_proxy_sum'] / df['num_cells'].clip(lower=1)

    # 负载极值特征（保留，用于捕捉基站内不均衡程度）
    # load_max 和 load_mean 来自 01_preprocess.py，不可相加但各有意义：
    #   load_max  → 热点小区压力，反映峰值负载
    #   load_mean → 站内负载均衡程度
    # 注：这两列在 01_preprocess.py 中已聚合，此处直接使用

    print(f"  - 核心特征: dyn_proxy_sum ({df['dyn_proxy_sum'].describe()['mean']:.3f} 均值)")
    print(f"  - 节能代理: {existing_es}")
    print(f"  - 派生: total_es_proxy_sum, net_dyn_proxy, dyn_proxy_per_cell")

    return df


# ===========================================================================
# 特征组 3：时序统计特征（基于 dyn_proxy_sum）
# ===========================================================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建时序统计特征。

    重要设计决策：
    - 时序滞后/滚动统计基于 dyn_proxy_sum（不再使用原始 load）
      物理理由：dyn_proxy_sum 已经考虑了硬件差异，是更准确的"全站动态强度"指标。
      直接对 load_mean 做时序操作，等于忽略了不同时刻小区配置可能不同的问题。

    - 时间周期特征使用正弦/余弦编码（连续且无边界问题）
    """
    print("\n>>> 构建时序统计特征...")

    df = df.sort_values(['bs_id', 'time']).reset_index(drop=True)

    # ---- 时间周期特征 ----
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ---- dyn_proxy_sum 的时序滞后特征 ----
    for lag in [1, 2, 3, 24]:
        df[f'dyn_proxy_lag_{lag}h'] = df.groupby('bs_id')['dyn_proxy_sum'].shift(lag)

    # ---- 滚动统计（基于 dyn_proxy_sum）----
    df['dyn_proxy_roll24_mean'] = df.groupby('bs_id')['dyn_proxy_sum'].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    df['dyn_proxy_roll24_std'] = df.groupby('bs_id')['dyn_proxy_sum'].transform(
        lambda x: x.rolling(window=24, min_periods=1).std()
    )

    # ---- dyn_proxy_sum 的一阶差分（变化率）----
    df['dyn_proxy_diff_1h'] = df.groupby('bs_id')['dyn_proxy_sum'].diff(1)

    # ---- Savitzky-Golay 导数（捕捉负载趋势转折点）----
    df['dyn_proxy_diff_sg'] = np.nan
    df['dyn_proxy_diff2_sg'] = np.nan

    window, polyorder = 5, 2
    for bs_id in df['bs_id'].unique():
        mask = df['bs_id'] == bs_id
        signal = df.loc[mask, 'dyn_proxy_sum'].values.copy()
        n = len(signal)
        win = min(window, n if n % 2 == 1 else max(n - 1, 3))
        if win >= polyorder + 2:
            df.loc[mask, 'dyn_proxy_diff_sg'] = savgol_filter(
                signal, window_length=win, polyorder=polyorder, deriv=1
            )
            df.loc[mask, 'dyn_proxy_diff2_sg'] = savgol_filter(
                signal, window_length=win, polyorder=polyorder, deriv=2
            )
        else:
            df.loc[mask, 'dyn_proxy_diff_sg'] = 0
            df.loc[mask, 'dyn_proxy_diff2_sg'] = 0

    print("  - 时间周期编码: hour_sin/cos, dow_sin/cos")
    print("  - 滞后特征: dyn_proxy_lag_1h/2h/3h/24h")
    print("  - 滚动统计: dyn_proxy_roll24_mean/std")
    print("  - 差分特征: dyn_proxy_diff_1h, dyn_proxy_diff_sg, dyn_proxy_diff2_sg")

    return df


# ===========================================================================
# 特征列定义（PI-ANP 分区）
# ===========================================================================

def define_feature_columns(df: pd.DataFrame) -> dict:
    """
    定义特征列分组，明确区分 PI-ANP 的两类输入。

    PI-ANP 架构中特征的使用方式：
    ┌─────────────────────────────────────────────────────────┐
    │  context_features → ANP Context Encoder                 │
    │  (对同一 BS 固定不变，编码"这个 BS 是谁")                 │
    ├─────────────────────────────────────────────────────────┤
    │  dynamic_features → ANP Target 时序输入                  │
    │  (随时间变化，驱动能耗预测)                               │
    └─────────────────────────────────────────────────────────┘

    pi_monotone_positive : 这些特征增大时，预测能耗应严格增大
        → 用于 PI 损失中的 ∂E/∂x > 0 梯度惩罚
    pi_monotone_negative : 这些特征增大时，预测能耗应严格减小
        → 用于 PI 损失中的 ∂E/∂x < 0 梯度惩罚
    """

    # ---- Context 特征（静态，BS 级不变量）----
    context_numeric = [
        'total_p_max',     # Σ p_max_i → P_static 上界代理
        'total_antennas',  # Σ ant_i   → 天线规模
        'num_cells',       # 小区总数
        'p_max_per_cell',  # 单小区平均最大发射功率
    ]
    context_onehot = [c for c in df.columns if c.startswith('ru_') or c.startswith('freq_')]
    context_features = context_numeric + context_onehot

    # ---- Dynamic 特征（随时间变化）----
    es_proxy_cols = [f'es{k}_proxy_sum' for k in range(1, 7) if f'es{k}_proxy_sum' in df.columns]

    dynamic_physics = [
        'dyn_proxy_sum',       # 核心动态功耗代理 (↑ → ↑ E)
        *es_proxy_cols,        # 各节能模式代理   (↑ → ↓ E)
        'total_es_proxy_sum',  # 总节能规模代理
        'net_dyn_proxy',       # 净动态代理
        'dyn_proxy_per_cell',  # 单小区动态强度
        'load_max',            # 热点小区峰值负载（无量纲，不可加，保留作辅助）
        'load_mean',           # 站内平均负载（均衡度指标）
    ]

    dynamic_temporal = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend',
        'dyn_proxy_lag_1h', 'dyn_proxy_lag_2h',
        'dyn_proxy_lag_3h', 'dyn_proxy_lag_24h',
        'dyn_proxy_roll24_mean', 'dyn_proxy_roll24_std',
        'dyn_proxy_diff_1h',
        'dyn_proxy_diff_sg', 'dyn_proxy_diff2_sg',
    ]

    dynamic_features = dynamic_physics + dynamic_temporal

    # ---- PI 约束分区 ----
    pi_monotone_positive = [
        'dyn_proxy_sum',       # ∂E/∂dyn_proxy_sum > 0
        'net_dyn_proxy',       # ∂E/∂net_dyn_proxy > 0
    ]
    pi_monotone_negative = es_proxy_cols  # ∂E/∂es{k}_proxy_sum ≤ 0

    # ---- 过滤不存在的列 ----
    def filter_existing(cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  [警告] 以下特征列不存在，已跳过: {missing}")
        return [c for c in cols if c in df.columns]

    context_features = filter_existing(context_features)
    dynamic_features = filter_existing(dynamic_features)
    all_features = list(dict.fromkeys(context_features + dynamic_features))

    print(f"\n[特征统计]")
    print(f"  - Context 特征（静态）: {len(context_features)} 维")
    print(f"    其中数值型: {len([c for c in context_numeric if c in df.columns])}, one-hot: {len(context_onehot)}")
    print(f"  - Dynamic 特征（时序）: {len(filter_existing(dynamic_features))} 维")
    print(f"    其中物理代理: {len(filter_existing(dynamic_physics))}, 时序统计: {len(filter_existing(dynamic_temporal))}")
    print(f"  - 总特征数: {len(all_features)} 维")
    print(f"  - PI 单调递增约束: {pi_monotone_positive}")
    print(f"  - PI 单调递减约束: {pi_monotone_negative}")

    return {
        'context': context_features,
        'dynamic': dynamic_features,
        'all': all_features,
        'pi_positive': pi_monotone_positive,
        'pi_negative': pi_monotone_negative,
    }


# ===========================================================================
# 缺失值处理
# ===========================================================================

def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    处理缺失值。

    缺失值来源：
    1. 滞后特征在序列开头处为 NaN（每个 BS 的前 N 个时间步）
    2. 滚动统计特征同上
    3. SG 导数在数据量不足时为 0（已在计算时处理）
    """
    print("\n>>> 处理缺失值...")

    # 滞后特征：前向填充后用中位数兜底
    lag_cols = [c for c in df.columns if 'lag' in c]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df.groupby('bs_id')[col].transform(
                lambda x: x.ffill().bfill()
            )
            df[col] = df[col].fillna(df[col].median())

    # 滚动统计：均值用全局均值，标准差用 0
    roll_mean_cols = [c for c in df.columns if 'roll' in c and 'mean' in c]
    roll_std_cols = [c for c in df.columns if 'roll' in c and 'std' in c]
    for col in roll_mean_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in roll_std_cols:
        df[col] = df[col].fillna(0)

    # 差分特征：序列开头的 NaN 置 0
    diff_cols = [c for c in df.columns if 'diff' in c]
    for col in diff_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 验证
    existing = [c for c in feature_cols if c in df.columns]
    missing_count = df[existing].isna().sum().sum()
    print(f"  - 处理后剩余缺失值: {missing_count}")

    return df


# ===========================================================================
# 保存
# ===========================================================================

def save_features(df: pd.DataFrame, feature_cols: dict):
    """保存特征工程结果"""
    output_path = FEATURE_DIR / "feature_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 特征数据已保存至: {output_path}")
    print(f"  - 形状: {df.shape}")

    cols_path = FEATURE_DIR / "feature_columns.pkl"
    with open(cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"[保存] 特征列定义已保存至: {cols_path}")


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """
    主流程：
    1. 加载 BS 级已聚合数据
    2. 构建静态特征（ANP Context）
    3. 构建动态物理特征（ANP Dynamic Input）
    4. 构建时序统计特征
    5. 定义特征列分组（含 PI 约束标注）
    6. 处理缺失值
    7. 保存
    """
    print("="*60)
    print("基站能耗预测 - 特征工程（PI-ANP 物理机理版）")
    print("="*60)

    # 加载
    df = load_processed_data()

    # 构建特征
    df = create_static_features(df)
    df = create_dynamic_physics_features(df)
    df = create_temporal_features(df)

    # 定义特征列
    feature_cols = define_feature_columns(df)

    # 处理缺失值
    df = handle_missing_values(df, feature_cols['all'])

    # 保存
    save_features(df, feature_cols)

    # 特征与能耗的相关性（快速验证）
    print("\n" + "="*60)
    print("特征与能耗的相关性 (Top 20)")
    print("="*60)
    numeric_feats = [c for c in feature_cols['all']
                     if c in df.columns and df[c].dtype in [np.float64, np.int64, float, int]]
    correlations = df[numeric_feats + ['energy']].corr()['energy'].drop('energy', errors='ignore')
    top_corr = correlations.abs().sort_values(ascending=False).head(20)
    for feat, corr_val in top_corr.items():
        print(f"  {feat:40s}: {corr_val:.4f}")
    print("="*60)

    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = main()
