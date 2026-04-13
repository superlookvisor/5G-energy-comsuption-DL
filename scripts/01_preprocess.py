"""
01_preprocess.py - 数据加载、物理聚合与合并
基站能耗预测建模项目

数据流：
  CLdata (小区级) + BSinfo (小区级硬件) → 物理加权聚合 → BS级动态特征
  + ECdata (BS级能耗) → 内连接 → BS级建模数据集

物理功耗分解：
  E_BS = P_common + Σ_i P_static_i + Σ_i P_dynamic_i(ρ_i)

关键设计决策：
- 负载率 ρ_i 是无量纲比例，不同小区之间不能直接相加
  → 先与小区硬件参数相乘得到有量纲的"绝对动态功耗代理"，再跨小区求和
- ESMode 数值代表激活时长比例（0~1），同一小时内多个模式可重叠
  → 加权时长 = ESMode_k_i × p_max_i × antennas_i，跨小区求和有物理意义
- BSinfo 中的静态特征（天线、频段等）对应 P_common + Σ P_static 的"规模底座"
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 第一层：原始数据加载
# ===========================================================================

def load_bsinfo(filepath: str) -> pd.DataFrame:
    """
    加载基站硬件配置信息（小区粒度）

    返回含以下关键列的 DataFrame：
        bs_id, cell_name, ru_type, transmission_mode,
        frequency, bandwidth, antennas, p_max
    """
    df = pd.read_csv(filepath, encoding='gbk')
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        'BS': 'bs_id',
        'CellName': 'cell_name',
        'RUType': 'ru_type',
        'Transimission_mode': 'transmission_mode',
        'Frequency': 'frequency',
        'Bandwidth': 'bandwidth',
        'Antennas': 'antennas',
        'maximum_transimission_power': 'p_max'
    })

    print(f"[BSinfo] 加载完成: {len(df)} 行（小区级）")
    print(f"  - 独立基站数: {df['bs_id'].nunique()}")
    print(f"  - RUType: {sorted(df['ru_type'].unique())}")
    print(f"  - 天线数分布: {sorted(df['antennas'].unique())}")

    return df


def load_cldata(filepath: str) -> pd.DataFrame:
    """
    加载小区负载与节能模式数据（小区粒度，每行=一个小区×一个时间步）

    ESMode1~6 的数值含义：该小时内该节能模式的激活时长比例（0~1）
    多个模式在同一小时内可同时激活（数值之和可超过1）
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        'Time': 'time',
        'BS': 'bs_id',
        'CellName': 'cell_name',
        'load': 'load'
    })

    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')

    print(f"[CLdata] 加载完成: {len(df):,} 条记录（小区级）")
    print(f"  - 独立基站数: {df['bs_id'].nunique()}")
    print(f"  - 时间范围: {df['time'].min()} 到 {df['time'].max()}")
    print(f"  - 负载率范围: [{df['load'].min():.4f}, {df['load'].max():.4f}]")

    return df


def load_ecdata(filepath: str) -> pd.DataFrame:
    """
    加载基站能耗数据（BS粒度，每行=一个基站×一个时间步）

    Energy 是 BS 整体的小时能耗（含所有小区+公共设备）
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        'Time': 'time',
        'BS': 'bs_id',
        'Energy': 'energy'
    })

    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')

    print(f"[ECdata] 加载完成: {len(df):,} 条记录（BS级）")
    print(f"  - 独立基站数: {df['bs_id'].nunique()}")
    print(f"  - 能耗范围: [{df['energy'].min():.2f}, {df['energy'].max():.2f}] kWh")
    print(f"  - 能耗均值: {df['energy'].mean():.2f} kWh")

    return df


# ===========================================================================
# 第二层：在小区级计算有量纲的物理代理量
# ===========================================================================

def attach_hardware_to_cells(cl_df: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 BSinfo 中的小区硬件参数关联到 CLdata 的每条记录上。

    之所以在此步骤合并（而非在 BS 级聚合后合并），是因为：
    不同小区的 p_max、antennas 不同，必须先在小区级完成加权计算，
    再跨小区求和，才能得到物理正确的 BS 级特征。
    """
    cell_hw_cols = ['bs_id', 'cell_name', 'p_max', 'antennas', 'bandwidth', 'frequency']
    merged = pd.merge(
        cl_df,
        bs_df[cell_hw_cols],
        on=['bs_id', 'cell_name'],
        how='left'
    )

    missing = merged['p_max'].isna().sum()
    if missing > 0:
        print(f"  [警告] {missing} 条小区记录在 BSinfo 中找不到匹配的硬件配置")

    return merged


def compute_cell_level_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    在小区粒度计算有量纲的物理代理变量。

    背景：
        P_dynamic_i ≈ α * ρ_i * p_max_i * antennas_i
        其中 ρ_i 是无量纲负载率，乘以 p_max_i 和 antennas_i 后
        转化为与实际射频功耗成比例的有量纲量。
        此时对多个小区做求和，在量纲上是自洽的（相当于功率叠加）。

    ESMode 处理：
        ESMode_k_i 是该小时内节能模式 k 在小区 i 的激活时长比例。
        小区 i 通过节能模式 k 节省的功耗正比于 ESMode_k_i * p_max_i * antennas_i，
        对多个小区求和得到基站级的绝对节能规模代理量。

    列命名规则：
        dyn_proxy        : 动态功耗代理（负载 × 硬件规模）
        es{k}_proxy      : 第 k 种节能模式的绝对节能规模代理
    """
    hw_scale = df['p_max'] * df['antennas']          # 单小区硬件规模

    # 动态功耗代理：有量纲，可跨小区叠加
    df['dyn_proxy'] = df['load'] * hw_scale

    # 节能模式代理：激活时长 × 硬件规模，可跨小区叠加
    es_cols = [f'ESMode{k}' for k in range(1, 7)]
    for col in es_cols:
        k = col.replace('ESMode', '')
        df[f'es{k}_proxy'] = df[col] * hw_scale

    return df


# ===========================================================================
# 第三层：从小区级聚合到 BS 级
# ===========================================================================

def aggregate_to_bs_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    将小区级的物理代理量聚合到 BS 级（每行=一个基站×一个时间步）。

    聚合逻辑与物理含义：
        dyn_proxy_sum  : Σ_i (ρ_i * p_max_i * ant_i) → 全站总射频动态功耗代理
        es{k}_proxy_sum: Σ_i (ESMode_k_i * p_max_i * ant_i) → 全站第k种节能的绝对规模
        n_cells        : 当前时间步该基站有数据的小区数
        load_max       : 各小区中最高的负载率（反映热点压力）

    注意：load_mean / load_sum 单独保留，但赋予其正确的物理解释：
        load_mean 是有意义的（反映各小区负载均衡程度）
        load_sum  无明确物理意义（不建议直接输入模型，但保留供分析对比）
    """
    es_proxy_cols = {f'es{k}_proxy': 'sum' for k in range(1, 7)}

    agg_dict = {
        'dyn_proxy': 'sum',       # Σ_i (ρ_i * p_max_i * ant_i)
        **es_proxy_cols,
        'load': ['max', 'mean'],   # 负载极值与均值（不可加，单独保留）
        'cell_name': 'count',      # 小区数量
    }

    bs_df = df.groupby(['time', 'bs_id']).agg(agg_dict).reset_index()

    # 展平多级列名
    bs_df.columns = [
        '_'.join(filter(None, col)) if isinstance(col, tuple) else col
        for col in bs_df.columns
    ]
    bs_df = bs_df.rename(columns={
        'cell_name_count': 'n_cells',
        'load_max': 'load_max',
        'load_mean': 'load_mean',
        'dyn_proxy_sum': 'dyn_proxy_sum',
    })
    # 节能代理统一重命名
    for k in range(1, 7):
        old = f'es{k}_proxy_sum'
        if old in bs_df.columns:
            bs_df = bs_df.rename(columns={old: f'es{k}_proxy_sum'})

    print(f"[聚合] 小区级 → BS级: {len(bs_df):,} 条记录")
    print(f"  - 独立基站数: {bs_df['bs_id'].nunique()}")

    return bs_df


def attach_bs_static_features(bs_ts_df: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 BSinfo 中 BS 级的静态特征（对应 P_common + Σ P_static 的规模底座）
    聚合并关联到 BS 时序数据上。

    以下特征与时间无关，反映基站的"硬件规模"：
        num_cells      : 该基站的小区总数（即使有些小区当前在节能模式）
        total_p_max    : Σ_i p_max_i（最大发射功率之和，正比于静态功耗上界）
        total_antennas : Σ_i antennas_i（总天线数）
        dominant_ru    : 该 BS 拥有最多小区的 RUType（代表设备类型）
    """
    bs_static = bs_df.groupby('bs_id').agg(
        num_cells=('cell_name', 'count'),
        total_p_max=('p_max', 'sum'),
        total_antennas=('antennas', 'sum'),
        # 选取出现次数最多的 RUType 作为 BS 的代表性设备类型
        dominant_ru=('ru_type', lambda x: x.mode()[0]),
        dominant_freq=('frequency', lambda x: x.mode()[0]),
    ).reset_index()

    merged = pd.merge(bs_ts_df, bs_static, on='bs_id', how='left')

    missing = merged['num_cells'].isna().sum()
    if missing > 0:
        print(f"  [警告] {missing} 条 BS 时序记录找不到对应的静态配置")

    print(f"[静态特征] 已关联 BS 级硬件配置: {bs_static['bs_id'].nunique()} 个基站")

    return merged


# ===========================================================================
# 第四层：与能耗数据合并（均为 BS 级，可安全内连接）
# ===========================================================================

def merge_with_energy(bs_ts_df: pd.DataFrame, ec_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 BS 级时序特征与 BS 级能耗数据按 (time, bs_id) 内连接。

    此时两边均为 BS 粒度，内连接在逻辑上是严格正确的：
    - 左侧：从 CLdata 聚合而来的 BS 级特征
    - 右侧：ECdata 的 BS 级能耗（目标变量）
    丢弃的记录 = ECdata 中存在但 CLdata 中缺失对应小区数据的时间步
    """
    merged = pd.merge(
        bs_ts_df,
        ec_df[['time', 'bs_id', 'energy']],
        on=['time', 'bs_id'],
        how='inner'
    )

    n_before = len(bs_ts_df)
    n_after = len(merged)
    print(f"\n[合并能耗] BS级特征 × ECdata 内连接: {n_after:,} 条记录")
    print(f"  - 聚合后 BS级特征: {n_before:,} 条")
    print(f"  - 因 ECdata 缺失被排除: {n_before - n_after:,} 条")
    print(f"  - 保留基站数: {merged['bs_id'].nunique()}")

    return merged


# ===========================================================================
# 辅助：统计与保存
# ===========================================================================

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取时间特征（在 BS 级数据上操作）"""
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['date'] = df['time'].dt.date
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    """输出 BS 级数据集基本统计信息"""
    stats = {
        'total_records': len(df),
        'unique_bs': df['bs_id'].nunique(),
        'energy_mean': df['energy'].mean(),
        'energy_std': df['energy'].std(),
        'energy_min': df['energy'].min(),
        'energy_max': df['energy'].max(),
        'dyn_proxy_mean': df['dyn_proxy_sum'].mean(),
    }

    print("\n" + "="*60)
    print("数据集统计信息（BS 级）")
    print("="*60)
    print(f"总记录数: {stats['total_records']:,}")
    print(f"基站数量: {stats['unique_bs']}")
    print(f"\n能耗 (kWh):")
    print(f"  均值: {stats['energy_mean']:.2f}")
    print(f"  标准差: {stats['energy_std']:.2f}")
    print(f"  范围: [{stats['energy_min']:.2f}, {stats['energy_max']:.2f}]")
    print(f"\n动态功耗代理 dyn_proxy_sum:")
    print(f"  均值: {stats['dyn_proxy_mean']:.4f}")
    print(f"\n小区数量分布（num_cells）:")
    print(df['num_cells'].value_counts().sort_index().to_string())
    print("="*60)

    return stats


def save_processed_data(df: pd.DataFrame, output_path: str):
    """保存处理后的数据集"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {output_path}")
    print(f"  - 形状: {df.shape}")
    print(f"  - 列: {df.columns.tolist()}")


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """
    主流程：
    1. 加载原始数据（三个 CSV 文件）
    2. 小区级：关联硬件参数，计算物理代理量
    3. BS 级：跨小区聚合，关联静态特征
    4. 合并能耗数据（均为 BS 级，内连接）
    5. 提取时间特征，保存
    """
    print("="*60)
    print("基站能耗预测 - 数据预处理（物理聚合版）")
    print("="*60)

    # 数据文件路径
    bsinfo_path = str(PROJECT_ROOT.parent / "BSinfo.csv")
    cldata_path = str(PROJECT_ROOT.parent / "CLdata.csv")
    ecdata_path = str(PROJECT_ROOT.parent / "ECdata.csv")

    # ---- Step 1: 加载原始数据 ----
    print("\n>>> Step 1: 加载原始数据...")
    bs_df = load_bsinfo(bsinfo_path)
    cl_df = load_cldata(cldata_path)
    ec_df = load_ecdata(ecdata_path)

    # ---- Step 2: 小区级物理代理计算 ----
    print("\n>>> Step 2: 小区级 - 关联硬件参数，计算物理代理量...")
    cl_df = attach_hardware_to_cells(cl_df, bs_df)
    cl_df = compute_cell_level_physics(cl_df)

    # ---- Step 3: 聚合到 BS 级 ----
    print("\n>>> Step 3: 聚合到 BS 级...")
    bs_ts_df = aggregate_to_bs_level(cl_df)
    bs_ts_df = attach_bs_static_features(bs_ts_df, bs_df)

    # ---- Step 4: 合并能耗数据 ----
    print("\n>>> Step 4: 合并 ECdata（BS 级内连接）...")
    merged_df = merge_with_energy(bs_ts_df, ec_df)

    # ---- Step 5: 时间特征 + 统计 ----
    print("\n>>> Step 5: 提取时间特征...")
    merged_df = extract_time_features(merged_df)
    stats = basic_stats(merged_df)

    # ---- 保存 ----
    output_path = OUTPUT_DIR / "merged_data.csv"
    save_processed_data(merged_df, str(output_path))

    return merged_df, stats


if __name__ == "__main__":
    merged_df, stats = main()
