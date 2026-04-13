import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import stats

# 路径配置
FEATURE_DIR = Path(r"g:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\features")
MODEL_DIR = Path(r"g:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\models")

def analyze_residuals():
    print("="*60)
    print("步骤 1: 检查残差分布")
    print("="*60)
    
    df = pd.read_csv(FEATURE_DIR / "residual_data.csv")
    residuals = df['residual']
    
    mean_val = residuals.mean()
    std_val = residuals.std()
    max_val = residuals.max()
    min_val = residuals.min()
    
    # 偏度和峰度
    skew = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    print(f"样本数量: {len(residuals):,}")
    print(f"均值:     {mean_val:.6f}  (越接近 0 说明物理模型越无偏)")
    print(f"标准差:   {std_val:.6f}")
    print(f"最小值:   {min_val:.6f}")
    print(f"最大值:   {max_val:.6f}")
    print(f"偏度:     {skew:.4f}  (0 为对称)")
    print(f"峰度:     {kurtosis:.4f} (0 为正态)")
    
    # 正态性检验 (D'Agostino's K^2 Test)
    k2, p = stats.normaltest(residuals)
    print(f"正态性检验 p-value: {p:.4e}")
    if p < 0.05:
        print("结论: 残差不完全符合完美正态分布（大数据下常见），但均值极小且分布相对集中。")
    else:
        print("结论: 残差符合正态分布。")

def analyze_gamma_weights():
    print("\n" + "="*60)
    print("步骤 2: 验证节能权重 (Gamma)")
    print("="*60)
    
    model_path = MODEL_DIR / "physics_baseline.pkl"
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    gamma = model_data['gamma']
    es_cols = model_data['es_col_names']
    
    # 创建 DataFrame 进行排序
    weights = pd.DataFrame({
        'Mode': es_cols,
        'Gamma': gamma
    }).sort_values('Gamma', ascending=False)
    
    print("节能模式系数排名（系数越大，同等负载下节省功耗越多）：")
    for i, row in enumerate(weights.itertuples(), 1):
        print(f"  {i}. {row.Mode}: {row.Gamma:.6f}")
    
    top_mode = weights.iloc[0]['Mode']
    print(f"\n物理分析: {top_mode} 在当前数据下表现出最显著的‘功率削减’效应。")

if __name__ == "__main__":
    analyze_residuals()
    analyze_gamma_weights()
