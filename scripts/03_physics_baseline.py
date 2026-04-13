"""
03_physics_baseline.py - 物理基准模型（BS 级聚合版）
基站能耗预测建模项目

物理模型定义（BS 级）：
    E_BS = P_common(hw) + Σ_i P_static_i(hw_i) + Σ_i P_dynamic_i(ρ_i, hw_i) - Σ_k ΔP_save_k

用已聚合的物理代理特征表达为：
    E_BS ≈ α₀ + α₁·total_p_max + α₂·total_antennas + α₃·num_cells   # 静态项
           + β·dyn_proxy_sum                                           # 动态项（↑ → ↑ E）
           - Σ_k γ_k·es{k}_proxy_sum                                  # 节能项（↑ → ↓ E）

设计参数约束（Physics-Informed）：
    α_i ≥ 0  （硬件规模越大，基础功耗越高）
    β   ≥ 0  （动态功耗代理越大，总能耗越高）
    γ_k ≥ 0  （节能规模越大，节省越多）

本脚本功能：
    1. 用带约束的线性回归估计上述参数（OLS with non-negativity / Ridge）
    2. 计算物理基准预测值 P_base 与残差 residual = E - P_base
    3. 保存模型参数与残差数据，供 PI-ANP 使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import pickle
import warnings
warnings.filterwarnings('ignore')

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "features"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 数据加载
# ===========================================================================

def load_feature_data() -> pd.DataFrame:
    """加载特征工程后的 BS 级数据"""
    filepath = FEATURE_DIR / "feature_data.csv"
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['bs_id', 'time']).reset_index(drop=True)
    print(f"[加载数据] {len(df):,} 条记录（BS 级），{df['bs_id'].nunique()} 个基站")
    return df


# ===========================================================================
# 物理基准模型
# ===========================================================================

class PhysicsBaselineModel:
    """
    物理基准模型（BS 级）

    使用已在 01_preprocess.py 中完成物理聚合的特征：
        静态项输入：total_p_max, total_antennas, num_cells
        动态项输入：dyn_proxy_sum       = Σ_i (ρ_i × p_max_i × ant_i)
        节能项输入：es{k}_proxy_sum     = Σ_i (ESMode_k_i × p_max_i × ant_i)

    参数约束（物理先验）：
        所有系数 ≥ 0（静态、动态均为正向贡献）
        节能系数 γ_k ≥ 0（节能项在公式中以负号出现，γ 本身非负）

    两步拟合策略：
        Step 1：仅用静态特征拟合低负载样本（去除动态干扰）
        Step 2：在静态残差上拟合动态特征和节能特征
    """

    # 节能代理列名
    ES_PROXY_COLS = [f'es{k}_proxy_sum' for k in range(1, 7)]

    def __init__(self):
        self.alpha = None      # 静态项系数 [intercept, total_p_max, total_antennas, num_cells]
        self.beta = None       # 动态项系数（dyn_proxy_sum 的系数）
        self.gamma = None      # 节能项系数（各 es{k}_proxy_sum 的系数，非负）
        self.fitted = False

    # ------------------------------------------------------------------
    # 子模块：构建设计矩阵
    # ------------------------------------------------------------------

    def _static_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        静态设计矩阵：对应 P_common + Σ P_static_i

        不含截距列（由优化器单独处理），含：
            total_p_max     : Σ_i p_max_i
            total_antennas  : Σ_i ant_i
            num_cells       : 小区总数（P_static 计数项）
        """
        return df[['total_p_max', 'total_antennas', 'num_cells']].values

    def _dynamic_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        动态设计矩阵：对应 Σ_i P_dynamic_i(ρ_i)

        dyn_proxy_sum = Σ_i (ρ_i × p_max_i × ant_i)
        已在预处理阶段完成物理加权，可直接作为线性项使用。
        dyn_proxy_sum² 捕捉非线性负载效应（射频功放效率非线性）
        """
        x = df['dyn_proxy_sum'].values
        return np.column_stack([x, x ** 2])

    def _es_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        节能设计矩阵：对应 - Σ_k ΔP_save_k

        es{k}_proxy_sum = Σ_i (ESMode_k_i × p_max_i × ant_i)
        已在预处理阶段完成物理加权。
        系数 γ_k ≥ 0，在模型公式中以负号出现。
        """
        existing = [c for c in self.ES_PROXY_COLS if c in df.columns]
        return df[existing].values, existing

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> 'PhysicsBaselineModel':
        """
        两步拟合物理基准模型。

        Step 1 - 静态功耗估计：
            仅选取低负载样本（dyn_proxy_sum < 20th 分位数），
            此时动态功耗贡献较小，主要反映静态功耗。
            用 Ridge(positive=True) 确保系数非负。

        Step 2 - 动态+节能功耗估计：
            在全量数据上，以静态残差为目标，
            用带界约束的 L-BFGS-B 拟合动态和节能参数。
        """
        print("\n>>> 拟合物理基准模型（BS 级物理代理版）...")
        y = df['energy'].values

        # ---- Step 1: 静态功耗估计 ----
        print("  [Step 1] 估计静态功耗参数（低负载样本）...")

        low_load_mask = df['dyn_proxy_sum'] <= df['dyn_proxy_sum'].quantile(0.20)
        df_low = df[low_load_mask]
        y_low = df_low['energy'].values
        X_static_low = self._static_features(df_low)

        # Ridge with non-negative constraint
        static_model = Ridge(alpha=10.0, positive=True)
        static_model.fit(X_static_low, y_low)

        self._static_model = static_model
        P_static_all = static_model.predict(self._static_features(df))

        # 检验静态模型在低负载上的质量
        P_static_low = static_model.predict(X_static_low)
        r2_static = 1 - np.sum((y_low - P_static_low)**2) / np.sum((y_low - y_low.mean())**2)
        print(f"    静态模型（低负载子集）R2: {r2_static:.4f}")
        print(f"    截距（公共功耗基底）: {static_model.intercept_:.2f} kWh")
        print(f"    total_p_max 系数:    {static_model.coef_[0]:.4f}")
        print(f"    total_antennas 系数: {static_model.coef_[1]:.4f}")
        print(f"    num_cells 系数:      {static_model.coef_[2]:.4f}")

        # ---- Step 2: 动态+节能功耗估计 ----
        print("  [Step 2] 估计动态功耗和节能系数（全量数据）...")

        residual_static = y - P_static_all             # E - P_static_pred
        X_dyn = self._dynamic_features(df)             # [dyn_proxy, dyn_proxy²]
        X_es, es_col_names = self._es_features(df)     # [es1~6_proxy_sum]

        n_dyn = X_dyn.shape[1]    # 2
        n_es = X_es.shape[1]      # 最多 6

        def loss(params):
            beta = params[:n_dyn]     # 动态系数（非负）
            gamma = params[n_dyn:]    # 节能系数（非负，以负号出现）
            P_dyn = X_dyn @ beta
            P_save = X_es @ gamma
            pred = P_dyn - P_save
            return np.mean((residual_static - pred) ** 2)

        # 初始值：动态系数初始化为正数，节能系数初始化为小正数
        x0 = np.concatenate([
            np.full(n_dyn, 2.0),
            np.full(n_es, 0.5)
        ])
        bounds = [(0, None)] * (n_dyn + n_es)

        result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12})

        self.beta = result.x[:n_dyn]
        self.gamma = result.x[n_dyn:]

        print(f"    优化收敛: {result.success}（迭代 {result.nit} 次）")
        print(f"    β (dyn_proxy 一次项):  {self.beta[0]:.4f}")
        print(f"    β (dyn_proxy 二次项):  {self.beta[1]:.6f}")
        for i, col in enumerate(es_col_names):
            print(f"    γ ({col}): {self.gamma[i]:.4f}")

        # 保存系数
        self.alpha = {
            'intercept': static_model.intercept_,
            'total_p_max': static_model.coef_[0],
            'total_antennas': static_model.coef_[1],
            'num_cells': static_model.coef_[2],
        }
        self.es_col_names = es_col_names
        self.fitted = True

        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算物理基准预测值。

        P_base = P_static + P_dynamic - P_save
               = [α₀ + α₁·total_p_max + α₂·total_antennas + α₃·num_cells]
                 + [β₀·dyn_proxy + β₁·dyn_proxy²]
                 - [Σ_k γ_k·es{k}_proxy_sum]

        物理语义：
            第一项 → P_common + Σ P_static_i
            第二项 → Σ P_dynamic_i（非线性射频负载效应）
            第三项 → 节能模式带来的功耗削减（负贡献，γ ≥ 0 确保减少）
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        P_static = self._static_model.predict(self._static_features(df))
        X_dyn = self._dynamic_features(df)
        P_dyn = X_dyn @ self.beta

        X_es, _ = self._es_features(df)
        P_save = X_es @ self.gamma

        return P_static + P_dyn - P_save

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, label: str = "全量数据") -> dict:
        """评估物理基准模型性能"""
        y_true = df['energy'].values
        y_pred = self.predict(df)

        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        metrics = {'MAPE': mape, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

        print("\n" + "="*60)
        print(f"物理基准模型评估 - {label}")
        print("="*60)
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.4f} kWh")
        print(f"  MAE:  {mae:.4f} kWh")
        print(f"  R2:   {r2:.4f}")

        # 验证物理约束符合预期
        print("\n物理约束验证：")
        corr_dyn = np.corrcoef(df['dyn_proxy_sum'].values, y_pred)[0, 1]
        print(f"  corr(dyn_proxy_sum, P_base)  = {corr_dyn:.4f}  (期望 > 0)")
        for i, col in enumerate(self.es_col_names):
            if col in df.columns:
                corr_es = np.corrcoef(df[col].values, y_pred)[0, 1]
                print(f"  corr({col}, P_base) = {corr_es:.4f}  (期望 ≤ 0)")

        print("="*60)
        return metrics

    # ------------------------------------------------------------------
    # 分析：功耗分解
    # ------------------------------------------------------------------

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输出每条记录的功耗分解结果（P_static / P_dynamic / P_save）。
        便于可视化和分析功耗结构。
        """
        P_static = self._static_model.predict(self._static_features(df))
        X_dyn = self._dynamic_features(df)
        P_dyn = X_dyn @ self.beta
        X_es, _ = self._es_features(df)
        P_save = X_es @ self.gamma
        P_base = P_static + P_dyn - P_save

        result = df[['time', 'bs_id', 'energy']].copy()
        result['P_static'] = P_static
        result['P_dynamic'] = P_dyn
        result['P_save'] = P_save
        result['P_base'] = P_base
        result['residual'] = df['energy'].values - P_base

        # 功耗占比
        result['pct_static'] = P_static / np.clip(P_base, 1e-6, None) * 100
        result['pct_dynamic'] = P_dyn / np.clip(P_base, 1e-6, None) * 100

        return result

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(self, filepath: Path):
        """保存模型"""
        model_data = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'es_col_names': self.es_col_names,
            '_static_model': self._static_model,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n[保存] 物理基准模型已保存至: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'PhysicsBaselineModel':
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.alpha = model_data['alpha']
        model.beta = model_data['beta']
        model.gamma = model_data['gamma']
        model.es_col_names = model_data['es_col_names']
        model._static_model = model_data['_static_model']
        model.fitted = True
        print(f"[加载] 物理基准模型已从: {filepath}")
        return model


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """
    主流程：
    1. 加载特征数据（BS 级，92,629 条）
    2. 拟合物理基准模型（两步法）
    3. 评估与功耗分解
    4. 保存模型参数与残差数据
    """
    print("="*60)
    print("基站能耗预测 - 物理基准模型（BS 级物理代理版）")
    print("="*60)

    # 加载
    df = load_feature_data()

    # 拟合
    model = PhysicsBaselineModel()
    model.fit(df)

    # 评估（全量）
    metrics = model.evaluate(df, label="全量数据")

    # 功耗分解
    decomp = model.decompose(df)

    print("\n功耗分解统计（均值）：")
    print(f"  P_static  : {decomp['P_static'].mean():.4f} kWh  "
          f"({decomp['pct_static'].mean():.1f}%)")
    print(f"  P_dynamic : {decomp['P_dynamic'].mean():.4f} kWh  "
          f"({decomp['pct_dynamic'].mean():.1f}%)")
    print(f"  P_save    : {decomp['P_save'].mean():.4f} kWh（节能削减）")
    print(f"  P_base    : {decomp['P_base'].mean():.4f} kWh（预测值）")
    print(f"  Energy    : {decomp['energy'].mean():.4f} kWh（真实值）")

    print("\n残差统计（residual = Energy - P_base）：")
    print(f"  均值:   {decomp['residual'].mean():.4f}")
    print(f"  标准差: {decomp['residual'].std():.4f}")
    print(f"  最小值: {decomp['residual'].min():.4f}")
    print(f"  最大值: {decomp['residual'].max():.4f}")
    print(f"  残差均值接近0说明物理模型无系统偏差，PI-ANP 可专注拟合非线性修正项")

    # 保存模型
    model_path = MODEL_DIR / "physics_baseline.pkl"
    model.save(model_path)

    # 保存残差数据（供 PI-ANP 训练使用）
    residual_path = FEATURE_DIR / "residual_data.csv"
    decomp.to_csv(residual_path, index=False, encoding='utf-8-sig')
    print(f"[保存] 功耗分解+残差数据已保存至: {residual_path}")
    print(f"  - 形状: {decomp.shape}")
    print(f"  - 列: {decomp.columns.tolist()}")

    return model, metrics, decomp


if __name__ == "__main__":
    model, metrics, decomp = main()
