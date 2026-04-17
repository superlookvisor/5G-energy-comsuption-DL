# 阶段A：基站静态基础能耗估计（Simulation Results 风格）

## 1. 代码与运行说明
- 主脚本：`run_phaseA_static_energy.py`
- 数据输入：`../data/ECdata.csv`, `../data/CLdata.csv`, `../data/BSinfo.csv`
- 关键输出：`outputs/` 下的CSV结果文件，以及 `outputs/figures_emf/` 下的EMF图像文件

## 2. 关键图表说明
- `outputs/figures_emf/fig_method_distribution.emf`：同一静态窗口中不同估计器（min / quantile / mean / median / trimmed mean）的 `P_base` 分布对比。
- `outputs/figures_emf/fig_scatter_pbase_vs_features.emf`：`P_base` 与 `sum_pmax`、`n_cells` 的散点关系，用于验证物理单调趋势。
- `outputs/figures_emf/fig_window_stability.emf`：固定估计方法下，不同静态窗口得到的 `P_base` 稳定性对比。
- `outputs/figures_emf/fig_window_coverage.emf`：不同静态窗口对BS样本覆盖率的影响。

## 3. 结果分析（TSG风格）
### 3.1 静态窗口敏感性分析
本实验在 BS-时间粒度上构建了 92629 条样本，覆盖 923 个基站。不同静态窗口对样本可用性影响显著：
- A窗口（严格）：样本数 16215，BS覆盖率 58.72%
- B窗口（均值负载约束）：样本数 28100，BS覆盖率 78.76%
- C窗口（每BS低负载5%分位）：样本数 3727，BS覆盖率 71.83%
- D窗口（宽松）：样本数 36941，BS覆盖率 87.76%

窗口越严格，样本数下降越明显，容易导致部分BS无可用静态样本；窗口过宽则会混入非静态行为，抬高 `P_base` 估计值。

### 3.2 估计方法对比
跨窗口稳定性统计见 `method_stability_summary.csv`。总体上：
- `min` 估计器更容易受偶发低值影响，存在低估风险。
- `mean` 受残余动态负载影响更明显，可能偏高。
- `quantile`/`median` 在偏差与稳定性之间更均衡，具备更好的鲁棒折中。

本次用于后续建模的目标采用 `D_relaxed + quantile_10`，BS覆盖率为 87.76%，兼顾覆盖度和稳健性。

### 3.3 物理合理性验证
以建模目标 `P_base` 为例，静态特征相关性如下：
- corr(`P_base`, `n_cells`) = 0.795
- corr(`P_base`, `sum_pmax`) = 0.801
- corr(`P_base`, `sum_antennas`) = 0.819

结果显示 `P_base` 随资源规模（小区数、发射功率总和、天线规模）总体呈正相关，满足物理直觉。

### 3.4 统计建模结果
在 BS 级静态特征上，分别训练 Linear / Ridge / Lasso / RandomForest。最佳模型为 `RandomForest`：
- Test R² = 0.8312
- Test MAE = 3.2272

随机森林特征重要性（Top-8）：
- sum_bandwidth: 0.4149
- ru_ratio_Type1: 0.3248
- sum_antennas: 0.1260
- sum_pmax: 0.0576
- ru_ratio_Type7: 0.0453
- mean_frequency: 0.0120
- ru_ratio_Type3: 0.0085
- ru_ratio_Type6: 0.0047

### 3.5 关键发现
- The minimum-based estimator tends to underestimate `P_base` due to extreme outliers.
- Quantile-based estimation provides a robust trade-off between bias and variance.
- Static-feature-driven models can explain a substantial portion of inter-BS base-energy variance.

## 4. 结论（bullet points）
- 静态窗口阈值直接决定样本覆盖率与偏差水平，需要在“纯静态性”与“可用样本量”之间平衡。
- 相比最小值估计，分位数/中位数估计对异常点更不敏感，更适合作为 `P_base` 监督信号。
- `P_base` 与 `sum_pmax`、`n_cells`、`sum_antennas` 呈正向关系，验证了模型构建的物理合理性。
- 基于静态配置特征可实现对 `P_base` 的有效回归，为阶段B动态能耗建模提供物理基座。

## 5. 额外分析（加分项）
- `bs_without_static_samples.csv` 给出各窗口下无静态样本的BS列表（按窗口展开）。
- 对“无静态窗口样本”BS，建议采用“配置相似BS迁移”：
  1) 在静态特征空间（`sum_pmax`, `n_cells`, `sum_antennas`, RU/Mode比例）做最近邻匹配；
  2) 以匹配集合的 `P_base` 分位数估计作为回填值，并附带不确定性区间。

无静态样本窗口：A_strict_weighted, B_mean_all_es_zero, C_bs_low5pct, D_relaxed
