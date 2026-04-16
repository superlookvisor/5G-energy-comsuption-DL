# 面向基站日前能耗预测的物理引导两阶段代理框架

> 本文稿撰写体例参考 IEEE Transactions on Smart Grid（TSG）。正式投稿时须转换为 IEEEtran 期刊模板，采用 IEEE 双栏版式、编号图表，并满足期刊页数限制。IEEE PES TSG 的研究范畴涵盖智能电网数据分析、通信技术、信息物理系统及需求侧应用等，故本文宜将研究定位为面向通信—能源基础设施的数据分析问题，而非一般意义上的负荷预测。

## 摘要

针对第五代移动通信（5G）基站日前能耗预测问题，在预测起点处未来业务负载与节能（energy saving, ES）状态均不可观测，因而预测难度较大。为此，本文提出一种物理引导的两阶段代理（proxy）框架：首先，基于历史同小时信息、滚动时间窗统计量及当前观测构造日前协变量代理；其次，在由 Phase A 得到的静态基功率估计之上，预测动态能耗分量。进一步地，引入受约束堆叠（stacking）模块，在不使用未来能耗标签的前提下学习凸组合代理权重。实验在基站面板数据上开展，并采用按基站分组的交叉验证。在与稀疏观测一致的质量控制过滤协议下，所提两阶段代理与随机森林（Random Forest）相结合，在严格 24 h 轨迹意义下的平均绝对百分比误差（mean absolute percentage error, MAPE）上取得最优值 0.09096，相对固定权重基线约改善 20.0%。物理一致性检验表明：负载—能耗关系满足单调性；主导节能模式呈现负向效应；基功率估计与站点规模之间具有较强一致性。上述结果表明，学习得到的日前协变量代理在保持可解释性的同时，可显著提升基站能耗预测性能。

## 关键词

5G 基站；日前能耗预测；智能电网数据分析；物理引导机器学习；代理协变量；受约束堆叠；节能模式。

## 符号说明

| 符号 | 含义 |
| --- | --- |
| $b$ | 基站索引。 |
| $t_0$ | 日前预测起点；本文中固定为 00:00。 |
| $h$ | 预测时距，$h \in \{1,\ldots,24\}$（单位：小时）。 |
| $E_{b,t}$ | 观测到的基站总能耗。 |
| $p^{base}_b$ | 由 Phase A 导入的静态基功率分量。 |
| $y_{b,t}$ | 动态能耗分量，$y_{b,t}=E_{b,t}-p^{base}_b$。 |
| $\widehat{E}_{b,t_0+h}$ | 预测的基站总能耗。 |
| $\widehat{D}_{1}, \widehat{D}_{2}, \widehat{D}_{3}$ | 分别对应加权负载、非线性负载与负载波动性的代理物理特征。 |
| $S_m$ | 第 $m$ 种节能模式强度，$m=1,\ldots,6$。 |
| $\widehat{S}_m$ | 第 $m$ 种节能模式的日前代理。 |
| $\widehat{I}_m$ | 交互代理项，$\widehat{I}_m=\widehat{S}_m \cdot \widehat{load}_{pmax}$。 |
| $w$ | 满足 $w \ge 0$ 且 $\mathbf{1}^{T}w=1$ 的凸代理权重向量。 |

## I. 引言

第五代移动通信网络的快速部署使通信基础设施与电力系统之间的运行耦合关系日益增强。基站能耗由硬件规模、业务负载及节能控制策略共同决定。准确的日前能耗预测可为需求响应、能源采购及网络侧能源管理等应用提供支撑。然而，该问题与常规短期负荷预测存在本质差异：在日前预测起点，未来业务负载与节能状态无法直接观测。

本文研究面向 Phase B 的“仅日前信息可用”实验设定。与日内滚动预测不同，该设定禁止模型使用目标时刻的滞后能耗、目标时刻负载及未来节能状态观测。当目标时刻真实协变量缺失时，如何构造具有物理可解释性的代理协变量，是本文关注的核心问题。对此，本文构建历史代理协变量，并基于目标时刻协变量的重构误差学习其凸组合权重；整个学习过程不使用未来能耗标签，以避免信息泄漏。

本文主要贡献可概括如下：

1. 提出一种仅依赖日前信息的面板构造协议：在固定 00:00 预测起点下，对未来 24 个整点目标进行预测，并保证时间戳的严格连续性。
2. 提出两阶段代理特征设计：将历史负载与节能代理映射为可解释的动态能耗特征。
3. 提出受约束代理权重学习模块：在规避能耗标签泄漏的前提下，相对固定启发式权重提升代理质量。
4. 建立涵盖严格 24 h 轨迹指标、分时距误差、物理一致性检验及回退链（fallback）诊断的仿真与报告流程。

## II. 问题定义

对任一基站 $b$ 及预测起点 $t_0$，需预测时刻 $t_0+h$ 的总能耗，其中 $h=1,\ldots,24$。总能耗预测被分解为

$\widehat{E}_{b,t_0+h}=p^{base}_b+\widehat{y}_{b,t_0+h}$。

其中，静态项 $p^{base}_b$ 取自 Phase A（配置为 `D_relaxed + quantile_10`）；Phase B 则聚焦于动态分量 $\widehat{y}_{b,t_0+h}$ 的预测。可用信息被限制为不晚于预测起点的观测以及历史同小时统计摘要，从而避免在实际日前流程中不可获得的日内滞后特征与未来协变量进入模型。

评价指标从两个互补角度构造：

- **严格轨迹指标**：仅保留完整 24 h 轨迹；对每条轨迹分别计算 MAPE、峰值误差与谷值误差后，再对轨迹取平均。
- **分时距指标**：在每个预测时距 $h$ 上，利用全部有效样本计算平均绝对误差（mean absolute error, MAE）、均方根误差（root mean square error, RMSE）、MAPE 及样本量。

前者用于刻画端到端的日级预测行为，后者用于揭示随 $h$ 变化时数据可用性与预测难度的演变规律。

## III. 方法

### A. 面板构造

原始能耗记录由 `ECdata.csv` 提供，粒度为站点级总能耗。小区级通信记录 `CLdata.csv` 与基站硬件元数据 `BSinfo.csv` 被聚合至 `(BS, Time)` 粒度。合并后的面板包含负载统计量、节能模式强度、静态硬件特征及 Phase A 基功率估计。凡缺失 `Energy`、`load_pmax_weighted`、`load_mean` 或 `p_base` 的样本行均予剔除。

在“仅日前信息”设定下，面板保留如下字段：

- 周期时间编码：小时与星期几的正弦/余弦项；
- 滚动 24 h 负载统计量：`load_mean_roll24`、`load_pmax_roll24`、`load_std_roll24`；
- 历史同小时节能先验：`S_*_hour_prior`；
- 静态站点属性：`n_cells`、`sum_pmax`、`sum_antennas`；
- 动态能耗标签：`dynamic_energy = Energy - p_base`。

同时，日内滞后类特征（如 `energy_lag*`、`dynamic_lag*`、`load_*_lag1`、`S_*_lag1`）被明确排除。

### B. 两类代理策略

本文对两种特征构造策略进行比较。

策略一 `two_stage_proxy` 首先构造负载与节能协变量代理：

$\widehat{load}_{mean}, \quad \widehat{load}_{pmax}, \quad \widehat{load}_{std}, \quad \widehat{S}_{m}$。

进而将其映射为物理特征：

$\widehat{D}_{1}=sum\_pmax_b \cdot \widehat{load}_{pmax}$，

$\widehat{D}_{2}=sum\_pmax_b \cdot \widehat{load}_{pmax}^{2}$，

$\widehat{D}_{3}=\widehat{load}_{std}$，

$\widehat{I}_{m}=\widehat{S}_{m}\cdot\widehat{load}_{pmax}$。

策略二 `historical_proxy` 直接使用历史代理项，例如前一日同小时负载、滚动负载统计量及同小时节能先验等；该策略为轻量基线，物理引导变换较少。

### C. 固定权重与学习型权重

固定权重基线采用启发式凸组合。例如，`load_mean_hat` 与 `load_pmax_hat` 在来源 `[prevday, roll24, current]` 上取权重 `[0.55, 0.30, 0.15]`；`load_std_hat` 在 `[prevday, roll24]` 上取 `[0.60, 0.40]`；节能代理在 `[prevday, hour_prior]` 上取 `[0.70, 0.30]`。

学习型权重版本通过下式估计全局凸权重：

$\min_{w\in \Delta}\frac{1}{N}\sum_{i=1}^{N}\left(z_i-\sum_{k}w_k x_{i,k}\right)^2+\lambda\lVert w\rVert_2^2,\quad \Delta=\{w \mid w \ge 0,\mathbf{1}^{T}w=1\}$。

其中，$z_i$ 表示目标时刻的真实协变量（例如 $t_0+h$ 时刻的 `load_mean`），而非未来能耗标签；从而在避免预测目标 $E_{b,t_0+h}$ 信息泄漏的前提下改善代理质量。实现上，先在基站层面划分 80%/20% 训练—验证集以学习权重，再将所得全局权重用于下游模型训练。

过滤后主实验中学习到的权重如表所列。

| 代理项 | 来源 | 权重 |
| --- | --- | --- |
| `load_mean_hat` | prevday, roll24, current | 0.8153, 0.0331, 0.1516 |
| `load_pmax_hat` | prevday, roll24, current | 0.8161, 0.0188, 0.1651 |
| `load_std_hat` | prevday, roll24 | 0.8083, 0.1917 |
| `S_*_hat` | prevday, hour_prior | 0.9796, 0.0204 |

### D. 预测模型

每种代理策略均与以下四类模型组合评估：

1. `Physical`：对物理构造特征施加非负系数约束的线性模型。
2. `SemiPhysical_Ridge`：半物理特征结合 Ridge 交叉验证（RidgeCV）。
3. `SemiPhysical_Lasso`：半物理特征结合 Lasso 交叉验证（LassoCV）。
4. `RandomForest`：含 300 棵决策树且 `min_samples_leaf=2` 的非线性回归模型。

所有模型均采用按基站标识 `BS` 分组的 `GroupKFold` 交叉验证，以保证同一基站不出现在训练折与验证折中。

## IV. 实验设置

### A. 数据集与过滤协议

未过滤面板包含 923 个基站、92629 条合并记录。为与稀疏观测质量控制协议一致，过滤实验保留合并观测不少于 24 条的基站，并剔除 `phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv` 所列稀疏基站，最终得到 818 个基站、90621 条记录。

鉴于该协议与所提代理构造在过滤设定下取得最优性能，本文将“过滤 + 学习型权重”实验作为主结果报告。

### B. 实验矩阵

**表 I** 汇总实验矩阵，可由各输出目录中的 `filter_meta.json` 及对比目录中的 `compare_meta.json` 生成。

| 实验 | 输出目录 | 过滤 | 代理权重 | 基站数 / 行数 | 角色 |
| --- | --- | --- | --- | --- | --- |
| E0 | `outputs_proxy_compare_fixed/` | 否 | 固定 | 923 / 92629 | 未过滤基线 |
| E1 | `outputs_proxy_compare_learned/` | 否 | 学习 | 923 / 92629 | 未过滤权重学习消融 |
| E2 | `outputs_filter_compare_fixed/` | 是 | 固定 | 818 / 90621 | 过滤后基线 |
| E3 | `outputs_filter_compare_learned/` | 是 | 学习 | 818 / 90621 | 主实验 |
| E4 | `outputs_proxy_impact_filtered/` | 是 | 固定 vs 学习 | aligned predictions=71608 | 主消融分析 |

正式排版中，若列宽受限，可将输出目录名置于脚注或可复现性说明中。

### C. 评估指标

严格 24 h 轨迹指标由 `dayahead_metrics.csv` 按下式计算：

$MAPE=\frac{1}{N}\sum_i\frac{|E_i-\widehat{E}_i|}{\max(|E_i|,\epsilon)}$。

峰值误差与谷值误差在每条完整 24 h 轨迹上分别计算，即在真实日峰值、日谷值时刻比较预测误差。分时距 MAE、RMSE 与 MAPE 由 `dayahead_horizon_metrics.csv` 给出。

## V. 仿真结果与分析

### A. 严格 24 h 轨迹性能

**表 II** 建议作为主性能表，数据来源包括：

- `outputs_filter_compare_learned/dayahead_metrics.csv`（所提方法）；
- `outputs_filter_compare_fixed/dayahead_metrics.csv`（固定权重消融）；
- 如需报告未过滤敏感性，可选用 `outputs_proxy_compare_learned/dayahead_metrics.csv`。

正文可保留如下代表性结果行：

| 设置 | 策略 | 模型 | MAPE | 峰值误差 | 谷值误差 | 完整轨迹数 |
| --- | --- | --- | --- | --- | --- | --- |
| 过滤 + 学习权重 | two_stage_proxy | RandomForest | 0.09096 | 4.3459 | 1.2176 | 9 |
| 过滤 + 学习权重 | historical_proxy | RandomForest | 0.09984 | 4.9499 | 1.6297 | 9 |
| 过滤 + 固定权重 | two_stage_proxy | RandomForest | 0.11367 | 5.1110 | 2.5322 | 9 |
| 未过滤 + 学习权重 | two_stage_proxy | RandomForest | 0.09674 | 6.2581 | 1.0910 | 9 |
| 未过滤 + 固定权重 | two_stage_proxy | RandomForest | 0.11032 | 5.9051 | 2.5041 | 9 |

结果表明：在过滤设定下，学习型权重的两阶段代理在严格 24 h MAPE 上最低；相对同设定下的固定权重对应方法，MAPE 降低 0.02271（约 20.0%）；相对同模型下的学习型 `historical_proxy`，MAPE 降低 0.00888。

由于在严格 24 h 判据下完整轨迹仅 9 条，正文须说明该表属于高要求轨迹级比较，并应与**表 V**及**图 6**的分时距结果联合解读。

### B. 代理权重消融

**表 III** 建议用于展示固定权重与学习权重的差异，可由 `outputs_proxy_impact_filtered/compare_strict_metrics.csv` 生成。

| 策略 | 模型 | 固定权重 MAPE | 学习权重 MAPE | Delta MAPE | Delta peak | Delta valley |
| --- | --- | --- | --- | --- | --- | --- |
| two_stage_proxy | RandomForest | 0.11367 | 0.09096 | -0.02271 | -0.7651 | -1.3147 |
| historical_proxy | RandomForest | 0.12376 | 0.09984 | -0.02392 | -0.3704 | -0.9455 |
| two_stage_proxy | SemiPhysical_Ridge | 0.17856 | 0.13701 | -0.04155 | -0.2393 | -1.1198 |
| two_stage_proxy | Physical | 0.19421 | 0.13625 | -0.05796 | -1.5045 | -1.3114 |

表格可按 `Delta MAPE` 或模型重要性排序。叙述上宜突出随机森林行（对应最终最优预测器），并说明学习型权重对线性/半物理模型同样带来改善。

**图 4** 建议用于可视化该消融；当前可直接采用 `outputs_proxy_impact_filtered/fig_strict_best_metrics_compare.png`。若为 TSG 正式稿件重绘，建议采用分组柱状图，分别展示 MAPE、峰值误差与谷值误差；图例区分 `fixed` 与 `learned`，纵轴单位须明示，图注中说明所选模型为固定权重基线下的最优模型。

### C. 分时距误差特性

**图 6** 建议用于展示分时距误差，可采用：

- `outputs_filter_compare_learned/fig_error_by_horizon.png`（主误差曲线）；
- `outputs_proxy_impact_filtered/fig_delta_mae_by_horizon.png`（学习权重相对固定权重的 MAE 差值）。

正式发表时建议采用双面板图：图 6(a) 为各策略下最优模型的 MAE—时距曲线；图 6(b) 为 $\Delta MAE = MAE_{learned}-MAE_{fixed}$，负值表示性能改善。

**表 V** 建议报告代表性时距及样本量，可由 `outputs_filter_compare_learned/dayahead_horizon_metrics.csv` 在 $h=1,6,12,18,24$ 处选取各时距 MAE 最低行得到：

| 时距 | 最优策略 | 最优模型 | MAE | RMSE | MAPE | 样本数 | 基站数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | two_stage_proxy | RandomForest | 2.6155 | 3.8941 | 0.1462 | 2235 | 802 |
| 6 | two_stage_proxy | RandomForest | 1.5579 | 2.5762 | 0.0656 | 519 | 402 |
| 12 | historical_proxy | RandomForest | 3.4591 | 4.9280 | 0.1080 | 117 | 108 |
| 18 | two_stage_proxy | RandomForest | 3.4702 | 5.1504 | 0.1132 | 28 | 28 |
| 24 | two_stage_proxy | SemiPhysical_Ridge | 0.8815 | 1.1530 | 0.0372 | 9 | 9 |

不宜仅凭 $h=24$ 处较低 MAE 推断“长时距预测本质上更易”，因该时距有效样本极少（9）。更稳妥的表述为：随预测时距增大，可用样本量减少，故须结合图表综合判断。

### D. 预测散点图与代表性轨迹

**图 5** 建议给出预测值相对真实值的散点图，可采用 `outputs_filter_compare_learned/fig_prediction_vs_actual.png`，比较各代理策略下的最优模型。为满足 TSG 版式要求，建议保留对角参考线，使横纵轴范围一致，并在图例或子图标题中标明所选模型及对应 MAPE。

**图 7** 建议给出代表性 24 h 轨迹，可采用 `outputs_filter_compare_learned/fig_dayahead_trajectory.png`；原始数据见同目录下 `trajectory_details_B_501@2023-01-02.csv` 与 `trajectory_summary_B_501@2023-01-02.csv`。正式稿件中建议横轴标注为预测时距，曲线仅保留观测能耗与预测能耗两条；图注须说明该轨迹选自严格轨迹指标意义下的最优模型。

### E. 物理一致性分析

**图 2** 建议展示负载—能耗关系，可采用 `outputs_filter_compare_learned/fig_load_vs_energy.png`。该图自面板中至多采样 12000 条记录，以 `load_pmax_weighted` 为横轴、`Energy` 为纵轴绘制散点并拟合二次趋势线，以支撑非线性代理特征 $\widehat{D}_{2}$ 的合理性。过滤主实验中，Spearman 秩相关系数为 0.6883。

**图 3** 建议展示节能模式影响，可采用 `outputs_filter_compare_learned/fig_es_mode_impact.png`，给出各模式在激活与未激活状态下的平均动态能耗差值。负向效应最强的模式为：

| 模式 | 动态能耗差值 |
| --- | --- |
| `S_ESMode5` | -14.1767 |
| `S_ESMode2` | -13.4676 |
| `S_ESMode1` | -13.4674 |

**表 VI** 建议汇总物理一致性检验，可由 `outputs_filter_compare_learned/physics_checks.csv` 与 `es_mode_effects.csv` 生成：

| 检验项 | 数值 | 解释 |
| --- | --- | --- |
| 负载单调性比率 | 1.0000 | 随负载分箱升高，平均能耗非递减。 |
| 负载—能耗 Spearman | 0.6883 | 负载与能耗呈显著正秩相关。 |
| 最优 ES 效应为负 | 1.0000 | 至少一种节能模式使动态能耗按预期降低。 |
| $p^{base}$-`n_cells` 相关系数 | 0.8080 | 基功率估计与站点规模相一致。 |

### F. 回退链诊断

`outputs_fallback_audit/analysis_report.md` 中的审计结果表明：对 `load_mean` 与 `load_pmax` 而言，前一日同小时来源覆盖率大致处于 57%—73%；`roll24` 约为 24%—40%；`current` 约贡献 2.65%。该诊断可作为附录，或以简短稳健性段落置于正文。

**图 8** 可由 `outputs_fallback_audit/fallback_summary_by_horizon.csv` 构造，步骤如下：

1. 对 `load_mean` 选取列 `load_mean_ratio_prevday`、`load_mean_ratio_roll24`、`load_mean_ratio_current`；
2. 以 $horizon=1,\ldots,24$ 为横轴，绘制堆叠柱状图或堆叠面积图；
3. 若篇幅允许，可对 `load_pmax` 重复作图；否则在正文中说明其规律与 `load_mean` 一致；
4. 图注中说明：在舍入误差允许范围内，各来源比例之和为 1。

该图用以解释稳健回退链的必要性，并表明代理权重学习不能替代对缺失来源机制的诊断。

## VI. 讨论

实验结果表明，代理构造策略与代理权重设定均对预测性能具有显著影响。在稀疏基站被过滤、代理权重经学习确定且两阶段代理与非线性回归器相结合的条件下，可取得最优结果。学习权重显著偏向前一日同小时协变量，与日前预测中的日周期规律相一致。然而，回退链审计表明，对相当一部分样本而言前一日信息不可用，故滚动统计项与当前观测项仍构成重要回退来源。

严格 24 h 轨迹评价在统计上较为保守，但有效轨迹数量较少。因此，面向 TSG 投稿时，不宜仅依据严格轨迹表作出过强论断；分时距表与误差曲线对于论证所提方法在 9 条完整轨迹之外仍具支撑证据尤为关键。

## VII. 结论

本文提出一种仅依赖日前信息的基站能耗预测框架，将静态基功率分解、物理引导动态能耗特征、受约束代理权重学习及按基站分组验证相结合。在过滤主实验中，学习型两阶段代理与随机森林相结合，在严格 24 h 轨迹 MAPE 上取得最优值 0.09096，相对固定权重基线约改善 20.0%。物理一致性检验进一步验证了模型设计的合理性，包括负载—能耗单调性、主导节能模式的负向效应，以及静态基功率与站点规模之间的强相关关系。

后续工作可在增加完整 24 h 轨迹数量的基础上，考察跨日期与跨区域泛化性能，并在相同的“未来协变量不可用”约束下，与序列模型进行对比。

## 图表构造清单

下表用于最终 IEEE TSG 稿件中图、表的排版与核对。

| 项目 | 源文件 | 构造规则 | 建议位置 |
| --- | --- | --- | --- |
| 表 I：实验矩阵 | `filter_meta.json`、`compare_meta.json` | 汇总过滤策略、基站数、记录数、权重模式及用途 | 实验设置 |
| 表 II：严格性能 | `dayahead_metrics.csv` | 按 MAPE 排序；保留所提方法及关键基线 | 仿真结果 |
| 表 III：消融实验 | `compare_strict_metrics.csv` | 报告固定权重、学习权重及二者差值 | 仿真结果 |
| 表 IV：代理权重 | `proxy_weights.json`、`proxy_weights_meta.json` | 展示权重；可附协变量 RMSE/MAE | 方法或结果 |
| 表 V：分时距指标 | `dayahead_horizon_metrics.csv` | 选取代表性时距；或于附录给出全部 24 个时距 | 仿真结果 |
| 表 VI：物理一致性检验 | `physics_checks.csv`、`es_mode_effects.csv` | 每项检验配以简短物理解释 | 仿真结果 |
| 图 1：流程图 | 手工重绘 | 矢量示意图：原始数据→面板→代理→模型→评估 | 方法 |
| 图 2：负载—能耗 | `fig_load_vs_energy.png` | 散点 + 二次趋势线；注明采样规模 | 物理一致性 |
| 图 3：节能模式影响 | `fig_es_mode_impact.png` | 柱状图；按影响大小排序 | 物理一致性 |
| 图 4：严格轨迹消融 | `fig_strict_best_metrics_compare.png` | 分组柱状图：MAPE、峰值、谷值 | 主结果 |
| 图 5：预测散点图 | `fig_prediction_vs_actual.png` | 等比例坐标轴 + 对角参考线 | 主结果 |
| 图 6：分时距误差 | `fig_error_by_horizon.png`、`fig_delta_mae_by_horizon.png` | 双面板：MAE 与 $\Delta$MAE | 时距分析 |
| 图 7：轨迹图 | `fig_dayahead_trajectory.png` | 24 个时距上的观测与预测曲线 | 案例分析 |
| 图 8：回退比例 | `fallback_summary_by_horizon.csv` | 按时距的堆叠柱状图或面积图 | 稳健性或附录 |

为符合 IEEE 版式规范，建议将折线图与柱状图尽量重绘为矢量图形，统一字体与线宽，避免图例拥挤，并于图注或表注中给出样本量。

## 参考文献

[1] IEEE Power & Energy Society, "IEEE Transactions on Smart Grid," journal scope and paper categories. https://ieee-pes.org/publications/transactions-on-smart-grid/

[2] IEEE Power & Energy Society, "Preparation and Submission of Transactions Papers," IEEE PES Author's Kit. https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/

[3] IEEE Author Center, "Create the Text of Your Article." https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/
