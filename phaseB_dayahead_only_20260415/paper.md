## X.X 日前预测训练流程与结果（仅日前，含稀疏观测一致性过滤）

### X.X.1 训练数据构建与样本筛选

本文以基站级整站能耗 `Energy`（EC）作为监督标签，并将小区级负载与节能状态（CL）在$ BS\text{-}Time$ 粒度聚合后与 EC 进行内连接，形成可用于日前预测的短面板样本。为保证“稀疏观测策略 A”的口径一致性，训练入口引入基站级最小观测数过滤：在完成 `Energy` / `load` / `p_base` 等关键字段的缺失剔除与必要的时间特征构建后，对每个基站统计 merged 样本行数，保留满足 `min_merged_obs_per_bs = 24` 的基站（过滤前 923 站、92629 行；过滤后 818 站、90621 行）。过滤元信息见 `phaseB_dayahead_only_20260415/outputs_filter/filter_meta.json`。

此外，日前任务以每日 `00:00` 为起报时刻，对未来 24 小时逐步预测动态项，并通过连续性校验确保每个预测步长对应的目标时刻满足严格整小时间隔（与 `make_continuity_mask` 一致），以避免跨越缺测间隔导致的标签错配。

### X.X.2 特征设计与模型训练

为匹配“仅日前”的信息可得性，本文采用两类代理特征策略，并在每个策略下训练 4 类模型（Physical / SemiPhysical_Ridge / SemiPhysical_Lasso / RandomForest），使用基站分组交叉验证（GroupKFold by BS）评估泛化性能。

- **Two-stage proxy（两阶段代理）**：先构造负载与 ES 的多源代理（如昨日同小时、滚动 24h 统计与历史同小时先验的加权组合），得到 $\hat{D}_1,\hat{D}_2,\hat{D}_3$ 以及 $\hat{S}_m,\hat{I}_m$ 等输入；在缺乏未来真实负载/节能状态的前提下，保持物理解释量的可用性。
- **Historical proxy（历史代理）**：以更轻量的历史代理（如昨日/上周同小时、滚动统计与同小时历史先验）直接形成日前可用的负载侧输入与 $ES$ 的同小时先验项。

评价指标采用能耗预测误差（以 $\hat{E} = p_{base} + \hat{y}$ 还原到整站能耗）：严格 24 小时轨迹层面的 MAPE 及峰谷误差（`dayahead_metrics.csv`），以及按预测步长聚合的 MAE / RMSE / MAPE（`dayahead_horizon_metrics.csv`）。

### X.X.3 图表插入位置与结果分析

**图 X — 负载与能耗关系（`fig_load_vs_energy.png`）**  
*建议位置：本小节中部或结果章节开头。*

![Load vs Energy](phaseB_dayahead_only_20260415/outputs_filter/fig_load_vs_energy.png)

负载聚合指标与整站能耗呈显著正相关：Spearman 相关系数为 0.6883（`physics_checks.csv`）。散点与二次趋势线共同提示中高负载区间存在非线性，为日前代理中保留 \(\hat{D}_2\)（二次项）提供经验支持。

---

**图 Y — ES 模式影响（`fig_es_mode_impact.png`）**  
*建议位置：紧随图 X。*

![ES mode impact](phaseB_dayahead_only_20260415/outputs_filter/fig_es_mode_impact.png)

图中为负载分层后“激活 vs 未激活”对应的动态能耗均值差。物理一致性检查中 `best_es_effect_is_negative = 1.0`；
Top-3 模式差异约为 $S_{ESMode5}=-14.18$、$S_{ESMode2}=-13.47$、$S_{ESMode1}=-13.47$（`analysis_report.md`），与节能模式应降低动态能耗的方向一致。

---

**图 Z — 日前预测轨迹示例（`fig_dayahead_trajectory.png`）**  
*建议位置：模型对比段落之后。*

![Day-ahead trajectory](phaseB_dayahead_only_20260415/outputs_filter/fig_dayahead_trajectory.png)

该图展示某一完整 24h 轨迹上真实能耗与预测能耗随预测步长的变化，便于讨论峰谷时段误差结构；与 MAPE 互补，可补充说明峰值/谷值误差（见 `dayahead_metrics.csv`）。

---

**图 W — 预测–真实散点（`fig_prediction_vs_actual.png`）**  
*建议位置：紧随图 Z。*

![Prediction vs actual](phaseB_dayahead_only_20260415/outputs_filter/fig_prediction_vs_actual.png)

左右子图分别对应两种代理策略下所选最佳模型的散点。整体上 **two-stage proxy** 点云更贴近对角线，表明在日前信息约束下，两阶段代理对负载/节能不确定性的刻画优于单一历史代理路径。

---

**图 V — 按预测步长的误差（`fig_error_by_horizon.png`）**  
*建议位置：本小节末。*

![Error by horizon](phaseB_dayahead_only_20260415/outputs_filter/fig_error_by_horizon.png)

按步长 $h=1,\ldots,24$ 汇总并比较各策略下误差曲线。单步 MAE 最优出现在 \(h=6\)，对应 **two-stage proxy + RandomForest**（`analysis_report.md`），说明短期日前步长更依赖对日内周期与近期历史的有效代理。

### X.X.4 量化结果总结

在严格 24h 轨迹层面，最佳方案为 **two-stage proxy + RandomForest**：MAPE = 0.1126，峰值误差 = 5.1656，谷值误差 = 2.4432（`dayahead_metrics.csv`）。作为对照，**historical_proxy + RandomForest** 的 MAPE = 0.1212，表明在同一模型族下两阶段代理带来稳定收益。此外，负载单调分箱比例为 100%，`p_base` 与 `n_cells` 的 Pearson 相关为 0.8079（`physics_checks.csv`），表明静态分解与动态建模在数据层面具有一致性。

---

**说明（排版）**：若论文使用 LaTeX，可将 `![...](...)` 改为 `\includegraphics`；图号 **X/Y/Z/W/V** 请按全文统一编号替换。