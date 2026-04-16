# Phase B Day-Ahead Energy Forecasting

本目录是 Phase B 的“仅日前预测”实验版本。任务是在每天 `00:00` 起报，对未来
`h = 1,...,24` 小时的基站整站能耗 `Energy` 进行预测。实现上先沿用 Phase A 的
静态基线功率 `p_base`，再预测动态能耗项：

```text
Energy_hat(b,t0+h) = p_base(b) + dynamic_energy_hat(b,t0+h)
```

主脚本为 [run_phaseB_dayahead_only.py](run_phaseB_dayahead_only.py)。

## 1. 本阶段做了哪些实验

当前目录中已经保留了多组输出。建议把 `outputs_filter_compare_learned/` 作为论文主结果，
其余目录用于消融、鲁棒性和诊断分析。

| 实验编号 | 输出目录 | BS 过滤 | 代理权重 | 目的 | 最佳严格 24h 结果 |
| --- | --- | --- | --- | --- | --- |
| E0 | `outputs/` 或 `outputs_proxy_compare_fixed/` | 无；923 个 BS，92629 行 | 固定权重 | 全量基线 | `two_stage_proxy + RandomForest`, MAPE=0.1103 |
| E1 | `outputs_proxy_compare_learned/` | 无；923 个 BS，92629 行 | 学习权重 | 全量权重学习消融 | `two_stage_proxy + RandomForest`, MAPE=0.0967 |
| E2 | `outputs_filter_compare_fixed/` | `min_merged_obs_per_bs=24`，剔除 105 个稀疏 BS；818 个 BS，90621 行 | 固定权重 | 稀疏观测一致性过滤基线 | `two_stage_proxy + RandomForest`, MAPE=0.1137 |
| E3 | `outputs_filter_compare_learned/` | 同 E2 | 学习权重 | 推荐主实验 | `two_stage_proxy + RandomForest`, MAPE=0.0910 |
| E4 | `outputs_proxy_impact/` | 无 | 固定 vs 学习对齐比较 | 权重学习影响分析 | 生成 delta 表和对比图 |
| E5 | `outputs_proxy_impact_filtered/` | 同 E2/E3 | 固定 vs 学习对齐比较 | 推荐主消融分析 | `two_stage_proxy + RandomForest` 的 MAPE 下降 0.0227 |
| E6 | `outputs_fallback_audit/` | 使用指定 panel 审计 | 不训练模型 | 统计 `prevday -> roll24 -> current` 回退链 | 诊断代理特征可靠性 |

注意：`outputs_filter/`、`outputs_filter_fixed/`、`outputs_filter_learned/` 当前只保留了早期
`analysis_report.md` 快照，不是本次论文主表应引用的完整结果目录。

## 2. 核心实验设计

数据构建流程：

1. 读取 `ECdata.csv`、`CLdata.csv`、`BSinfo.csv`。
2. 将小区级负载和节能状态聚合到 `(BS, Time)` 粒度。
3. 合并 Phase A 的 `p_base`，形成基站级 panel。
4. 删除日内滚动预测需要的滞后项，仅保留日前可用特征：
   `load_*_roll24`、昨日同小时代理、同小时历史先验、时间周期编码和硬件静态特征。
5. 每个 `origin_time=00:00` 构造 `h=1,...,24` 的目标样本，并用严格整小时间隔校验避免跨缺测间隔。

比较的两类日前代理策略：

- `two_stage_proxy`：先构造 `load_mean_hat`、`load_pmax_hat`、`load_std_hat`、`S_*_hat`，
  再形成物理解释量 `D1_hat`、`D2_hat`、`D3_hat` 和 `I_*_hat`。
- `historical_proxy`：直接使用昨日/上周同小时、滚动统计和同小时历史先验作为代理特征。

每种策略训练 4 类模型：

- `Physical`：带正系数约束的线性模型。
- `SemiPhysical_Ridge`：半物理特征 + RidgeCV。
- `SemiPhysical_Lasso`：半物理特征 + LassoCV。
- `RandomForest`：非线性集成模型，300 棵树，`min_samples_leaf=2`。

评估口径：

- `dayahead_metrics.csv`：严格 24 小时完整轨迹口径，报告 MAPE、峰值误差、谷值误差。
- `dayahead_horizon_metrics.csv`：按预测步长 `horizon` 汇总 MAE、RMSE、MAPE。
- 交叉验证使用 `GroupKFold by BS`，避免同一基站同时出现在训练和验证折。

## 3. 主要结论

推荐主结果来自 `outputs_filter_compare_learned/`：

- 最佳模型为 `two_stage_proxy + RandomForest`，严格 24h 轨迹 MAPE=0.09096，峰值误差=4.3459，谷值误差=1.2176。
- 相对过滤版固定权重基线，`two_stage_proxy + RandomForest` 的 MAPE 从 0.11367 降到 0.09096，
  绝对下降 0.02271，相对下降约 20.0%。
- 相对过滤版 `historical_proxy + RandomForest`，推荐模型的 MAPE 从 0.09984 降到 0.09096，
  说明两阶段代理在同一模型族下仍有收益。
- 物理一致性检查显示：负载分箱单调比例为 100%，`load_pmax_weighted` 与 `Energy` 的
  Spearman 相关为 0.6883，`p_base` 与 `n_cells` 的相关为 0.8080。
- ES 模式影响方向基本符合节能预期：`S_ESMode5=-14.1767`、`S_ESMode2=-13.4676`、
  `S_ESMode1=-13.4674`。

结果使用时需要说明一个限制：严格 24h 完整轨迹目前只有 9 条，适合作为完整日轨迹口径的
严格比较；按 horizon 的样本量更大，但会随 `h` 增大明显减少，例如过滤主实验中
`h=1` 有 2235 个样本，`h=24` 只有 9 个样本。因此论文图表必须同时报告 `n_samples`。

## 4. 代理权重学习

启用 `--learn-proxy-weights` 后，脚本只用目标时刻真实协变量监督学习代理权重，不使用未来
`Energy` 标签。过滤主实验 `outputs_filter_compare_learned/proxy_weights.json` 中的权重为：

| 代理变量 | 候选源顺序 | 学习权重 |
| --- | --- | --- |
| `load_mean_hat` | prevday, roll24, current | 0.8153, 0.0331, 0.1516 |
| `load_pmax_hat` | prevday, roll24, current | 0.8161, 0.0188, 0.1651 |
| `load_std_hat` | prevday, roll24 | 0.8083, 0.1917 |
| `S_*_hat` | prevday, hour_prior | 0.9796, 0.0204 |

这组权重表明，在当前过滤口径下，昨日同小时信息是主导来源，`current` 提供少量兜底，
`roll24` 主要承担缺失回退和稳定化作用。

## 5. 复现实验命令

建议从仓库根目录 `energy_model_anp/` 运行命令。

全量固定权重：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --output-subdir outputs_proxy_compare_fixed
```

全量学习权重：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --learn-proxy-weights \
  --output-subdir outputs_proxy_compare_learned
```

过滤固定权重：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv \
  --output-subdir outputs_filter_compare_fixed
```

过滤学习权重，推荐主实验：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv \
  --learn-proxy-weights \
  --output-subdir outputs_filter_compare_learned
```

对齐比较固定权重和学习权重：

```bash
python phaseB_dayahead_only_20260415/analyze_proxy_weight_impact.py \
  --fixed-dir phaseB_dayahead_only_20260415/outputs_filter_compare_fixed \
  --learned-dir phaseB_dayahead_only_20260415/outputs_filter_compare_learned \
  --out-dir phaseB_dayahead_only_20260415/outputs_proxy_impact_filtered
```

回退链审计：

```bash
python phaseB_dayahead_only_20260415/fallback_audit/analyze_fallback_usage.py \
  --panel phaseB_dynamic_energy_20260414_rebuild/outputs/panel_dataset.csv \
  --outdir phaseB_dayahead_only_20260415/outputs_fallback_audit
```

## 6. 输出文件怎么读

每个完整实验输出目录中主要文件如下：

- `filter_meta.json`：BS 过滤、输出目录、代理权重模式等元信息。
- `panel_dataset.csv`：训练/评估用 panel。
- `dayahead_predictions.csv`：每个 `(strategy, model, trajectory_id, horizon)` 的预测、真值和 `p_base`。
- `dayahead_metrics.csv`：严格 24h 完整轨迹指标。
- `dayahead_horizon_metrics.csv`：按预测步长的误差指标。
- `model_coefficients.csv`：线性模型系数或 RandomForest 特征重要性。
- `physics_checks.csv`：负载单调性、负载-能耗相关、ES 方向、`p_base` 与站点规模相关性。
- `es_mode_effects.csv`：各 ES 模式激活与未激活的动态能耗均值差。
- `proxy_weights.json`、`proxy_weights_meta.json`：仅学习权重实验生成。

主要图片：

- `fig_load_vs_energy.png`：负载与能耗散点及二次趋势线。
- `fig_es_mode_impact.png`：不同 ES 模式的动态能耗差。
- `fig_prediction_vs_actual.png`：两类策略最佳模型的预测-真实散点。
- `fig_error_by_horizon.png`：按 `horizon` 的最佳 MAE 曲线。
- `fig_dayahead_trajectory.png`：一条完整 24h 轨迹示例。
- `outputs_proxy_impact_filtered/fig_strict_best_metrics_compare.png`：固定权重 vs 学习权重对比。
- `outputs_proxy_impact_filtered/fig_delta_mae_by_horizon.png`：学习权重相对固定权重的 MAE 差值。
- `outputs_proxy_impact_filtered/fig_energy_pred_delta_hist.png`：同一预测点上学习权重与固定权重预测差分布。

## 7. 论文图表建议

论文正文建议使用以下表和图：

| 类型 | 建议编号 | 数据源 | 展现目的 |
| --- | --- | --- | --- |
| 表 | Table I | `filter_meta.json`、`compare_meta.json` | 实验组、过滤设置、样本规模和输出目录 |
| 表 | Table II | `dayahead_metrics.csv` | 严格 24h 轨迹性能，突出主模型和对照模型 |
| 表 | Table III | `compare_strict_metrics.csv` | 固定权重 vs 学习权重消融 |
| 表 | Table IV | `proxy_weights.json`、`proxy_weights_meta.json` | 学习到的代理权重及拟合误差 |
| 表 | Table V | `dayahead_horizon_metrics.csv` | 代表性 horizon 的 MAE/RMSE/MAPE 和样本数 |
| 表 | Table VI | `physics_checks.csv`、`es_mode_effects.csv` | 物理一致性检查 |
| 图 | Fig. 1 | 手工绘制流程图 | 数据构建、代理权重学习、模型评估流程 |
| 图 | Fig. 2 | `fig_load_vs_energy.png` | 说明负载-能耗关系和二次项动机 |
| 图 | Fig. 3 | `fig_es_mode_impact.png` | 展示 ES 模式方向是否符合物理预期 |
| 图 | Fig. 4 | `fig_strict_best_metrics_compare.png` | 固定权重与学习权重的主指标对比 |
| 图 | Fig. 5 | `fig_prediction_vs_actual.png` | 预测值与真实值贴合程度 |
| 图 | Fig. 6 | `fig_error_by_horizon.png` 和 `fig_delta_mae_by_horizon.png` | 误差随预测步长变化及消融收益 |
| 图 | Fig. 7 | `fig_dayahead_trajectory.png` | 代表性 24h 日前轨迹 |
| 图 | Fig. 8 | `fallback_summary_by_horizon.csv` 新绘制堆叠柱/面积图 | 显示 `prevday/roll24/current` 来源比例 |

构图时要在图注或表注中写清楚样本量，尤其是 `h=24` 和严格 24h 轨迹只有 9 条这一点。
