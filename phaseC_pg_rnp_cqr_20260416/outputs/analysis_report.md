# Phase C PG-RNP-CQR 分析报告

## 任务
本实验通过学习 Phase B 物理日前基线的残差，来预测 5G 基站的日前能耗。

## 数据
- 行数：8951
- 基站数：802
- 轨迹数：2235
- 完整 24 小时轨迹数：9
- 物理基线：`two_stage_proxy + Physical`

## 主要结果
- 按 MAPE 最优的测试集模型：`A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR`，MAPE=0.0732。
- PG-RNP-CQR 测试集：MAPE=0.0732，MAE=1.4293，按 horizon 平均覆盖率=0.838，平均区间宽度=7.952。

## 校准与可行性（物理约束）
- 分 horizon 的 CQR：对每个 horizon 单独学习一个残差分数分位数。
- 各 horizon 的平均 q_hat：2.6146
- 最终区间会被投影到 `[physical_lower, physical_upper]`；细节见 `physics_violation_report.csv`。

## 输出文件
- `residual_trajectory_dataset.csv`
- `pg_rnp_predictions.csv`
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `coverage_by_horizon.csv`
- `ablation_metrics.csv`
- `physics_violation_report.csv`

## 各输出文件内容含义（详细说明）

### 1) `residual_trajectory_dataset.csv`（Phase C 样本母表）
- **每行代表**：一个 `(BS, origin_time, horizon)` 样本（某基站在某次日前起点，对应某个预测步长的一条记录）。
- **关键字段**：
  - **定位键**：`BS, trajectory_id, origin_time, target_time, horizon`
  - **真值**：`energy_true`
  - **物理基线**：`E_phys_hat`（Phase B 的 `two_stage_proxy + Physical` 总能耗预测）、`dynamic_phys_hat`
  - **残差标签**：`residual_true = energy_true - E_phys_hat`
  - **物理可行域**：`physical_lower, physical_upper`
  - **轨迹完整性**：`is_complete_24`（该 `trajectory_id` 是否覆盖 1..24）
- **用途**：训练/推理/校准/评估的输入数据集（Phase C 的所有下游表基本都以它为基础扩展）。

### 2) `split_bs.json`（按基站划分数据集）
- **内容**：`train / calibration / test` 三个 key，对应三组 **基站 ID 列表**。
- **用途**：避免同一基站跨集合导致泄漏；`pg_rnp_predictions.csv` 中的 `split` 就来自该划分。

### 3) `pg_rnp_predictions.csv`（预测主表：点预测 + 区间）
- **每行代表**：同 `residual_trajectory_dataset.csv` 的一条样本行（并新增模型输出与校准区间）。
- **关键字段**：
  - **集合标识**：`split`（train/calibration/test）
  - **模型输出（残差分布参数）**：`residual_mu, residual_sigma`
  - **点预测**：`energy_pred = E_phys_hat + residual_mu`
  - **CQR 系数**：`cqr_q`（对应该行 `horizon` 的 \(q_h\)）
  - **区间（投影前）**：`lower_90_raw, upper_90_raw`
  - **区间（最终，投影后）**：`lower_90, upper_90`
  - **物理可行域**：`physical_lower, physical_upper`（用于投影/裁剪）
- **用途**：做轨迹可视化、覆盖率/宽度统计、误差诊断的主要数据表。

### 4) `cqr_quantiles.json`（每个 horizon 的 \(q_h\)）
- **内容**：键为 `1..24`（horizon），值为该 horizon 的 \(q_h\)。
- **含义**：在 **calibration split** 上计算的“标准化残差误差分数”的 \(1-\alpha\) 分位数，用于把 `residual_sigma` 转为 90% 区间半宽：
  - `E_phys_hat + residual_mu ± q_h * residual_sigma`
- **用途**：解释不同 horizon 的区间宽度差异；复现实验的校准参数记录。

### 5) `metrics_overall.csv`（总体点预测指标对比）
- **每行代表**：一个模型（含 Phase B 基线与 Phase C PG-RNP-CQR）在测试集上的总体指标。
- **关键字段**：`MAE, RMSE, MAPE, strict_24h_trajectory_MAPE, peak_error, valley_error, n_samples, n_bs, n_trajectories`
- **用途**：写报告/论文时做“整体性能对比”的主表。

### 6) `ablation_metrics.csv`（消融/对照指标汇总）
- **内容**：与 `metrics_overall.csv` 同源的对照汇总表（便于以“消融”命名引用）。
- **用途**：集中对比 PhaseB 与 PhaseC（以及不同 PhaseB 组合）的整体指标。

### 7) `metrics_by_horizon.csv`（按 horizon 的误差与区间质量）
- **每行代表**：一个 horizon（1..24）的统计结果。
- **关键字段**：
  - **区间质量**：`coverage_90`（真实值落入 `[lower_90, upper_90]` 的比例）、`avg_width_90`（平均区间宽度）
  - **点误差**：`mae, rmse, mape`
  - **规模**：`n_samples, n_bs`
- **用途**：分析“步长越远是否更难”“覆盖率是否随 horizon 变化”。

### 8) `coverage_by_horizon.csv`（按 horizon 覆盖率表）
- **内容**：`metrics_by_horizon.csv` 的拷贝（文件名更强调 coverage）。
- **用途**：绘图或引用时更直观。

### 9) `physics_violation_report.csv`（物理边界违规/投影修正统计）
- **每行代表**：一个 split（train/calibration/test）。
- **关键字段**：
  - **点预测越界率**：`energy_pred_below_lower_rate, energy_pred_above_upper_rate`
  - **原始区间越界率**：`raw_interval_below_lower_rate, raw_interval_above_upper_rate`（投影前 `lower_90_raw/upper_90_raw`）
  - **投影修正幅度**：`avg_projection_delta`（投影前后端点平均改动）
- **用途**：评估物理可行域约束对输出的影响强度；检查是否频繁发生裁剪。

### 10) `dataset_meta.json`（数据集构建元信息）
- **内容**：数据来源（PhaseB 目录）、规模统计、`feature_columns` 列表、PhaseB 指标摘要等。
- **用途**：溯源、审计与复现（这次训练到底用了哪些特征/数据规模是多少）。

### 11) `run_meta.json`（运行元信息）
- **内容**：`config`（训练超参）、`residual_stats`、`history_tail`、以及数据集元信息引用等。
- **用途**：复现实验与排错（这次 run 的参数与训练表现快照）。

### 12) `training_config.json`（训练配置快照）
- **内容**：训练使用的超参（`epochs/batch_size/hidden_dim/latent_dim/likelihood/lambda_*` 等）。
- **用途**：训练复现与对比不同 run 的设置差异。

### 13) `training_history.json`（训练曲线）
- **内容**：每个 epoch 的 `train` 与 `calibration` loss 序列。
- **用途**：诊断收敛/过拟合、早停位置等。

### 14) `phaseb_baseline_predictions.csv`（Phase B 基线预测拷贝）
- **内容**：从 PhaseB 输出中复制的基线预测结果（含 `strategy/model`、预测值与真值等列）。
- **用途**：在 PhaseC 输出目录内完成 PhaseB vs PhaseC 的对照分析与绘图，无需回溯 PhaseB 目录。

### 15) `analysis_report.md`（本报告）
- **内容**：本次运行的任务概述、数据规模、主结果、校准与物理可行性摘要，以及输出文件清单与解释。

### 16) `ieee_tsg_figures/figure_manifest.json`（绘图清单/摘要）
- **内容**：绘图脚本生成的图文件索引与关键摘要信息（字段以实际文件为准）。
- **用途**：快速定位生成的图、统一管理论文引用信息。

### 17) `ieee_tsg_figures/esmode_error_summary.csv`（ES 模式误差汇总）
- **内容**：按 ES 模式相关分组（若列存在且样本足够）汇总误差统计（字段以实际文件为准）。
- **用途**：分析“ES 模式激活/未激活”等条件下的误差差异，辅助机理解释。
