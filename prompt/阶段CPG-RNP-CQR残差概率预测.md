# Phase C：顶刊版 PG-RNP-CQR 残差概率预测实验计划

## Summary
本阶段将 Phase C 重新设计为一个更适合顶级期刊叙事的实验框架：**Physics-Guided Residual Neural Process with Conformal Calibration, PG-RNP-CQR**。模型不直接预测总能耗，而是预测物理日前基线无法解释的 24 小时残差轨迹，并通过 horizon-wise conformal calibration 输出物理可行的不确定性区间。

核心公式：

```text
E_true(b,t+h) = E_phys_hat(b,t,h) + r_true(b,t,h)

r_true(1:24) ~ PG-RNP(context, target_proxy, physics_features)

E_pred = E_phys_hat + mu_r
Interval(E) = Project_phys(E_phys_hat + conformal_interval(r))
```

## Key Changes
- 新建实验目录：`phaseC_pg_rnp_cqr_20260416/`。
- 主模型命名为 `PG-RNP-CQR`，即物理引导残差 Neural Process + 多步轨迹概率预测 + conformal 校准。
- 预测任务固定为 Phase B 的 day-ahead 设定：每天 `00:00` 预测未来 `h=1..24` 小时能耗。
- 学习目标从 `Energy` 改为残差 `r = Energy - E_phys_hat`。
- 物理基线固定为 `E_phys_hat = p_base + dynamic_phys_hat`，其中 `dynamic_phys_hat` 来自 Phase B 的 `two_stage_proxy + Physical`。
- PI-ANP 升级为残差轨迹模型：一次输出 `h=1..24` 的残差均值和不确定性，而不是把 24 个 horizon 当作完全独立样本。
- 概率头默认使用 `Student-t likelihood`，用于处理残差重尾和异常站点；Gaussian NLL 只作为消融对照。
- 区间校准默认使用 `horizon-wise conformal calibration`，每个 horizon 单独学习校准分位数。
- 最终区间投影到物理可行域，避免负能耗或明显超过站点能力上界的预测区间。

## Model And Training Design
- 数据样本以 `(BS, origin_date)` 为一个 24h trajectory episode，target 为该 BS 从 `origin_time+1h` 到 `origin_time+24h` 的残差序列。
- Context set 由同一 BS 的历史已观测样本组成，包含历史残差、历史物理基线、历史负载代理、ES 强度和时间编码。
- Target set 由日前可获得特征组成，包含 `horizon`、目标小时编码、`p_base`、`dynamic_phys_hat`、`E_phys_hat`、Phase B 生成的 load/ES proxy 预测量。
- 静态站点特征包括 `n_cells, sum_pmax, sum_bandwidth, sum_antennas, mean_frequency, high_freq_ratio, mode_ratio_* , ru_ratio_*`。
- 编码器采用 Neural Process 结构：context encoder 汇总站点历史行为，target encoder 表示未来 horizon 条件，latent variable 表示站点级不确定性。
- 时序模块采用轻量 Temporal Transformer 或 GRU，默认优先 Temporal Transformer，用于建模 24h 残差轨迹相关性。
- 输出头为 Student-t 参数：`mu_r(h), scale_r(h), nu_r(h)`；若实现复杂，`nu_r` 可设为全局可学习参数。
- 主训练损失为：

```text
L = L_student_t_residual
  + λ_kl L_latent_kl
  + λ_bound L_physical_bound
  + λ_smooth L_trajectory_smooth
```

- `L_physical_bound` 约束最终 `E_phys_hat + r_hat` 落在物理合理范围内。
- `L_trajectory_smooth` 约束 24h 残差轨迹不过度尖跳，但不强迫能耗曲线过平滑。
- 训练集只用于模型拟合，calibration 集只用于 early stopping 和 conformal 分位数，test 集只用于最终报告。
- 数据切分必须按 BS 分组，默认 `train=70% / calibration=15% / test=15%`，随机种子 `42`。

## Baselines And Ablations
- 主对照保留 Phase B 多模型体系：`Physical`、`SemiPhysical_Ridge`、`SemiPhysical_Lasso`、`RandomForest`。
- 主强基线为 `two_stage_proxy + RandomForest`，因为当前 Phase B 中它是最强经验模型。
- 主物理基线为 `two_stage_proxy + Physical`，因为它定义了残差学习对象。
- 消融实验固定为：

```text
A0: Phase B two_stage_proxy + Physical
A1: Phase B two_stage_proxy + RandomForest
A2: Residual-ANP + Gaussian NLL
A3: Residual-ANP + Student-t NLL
A4: PG-RNP + Student-t NLL
A5: PG-RNP + Student-t + physical bound loss
A6: PG-RNP + Student-t + physical bound + horizon-wise CQR
```

- 可选附录对照：`Direct Energy ANP`，用于证明残差学习优于直接黑箱预测总能耗。
- 不把 diffusion model、large Transformer foundation model 或复杂 Bayesian neural network 作为主线，避免复杂度过高且与小样本物理残差任务不匹配。

## Interfaces And Outputs
- 新增主脚本：`phaseC_pg_rnp_cqr_20260416/run_phaseC_pg_rnp_cqr.py`。
- 新增数据构造脚本：`phaseC_pg_rnp_cqr_20260416/build_residual_trajectory_dataset.py`。
- 新增模型脚本：`phaseC_pg_rnp_cqr_20260416/pg_rnp_model.py`。
- 新增校准评估脚本：`phaseC_pg_rnp_cqr_20260416/evaluate_pg_rnp_cqr.py`。
- 输出目录固定为 `phaseC_pg_rnp_cqr_20260416/outputs/`。
- 核心输出文件包括 `residual_trajectory_dataset.csv`、`split_bs.json`、`pg_rnp_predictions.csv`、`metrics_overall.csv`、`metrics_by_horizon.csv`、`coverage_by_horizon.csv`、`ablation_metrics.csv`、`physics_violation_report.csv`、`analysis_report.md`。
- `pg_rnp_predictions.csv` 必须包含：`BS, origin_time, target_time, horizon, energy_true, E_phys_hat, residual_true, residual_mu, residual_sigma, energy_pred, lower_90, upper_90, split, model_name`。
- `coverage_by_horizon.csv` 必须包含：`horizon, coverage_90, avg_width_90, mae, rmse, mape, n_samples, n_bs`。

## Test Plan
- 数据泄漏检查：确认 target 时刻真实 `Energy`、真实 `dynamic_energy`、真实 target load 不进入输入特征。
- 轨迹一致性检查：确认每条 episode 的 `target_time = origin_time + horizon hours`，且 horizon 完整覆盖 `1..24`。
- 分组切分检查：确认 train/calibration/test 的 BS 集合完全不重叠。
- 物理基线检查：确认 `E_phys_hat = p_base + dynamic_phys_hat`，且 `residual_true = energy_true - E_phys_hat`。
- 复现实验检查：在相同 test BS 上复算 Phase B 对照组，保证对照公平。
- 点预测指标：报告 MAE、RMSE、MAPE、strict 24h trajectory MAPE、peak_error、valley_error。
- 不确定性指标：报告 90% coverage、平均区间宽度、coverage-width score、按 horizon 的 coverage。
- 物理一致性指标：报告负能耗区间比例、超过物理上界比例、投影前后区间变化幅度。
- 消融验收标准：`A6` 应相对 `A0` 明显降低点预测误差，并在 test 集达到接近 90% 的 horizon-wise coverage。
- 稳健性检查：按 `horizon`、单/多 cell、低/中/高负载、`pbase_source` 分组报告误差和覆盖率。

## Assumptions
- 这里将你未说完的“重新设计一个计划把”理解为：把上一版 Phase C 方案升级为我推荐的顶刊版 PG-RNP-CQR 实验计划。
- 默认研究目标是论文主实验，而不是只做工程性能验证。
- 默认主任务仍是 day-ahead 24h 预测，不新增 intraday 或 online forecasting 任务。
- 默认 90% 为主置信水平；可在附录补充 80%、95%。
- 默认先实现轻量 Temporal Transformer 版本；如果训练不稳定，则回退到 GRU 版本，但实验命名仍保持 PG-RNP-CQR。
