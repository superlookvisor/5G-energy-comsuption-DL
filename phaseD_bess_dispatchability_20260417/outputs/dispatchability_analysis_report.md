# 第 D 阶段：BESS 可调度性分析报告

## 研究设计

第 D 阶段将现有的日前负载预测映射为 BESS 的备用容量（reserve）与可调度性指标。本阶段不会重新训练第 B 或第 C 阶段模型。流程如下：

1. Align hourly Phase B point forecasts, Phase C point forecasts, Phase C CQR upper bounds, and true energy.
2. Build consecutive `4`-hour outage windows within each day-ahead trajectory.
3. Calibrate Aggregate-CQR on calibration windows with one-sided scores `R_true - R_pred_phaseC`.
4. Evaluate reserve policies on `test` windows.
5. Convert reserve into dispatchable BESS capacity, reserve shortfall, overreserve, and simplified dispatch value.

## 输入与假设

- Phase C output directory: `G:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\phaseC_pg_rnp_cqr_20260416\outputs`
- Phase B point baseline: `two_stage_proxy + RandomForest`
- Backup duration: `4` hours
- CQR 风险水平 epsilon: `0.100`；目标可靠性：`0.900`
- Aggregate-CQR 备用裕量附加项 `q_(1-epsilon)`: `10.602972`
- BESS scenario: `C_b = 8.00 * p_base`, `SOC=1.00`, `SOC_min=0.00`
- Economic scenario: dispatch price `1.000`, shortfall penalty `10.000`

## 数据覆盖范围

- 对齐后的小时级样本行数：`8951`
- 全部 BESS 窗口数：`3848`
- 评估窗口数：`588`
- 评估基站数：`89`
- 评估的停电起始提前期（horizon）：`1` 到 `21`
- 在该固定容量情景下的 BESS 容量违约率：`0.0000`

## 主要指标

| policy | mean_dispatchable_capacity | dispatchable_capacity_ratio | shortfall_rate | mean_shortfall_energy | mean_overreserve_energy | reliability_satisfaction_rate | mean_reserve_error | net_dispatch_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle | 83.8771 | 0.4789 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 83.8771 |
| Phase B Point | 84.3497 | 0.4816 | 0.4014 | 3.5731 | 3.1004 | 0.5986 | -0.4727 | 48.6191 |
| Phase C Point | 85.1579 | 0.4862 | 0.5238 | 3.0682 | 1.7874 | 0.4762 | -1.2808 | 54.4759 |
| Phase C Aggregate-CQR | 74.5549 | 0.4257 | 0.0867 | 0.8506 | 10.1728 | 0.9133 | 9.3222 | 66.0486 |
| Hourly Upper Sum | 70.6255 | 0.4033 | 0.0459 | 0.2444 | 13.7231 | 0.9541 | 13.4787 | 68.1820 |

## 主要发现

- 相对于选定的第 B 阶段点预测基线，采用第 C 阶段点预测会将备用容量短缺率从 `0.4014` 变化为 `0.5238`。
- 在评估划分上，Aggregate-CQR 的可靠性满足率达到 `0.9133`，高于目标值 `0.9000`。
- 与 Hourly Upper Sum 相比，Aggregate-CQR 每个窗口平均释放更多 `3.9294` 的可调度容量。
- Aggregate-CQR 的平均过度预留（overreserve）为 `10.1728`，而 Hourly Upper Sum 为 `13.7231`；这量化了“直接对每小时上界求和”的成本。
- 在所设定的经济情景下，Aggregate-CQR 的净价值为 `66.0486`，Hourly Upper Sum 的净价值为 `68.1820`。当调度价格为 `1.000` 时，若短缺惩罚低于约 `6.4812`，Aggregate-CQR 的净价值将高于 Hourly Upper Sum。
- 备用误差具有累积性：图 1 将停电起始时刻的小时预测误差与 `4` 小时备用误差联系起来，图 4 展示该误差如何随停电起始提前期变化。

## 输出

- `aligned_hourly_predictions.csv`
- `bess_window_dataset.csv`
- `reserve_policy_metrics.csv`
- `reserve_error_by_horizon.csv`
- `run_meta.json`
- `fig1_error_propagation.png`
- `fig2_mean_dispatchable_capacity.png`
- `fig3_shortfall_dispatch_tradeoff.png`
- `fig4_reserve_error_by_horizon.png`
- `fig5_cqr_hourly_upper_comparison.png`

## 结果解读

结果直接支持第 D 阶段的主张：日前预测误差不仅是能耗预测层面的问题，因为它会传播到备用容量需求。点预测在低估备用需求时，可能释放更多可调度容量，但这种容量提升是以更高的短缺风险为代价。Aggregate-CQR 在“累计的备份窗口”层级上进行校准，因此相比于对每小时上界求和，它能更直接地对准 BESS 的实际可靠性约束。
