# 第 D 阶段：BESS 可调度性（Dispatchability）

本阶段评估：基站日前能耗预测的不确定性，如何影响备用电池储能系统（BESS）的备用容量（reserve）、可调度容量、可靠性风险，以及简化的调度价值（dispatch value）。

## 方法

实验读取已完成的第 C 阶段输出：

- `phaseC_pg_rnp_cqr_20260416/outputs/pg_rnp_predictions.csv`
- `phaseC_pg_rnp_cqr_20260416/outputs/phaseb_baseline_predictions.csv`

然后执行：

1. Aligns Phase B point forecasts, Phase C point forecasts, Phase C upper prediction bounds, and true hourly energy.
2. Builds consecutive outage windows of fixed duration `T_backup`.
3. Calibrates a one-sided Aggregate-CQR reserve margin on calibration windows:
   `q_(1-epsilon) = Quantile(R_true - R_pred_phaseC)`.
4. Evaluates five reserve policies on the held-out test split:
   Oracle, Phase B Point, Phase C Point, Phase C Aggregate-CQR, and Hourly Upper Sum.
5. Writes window-level data, policy metrics, figures, and a dispatchability report.

由于原始项目数据不包含已安装电池容量，默认的 BESS 情景为合成设定：

```text
C_b = 8 * p_base
SOC = 1
SOC_min = 0
T_backup = 4 hours
epsilon = 0.10
```

上述假设均为命令行参数，因此无需改动代码即可重复运行，以分析容量与可靠性的敏感性。

## 运行

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py
```

常用的敏感性运行示例：

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --backup-duration 6 --capacity-hours 10
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --epsilon 0.05
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --phaseb-strategy two_stage_proxy --phaseb-model Physical
```

## 输出

所有输出写入 `phaseD_bess_dispatchability_20260417/outputs/`：

- `aligned_hourly_predictions.csv`
- `bess_window_dataset.csv`
- `reserve_policy_metrics.csv`
- `reserve_error_by_horizon.csv`
- `dispatchability_analysis_report.md`
- `run_meta.json`
- `fig1_error_propagation.png`
- `fig2_mean_dispatchable_capacity.png`
- `fig3_shortfall_dispatch_tradeoff.png`
- `fig4_reserve_error_by_horizon.png`
- `fig5_cqr_hourly_upper_comparison.png`
