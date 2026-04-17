# Phase D BESS Dispatchability

This phase evaluates how day-ahead base-station energy forecast uncertainty affects backup BESS reserve, dispatchable capacity, reliability risk, and simplified dispatch value.

## Method

The experiment reads the completed Phase C outputs:

- `phaseC_pg_rnp_cqr_20260416/outputs/pg_rnp_predictions.csv`
- `phaseC_pg_rnp_cqr_20260416/outputs/phaseb_baseline_predictions.csv`

It then:

1. Aligns Phase B point forecasts, Phase C point forecasts, Phase C upper prediction bounds, and true hourly energy.
2. Builds consecutive outage windows of fixed duration `T_backup`.
3. Calibrates a one-sided Aggregate-CQR reserve margin on calibration windows:
   `q_(1-epsilon) = Quantile(R_true - R_pred_phaseC)`.
4. Evaluates five reserve policies on the held-out test split:
   Oracle, Phase B Point, Phase C Point, Phase C Aggregate-CQR, and Hourly Upper Sum.
5. Writes window-level data, policy metrics, figures, and a dispatchability report.

The default BESS scenario is synthetic because the raw project data do not contain installed battery capacities:

```text
C_b = 8 * p_base
SOC = 1
SOC_min = 0
T_backup = 4 hours
epsilon = 0.10
```

These assumptions are command-line parameters, so capacity and reliability sensitivity can be rerun without changing the code.

## Run

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py
```

Useful sensitivity runs:

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --backup-duration 6 --capacity-hours 10
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --epsilon 0.05
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --phaseb-strategy two_stage_proxy --phaseb-model Physical
```

## Outputs

All outputs are written to `phaseD_bess_dispatchability_20260417/outputs/`:

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
