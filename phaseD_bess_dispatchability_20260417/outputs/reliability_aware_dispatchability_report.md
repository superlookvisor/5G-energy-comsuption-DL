# Phase D Reliability-Aware Dispatchability Report

## Research Design

This run models communication reliability through outage repair time and BESS autonomy time. The key event is `D_b <= tau_b,t`, meaning the outage is repaired before backup energy is exhausted. Phase C uncertainty builds an Aggregate-CQR upper bound for cumulative energy over the required autonomy duration.

## Inputs And Assumptions

- Phase C output directory: `G:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\phaseC_pg_rnp_cqr_20260416\outputs`
- BESS scenario: `C_b = 8.00 * p_base`, `SOC=1.00`, `SOC_min=0.00`
- Repair distribution: `exponential`
- Reliability target `R_min`: `0.900000`
- Required autonomy time: `3.2894` hours, discretized to `4` hours
- Energy uncertainty risk epsilon: `0.1000`
- Aggregate-CQR margin at required duration: `10.602972`
- Evaluation split: `test`

## Data Coverage

- Aligned hourly rows: `8951`
- Reliability trajectory rows: `8951`
- Evaluation decisions: `1379`
- Evaluation base stations: `121`
- Horizon infeasibility rate: `0.5736`

## Main Metrics

| policy | mean_dispatchable_capacity | mean_comm_reliability | reliability_violation_rate | mean_expected_interruption_duration | mean_expected_unserved_traffic | net_dispatch_value | capacity_infeasibility_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle Reliability | 83.8771 | 0.9392 | 0.0000 | 0.0869 | 0.0070 | 83.0013 | 0.0000 |
| Phase C Point Reliability | 85.1579 | 0.9042 | 0.5238 | 0.1369 | 0.0140 | 83.7754 | 0.0000 |
| Reliability-Aware CQR | 74.5549 | 0.9354 | 0.0867 | 0.0923 | 0.0085 | 73.6231 | 0.0000 |
| Hourly Upper Reliability | 70.6255 | 0.9412 | 0.0459 | 0.0840 | 0.0062 | 69.7792 | 0.0187 |
| Full BESS Backup | 0.0000 | 0.9730 | 0.0000 | 0.0386 | 0.0010 | -0.3865 | 0.0000 |

## Interpretation

The reliability-aware CQR policy reserves enough BESS energy to cover the Phase C cumulative energy upper bound over the repair-time quantile. Increasing the reliability target increases the required autonomy time, which increases retained communication-backup energy and reduces dispatchable capacity. Full BESS Backup gives the maximum communication reliability under installed capacity, while Oracle Reliability is an infeasible benchmark that uses true future energy.

## Outputs

- `aligned_hourly_predictions.csv`
- `reliability_trajectory_dataset.csv`
- `reliability_policy_decisions.csv`
- `reliability_policy_metrics.csv`
- `reliability_by_horizon.csv`
- `energy_cqr_quantiles_by_duration.csv`
- `reliability_aware_dispatchability_report.md`
- `run_meta.json`
- `fig1_repair_distribution_reliability_curve.png`
- `fig2_autonomy_time_distribution.png`
- `fig3_dispatchable_capacity_vs_reliability_target.png`
- `fig4_comm_reliability_dispatch_tradeoff.png`
- `fig5_expected_interruption_duration.png`
