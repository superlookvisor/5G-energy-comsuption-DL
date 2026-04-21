# Phase D Reliability-Aware Dispatchability Report

## Research Design

This run models communication reliability through outage repair time and BESS autonomy time. The key event is `D_b <= tau_b,t`, meaning the outage is repaired before backup energy is exhausted. Phase C uncertainty builds an Aggregate-CQR upper bound for cumulative energy over the required autonomy duration.

## Inputs And Assumptions

- Phase C output directory: `G:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\phaseC_pg_rnp_cqr_20260416\outputs`
- BESS scenario: `C_b = 8.00 * p_base`, `SOC=1.00`, `SOC_min=0.00`
- Repair distribution: `exponential`
- Reliability target `R_min`: `0.990000`
- Required autonomy time: `6.5788` hours, discretized to `7` hours
- Energy uncertainty risk epsilon: `0.1000`
- Aggregate-CQR margin at required duration: `17.356552`
- Evaluation split: `test`

## Data Coverage

- Aligned hourly rows: `8951`
- Reliability trajectory rows: `8951`
- Evaluation decisions: `1379`
- Evaluation base stations: `121`
- Horizon infeasibility rate: `0.8071`

## Main Metrics

| policy | mean_dispatchable_capacity | mean_comm_reliability | reliability_violation_rate | mean_expected_interruption_duration | mean_expected_unserved_traffic | net_dispatch_value | capacity_infeasibility_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle Reliability | 15.1418 | 0.9884 | 0.2632 | 0.0166 | 0.0020 | 14.9739 | 0.2632 |
| Phase C Point Reliability | 16.0218 | 0.9852 | 0.6692 | 0.0211 | 0.0024 | 15.8081 | 0.2293 |
| Reliability-Aware CQR | 6.0171 | 0.9885 | 0.2744 | 0.0165 | 0.0020 | 5.8506 | 0.6541 |
| Hourly Upper Reliability | 6.0039 | 0.9887 | 0.2744 | 0.0162 | 0.0019 | 5.8404 | 0.5677 |
| Full BESS Backup | 0.0000 | 0.9890 | 0.2632 | 0.0157 | 0.0019 | -0.1590 | 0.0000 |

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
