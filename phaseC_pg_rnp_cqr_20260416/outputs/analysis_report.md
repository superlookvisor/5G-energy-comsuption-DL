# Phase C PG-RNP-CQR Analysis Report

## Task
This experiment predicts day-ahead 5G base-station energy by learning the residual of the Phase B physical day-ahead baseline.

## Data
- Rows: 8951
- Base stations: 802
- Trajectories: 2235
- Complete 24h trajectories: 9
- Physical baseline: `two_stage_proxy + Physical`

## Main Result
- Best test model by MAPE: `A6_PG-RNP_StudentT_PhysicsBound_HorizonCQR` with MAPE=0.0732.
- PG-RNP-CQR test MAPE=0.0732, MAE=1.4293, mean horizon coverage=0.838, mean width=7.952.

## Calibration And Feasibility
- Horizon-wise CQR uses an independent residual score quantile per horizon.
- Mean q_hat across horizons: 2.6146
- Final intervals are projected to `[physical_lower, physical_upper]`; details are in `physics_violation_report.csv`.

## Outputs
- `residual_trajectory_dataset.csv`
- `pg_rnp_predictions.csv`
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `coverage_by_horizon.csv`
- `ablation_metrics.csv`
- `physics_violation_report.csv`
