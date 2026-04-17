# Phase D BESS Dispatchability Analysis Report

## Research Design

Phase D maps the existing day-ahead load forecasts into BESS reserve and dispatchability metrics. It does not retrain Phase B or Phase C models. The pipeline is:

1. Align hourly Phase B point forecasts, Phase C point forecasts, Phase C CQR upper bounds, and true energy.
2. Build consecutive `4`-hour outage windows within each day-ahead trajectory.
3. Calibrate Aggregate-CQR on calibration windows with one-sided scores `R_true - R_pred_phaseC`.
4. Evaluate reserve policies on `test` windows.
5. Convert reserve into dispatchable BESS capacity, reserve shortfall, overreserve, and simplified dispatch value.

## Inputs And Assumptions

- Phase C output directory: `G:\5G-energy-comsuption-DL\Small-sample-MLM\energy_model_anp\phaseC_pg_rnp_cqr_20260416\outputs`
- Phase B point baseline: `two_stage_proxy + RandomForest`
- Backup duration: `4` hours
- CQR risk level epsilon: `0.100`; target reliability: `0.900`
- Aggregate-CQR reserve add-on `q_(1-epsilon)`: `10.602972`
- BESS scenario: `C_b = 8.00 * p_base`, `SOC=1.00`, `SOC_min=0.00`
- Economic scenario: dispatch price `1.000`, shortfall penalty `10.000`

## Data Coverage

- Aligned hourly rows: `8951`
- All BESS windows: `3848`
- Evaluation windows: `588`
- Evaluation base stations: `89`
- Evaluation outage-start horizons: `1` to `21`
- BESS capacity violation rate under this fixed-capacity scenario: `0.0000`

## Main Metrics

| policy | mean_dispatchable_capacity | dispatchable_capacity_ratio | shortfall_rate | mean_shortfall_energy | mean_overreserve_energy | reliability_satisfaction_rate | mean_reserve_error | net_dispatch_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle | 83.8771 | 0.4789 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 83.8771 |
| Phase B Point | 84.3497 | 0.4816 | 0.4014 | 3.5731 | 3.1004 | 0.5986 | -0.4727 | 48.6191 |
| Phase C Point | 85.1579 | 0.4862 | 0.5238 | 3.0682 | 1.7874 | 0.4762 | -1.2808 | 54.4759 |
| Phase C Aggregate-CQR | 74.5549 | 0.4257 | 0.0867 | 0.8506 | 10.1728 | 0.9133 | 9.3222 | 66.0486 |
| Hourly Upper Sum | 70.6255 | 0.4033 | 0.0459 | 0.2444 | 13.7231 | 0.9541 | 13.4787 | 68.1820 |

## Findings

- Phase C point forecasts change the reserve shortfall rate from `0.4014` to `0.5238` relative to the selected Phase B point baseline.
- Aggregate-CQR reaches reliability satisfaction `0.9133` against the target `0.9000` on the evaluation split.
- Compared with Hourly Upper Sum, Aggregate-CQR releases `3.9294` more mean dispatchable capacity per window.
- Aggregate-CQR mean overreserve is `10.1728`, while Hourly Upper Sum mean overreserve is `13.7231`; this quantifies the cost of simply summing hourly upper bounds.
- Under the configured economic scenario, Aggregate-CQR net value is `66.0486` and Hourly Upper Sum net value is `68.1820`. At dispatch price `1.000`, Aggregate-CQR has higher net value than Hourly Upper Sum when the shortfall penalty is below approximately `6.4812`.
- Reserve error is cumulative: Fig. 1 links outage-start hourly forecast error to the `4`-hour reserve error, and Fig. 4 shows how this error varies across outage start horizons.

## Outputs

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

## Interpretation

The results directly support the Phase D claim: day-ahead forecast error is not only an energy-prediction issue, because it propagates into the backup reserve requirement. Point forecasts can release more dispatchable capacity when they underestimate reserve demand, but that capacity is purchased with higher shortfall risk. Aggregate-CQR uses calibration at the cumulative backup-window level, so it targets the actual BESS reliability constraint more directly than summing hourly upper bounds.
