# Phase C: PG-RNP-CQR Residual Day-Ahead Experiment

This phase implements a physics-guided residual neural process for day-ahead
5G base-station energy forecasting.

The model learns:

```text
residual_true = energy_true - E_phys_hat
E_phys_hat = p_base + dynamic_phys_hat
energy_pred = E_phys_hat + residual_mu
```

`dynamic_phys_hat` is taken from the Phase B `two_stage_proxy + Physical`
day-ahead baseline.  The strongest Phase B empirical baselines are preserved in
`ablation_metrics.csv`.

## Run

From the repository root:

```bash
python phaseC_pg_rnp_cqr_20260416/run_phaseC_pg_rnp_cqr.py
```

Useful smoke-test command:

```bash
python phaseC_pg_rnp_cqr_20260416/run_phaseC_pg_rnp_cqr.py ^
  --epochs 1 ^
  --batch-size 4 ^
  --hidden-dim 32 ^
  --latent-dim 8 ^
  --max-rows 240 ^
  --output-dir phaseC_pg_rnp_cqr_20260416/outputs_smoke
```

PyTorch is required for training.  The data builder itself only depends on the
existing Phase B artifacts and standard scientific Python packages.

## Outputs

Default output directory:

```text
phaseC_pg_rnp_cqr_20260416/outputs/
```

Core files:

- `residual_trajectory_dataset.csv`
- `split_bs.json`
- `pg_rnp_predictions.csv`
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `coverage_by_horizon.csv`
- `ablation_metrics.csv`
- `physics_violation_report.csv`
- `analysis_report.md`

## Notes

The Phase B filtered learned comparison artifact currently contains many
observed horizon samples but only a small number of complete 24h trajectories.
The implementation therefore trains on masked 24h trajectories: every episode
has a 24-slot target layout, and unavailable horizons are masked out of the
loss and metrics.

