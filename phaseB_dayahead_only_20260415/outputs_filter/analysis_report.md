# 阶段B（仅日前）交付说明

## 1. 任务范围
- **仅日前预测**：每日 `00:00` 起报，预测未来 24 小时动态能耗分解项；策略 `two_stage_proxy` 与 `historical_proxy` 均保留。
- **不做日内滚动**；panel 不计算日内滞后特征（`energy_lag*`、`dynamic_lag*`、`load_*_lag1`、`S_*_lag1`）及 `load_sq` / `load_x_*`。
- `P_base` 仍来自阶段A：`D_relaxed + quantile_10`。

## 2. BS 过滤（与稀疏 QC 对齐）
```json
{
  "n_bs_before": 923,
  "n_rows_before": 92629,
  "min_merged_obs_per_bs": 24,
  "exclude_bs_csv": "phase_sparse_observed_20260415\\outputs\\bs_below_hour_threshold.csv",
  "after_min_obs_n_bs": 818,
  "after_min_obs_n_rows": 90621,
  "excluded_bs_count_from_file": 105,
  "n_rows_removed_by_exclude_csv": 0,
  "n_bs_after": 818,
  "n_rows_after": 90621,
  "output_subdir": "outputs_filter",
  "proxy_weight_mode": "learned",
  "proxy_weights_json": null
}
```

## 3. 关键输出文件
- `panel_dataset.csv`：瘦 panel。
- `dayahead_metrics.csv`、`dayahead_horizon_metrics.csv`、`dayahead_predictions.csv`、`model_coefficients.csv`（仅日前）。
- `physics_checks.csv`、`es_mode_effects.csv`、`pbase_complete.csv`。
- 图：`fig_load_vs_energy.png`、`fig_es_mode_impact.png`、`fig_dayahead_trajectory.png`、`fig_prediction_vs_actual.png`、`fig_error_by_horizon.png`。

## 4. 结果摘要
- 最佳日前（按完整 24h 轨迹 MAPE）：`two_stage_proxy` + `RandomForest`，MAPE=0.1171。
- 负载单调分箱比例 100.00%，Spearman=0.6883；`p_base` 与 `n_cells` 相关 0.8079。
- ES 模式均值差（Top 3）：  
- S_ESMode5: -14.1775
- S_ESMode2: -13.4677
- S_ESMode1: -13.4677
- 各步 MAE 最优单步：h=6，`two_stage_proxy` + `RandomForest`。

## 5. 输出目录
本报告与 CSV/图位于同一目录：`outputs_filter/`。
