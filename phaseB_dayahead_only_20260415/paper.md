# A Physics-Guided Two-Stage Proxy Framework for Day-Ahead Base-Station Energy Forecasting

> Draft prepared in an IEEE Transactions on Smart Grid style. The final submission should be converted to the IEEEtran journal template, use two-column IEEE formatting, numbered figures/tables, and the journal page limit. The IEEE PES TSG scope covers smart-grid data analytics, communication technologies, cyber-physical systems, and demand-side applications; this work should therefore be positioned as data analytics for communication-energy infrastructure rather than as generic load forecasting.

## Abstract

Day-ahead energy forecasting for 5G base stations is challenging because the future traffic load and energy-saving (ES) states are unavailable at the dispatch origin. This paper develops a physics-guided two-stage proxy framework that first constructs day-ahead covariate proxies from historical same-hour, rolling-window, and current observations, and then predicts the dynamic energy component on top of a static base-power estimate. A constrained stacking module is further introduced to learn convex proxy weights without using future energy labels. Experiments are conducted on base-station panel data with group-wise cross validation by station. Under a sparse-observation-consistent filtering protocol, the proposed two-stage proxy with Random Forest achieves the best strict 24-h trajectory MAPE of 0.09096, improving over the fixed-weight counterpart by approximately 20.0%. Physical checks show monotonic load-energy behavior, negative effects for dominant ES modes, and consistent correlation between base power and station scale. The results indicate that learned day-ahead covariate proxies can materially improve base-station energy prediction while preserving interpretability.

## Index Terms

5G base stations, day-ahead energy forecasting, smart grid data analytics, physics-guided machine learning, proxy covariates, constrained stacking, energy-saving modes.

## Nomenclature


| Symbol                                            | Meaning                                                                               |
| ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| b                                                 | Base station index.                                                                   |
| t_0                                               | Day-ahead forecast origin, fixed at 00:00 in this study.                              |
| h                                                 | Forecast horizon, h \in 1,\ldots,24 hours.                                            |
| E_{b,t}                                           | Observed total station energy.                                                        |
| p^{base}_b                                        | Static base-power component imported from Phase A.                                    |
| y_{b,t}                                           | Dynamic energy component, y_{b,t}=E_{b,t}-p^{base}_b.                                 |
| \widehat{E}_{b,t_0+h}                             | Predicted total station energy.                                                       |
| \widehat{D}*{1}, \widehat{D}*{2}, \widehat{D}_{3} | Proxy physical load features for weighted load, nonlinear load, and load variability. |
| S_m                                               | Strength of ES mode m, m=1,\ldots,6.                                                  |
| \widehat{S}_m                                     | Day-ahead proxy of ES mode m.                                                         |
| \widehat{I}_m                                     | Interaction proxy, \widehat{I}_m=\widehat{S}*m \cdot \widehat{load}*{pmax}.           |
| w                                                 | Convex proxy weight vector satisfying w \ge 0 and \mathbf{1}^{T}w=1.                  |


## I. Introduction

The rapid deployment of 5G networks increases the operational coupling between communication infrastructure and power systems. Base-station energy is shaped by hardware scale, traffic load, and energy-saving control modes. Accurate day-ahead forecasts can support demand response, energy procurement, and network-side energy management, but the forecasting problem differs from standard short-term load forecasting: at the day-ahead origin, future traffic load and ES states are not directly known.

This paper studies a day-ahead-only Phase B experiment. Compared with an intraday rolling setting, the model is not allowed to use target-hour lagged energy, target-hour load, or future ES observations. The central question is how to construct physically meaningful covariates when the true target-time covariates are missing. We answer this by building historical proxy covariates and by learning their convex combination weights from target-time covariate reconstruction errors, without using future energy labels.

The contributions are:

1. A day-ahead-only panel construction protocol that predicts 24 future hourly targets from a fixed 00:00 origin while preserving strict timestamp continuity.
2. A two-stage proxy feature design that converts historical load and ES proxies into interpretable dynamic-energy features.
3. A constrained proxy-weight learning module that improves fixed heuristic weights while avoiding energy-label leakage.
4. A simulation and reporting protocol that combines strict 24-h trajectory metrics, horizon-wise errors, physical consistency checks, and fallback-chain diagnostics.

## II. Problem Formulation

For each base station b and origin time t_0, the task is to predict the total energy at t_0+h, where h=1,\ldots,24. The prediction is decomposed as


\widehat{E}_{b,t_0+h}=p^{base}*b+\widehat{y}*{b,t_0+h}.


The static term p^{base}*b is taken from Phase A (`D_relaxed + quantile_10`), and Phase B focuses on predicting the dynamic component \widehat{y}*{b,t_0+h}. The available information is restricted to observations no later than the forecast origin and historical same-hour summaries. This restriction prevents the model from using intraday lags or future covariates that would not be available in a practical day-ahead workflow.

The evaluation uses two complementary views:

- Strict trajectory metrics: only complete 24-h trajectories are retained; MAPE, peak error, and valley error are averaged over trajectories.
- Horizon-wise metrics: all valid samples at each horizon are used to compute MAE, RMSE, MAPE, and sample counts.

The strict trajectory view tests end-to-end daily forecasting behavior, whereas the horizon-wise view reveals how data availability and prediction difficulty change with h.

## III. Methodology

### A. Panel Construction

The raw energy records (`ECdata.csv`) provide station-level total energy. The cell-level communication records (`CLdata.csv`) and hardware metadata (`BSinfo.csv`) are aggregated to the `(BS, Time)` level. The merged panel contains load statistics, ES mode strengths, static hardware features, and Phase A base-power estimates. Rows missing `Energy`, `load_pmax_weighted`, `load_mean`, or `p_base` are removed.

For the day-ahead-only setting, the panel keeps:

- cyclic time encodings: hour and day-of-week sine/cosine terms;
- rolling 24-h load summaries: `load_mean_roll24`, `load_pmax_roll24`, `load_std_roll24`;
- historical same-hour ES priors: `S_*_hour_prior`;
- static station attributes: `n_cells`, `sum_pmax`, `sum_antennas`;
- dynamic energy label: `dynamic_energy = Energy - p_base`.

It intentionally excludes intraday lag features such as `energy_lag`*, `dynamic_lag*`, `load_*_lag1`, and `S_*_lag1`.

### B. Two Proxy Strategies

Two feature-construction strategies are compared.

The first strategy, `two_stage_proxy`, constructs proxy load and ES covariates:


\widehat{load}*{mean}, \quad \widehat{load}*{pmax}, \quad \widehat{load}*{std}, \quad \widehat{S}*{m}.


These proxies are then transformed into physical features:


\widehat{D}*{1}=sumpmax_b \cdot \widehat{load}*{pmax},



\widehat{D}*{2}=sumpmax_b \cdot \widehat{load}*{pmax}^{2},



\widehat{D}*{3}=\widehat{load}*{std},



\widehat{I}*{m}=\widehat{S}*{m}\cdot\widehat{load}_{pmax}.


The second strategy, `historical_proxy`, directly uses historical proxy terms such as previous-day same-hour load, rolling load statistics, and same-hour ES priors. It is a lighter baseline with fewer physics-guided transformations.

### C. Fixed and Learned Proxy Weights

The fixed-weight baseline uses heuristic convex combinations. For example, `load_mean_hat` and `load_pmax_hat` use weights `[0.55, 0.30, 0.15]` over `[prevday, roll24, current]`; `load_std_hat` uses `[0.60, 0.40]` over `[prevday, roll24]`; and ES proxies use `[0.70, 0.30]` over `[prevday, hour_prior]`.

The learned-weight version estimates global convex weights:


\min_{w\in \Delta}\frac{1}{N}\sum_{i=1}^{N}
\left(z_i-\sum_{k}w_k x_{i,k}\right)^2+\lambda\lVert w\rVert_2^2,
\quad
\Delta=w \mid w \ge 0,\mathbf{1}^{T}w=1.


Here z_i is the target-time covariate, such as true `load_mean` at t_0+h, not the future energy label. This design learns better proxies while avoiding leakage from the prediction target E_{b,t_0+h}. The implementation uses an 80%/20% station-level split for weight learning and applies the learned global weights to downstream model training.

For the filtered main experiment, the learned weights are:


| Proxy           | Sources                  | Weights                |
| --------------- | ------------------------ | ---------------------- |
| `load_mean_hat` | prevday, roll24, current | 0.8153, 0.0331, 0.1516 |
| `load_pmax_hat` | prevday, roll24, current | 0.8161, 0.0188, 0.1651 |
| `load_std_hat`  | prevday, roll24          | 0.8083, 0.1917         |
| `S_*_hat`       | prevday, hour_prior      | 0.9796, 0.0204         |


### D. Forecasting Models

Each proxy strategy is evaluated with four model classes:

1. `Physical`: a linear model with positive coefficients for physically constructed features.
2. `SemiPhysical_Ridge`: semi-physical features with RidgeCV.
3. `SemiPhysical_Lasso`: semi-physical features with LassoCV.
4. `RandomForest`: nonlinear regression with 300 trees and `min_samples_leaf=2`.

All models are evaluated using `GroupKFold` by `BS`, which prevents the same base station from appearing in both training and validation folds.

## IV. Experimental Setup

### A. Dataset and Filtering Protocol

The unfiltered panel contains 923 base stations and 92629 merged rows. To align with the sparse-observation quality-control protocol, the filtered experiments retain stations with at least 24 merged observations and exclude the sparse-BS list from `phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv`. This leaves 818 stations and 90621 rows.

The paper should treat the filtered learned-weight experiment as the main result because it combines the quality-control protocol with the best-performing proxy construction.

### B. Experiment Matrix

**Table I should report the experiment matrix.** Construct it from `filter_meta.json` in each output directory and `compare_meta.json` in the comparison directories.


| Experiment | Output directory                  | Filter | Proxy weights    | BS / rows                 | Role                                |
| ---------- | --------------------------------- | ------ | ---------------- | ------------------------- | ----------------------------------- |
| E0         | `outputs_proxy_compare_fixed/`    | No     | Fixed            | 923 / 92629               | Unfiltered baseline                 |
| E1         | `outputs_proxy_compare_learned/`  | No     | Learned          | 923 / 92629               | Unfiltered weight-learning ablation |
| E2         | `outputs_filter_compare_fixed/`   | Yes    | Fixed            | 818 / 90621               | Filtered baseline                   |
| E3         | `outputs_filter_compare_learned/` | Yes    | Learned          | 818 / 90621               | Main experiment                     |
| E4         | `outputs_proxy_impact_filtered/`  | Yes    | Fixed vs learned | aligned predictions=71608 | Main ablation analysis              |


In the final IEEE table, use `Table I` with concise columns. Keep output directory names in a footnote or reproducibility paragraph if the table becomes too wide.

### C. Evaluation Metrics

The strict 24-h trajectory metrics are computed from `dayahead_metrics.csv`:


MAPE=\frac{1}{N}\sum_i\frac{|E_i-\widehat{E}_i|}{\max(|E_i|,\epsilon)}.


Peak and valley errors are computed on each complete 24-h trajectory by comparing the prediction at the true daily peak and valley. Horizon-wise MAE/RMSE/MAPE are computed from `dayahead_horizon_metrics.csv`.

## V. Simulation Results and Analysis

### A. Strict 24-h Trajectory Performance

**Table II should be the main performance table.** Build it from:

- `outputs_filter_compare_learned/dayahead_metrics.csv` for the proposed method;
- `outputs_filter_compare_fixed/dayahead_metrics.csv` for the fixed-weight ablation;
- optionally `outputs_proxy_compare_learned/dayahead_metrics.csv` for unfiltered sensitivity.

Recommended rows for the main text:


| Setting              | Strategy         | Model        | MAPE    | Peak error | Valley error | Complete trajectories |
| -------------------- | ---------------- | ------------ | ------- | ---------- | ------------ | --------------------- |
| Filtered + learned   | two_stage_proxy  | RandomForest | 0.09096 | 4.3459     | 1.2176       | 9                     |
| Filtered + learned   | historical_proxy | RandomForest | 0.09984 | 4.9499     | 1.6297       | 9                     |
| Filtered + fixed     | two_stage_proxy  | RandomForest | 0.11367 | 5.1110     | 2.5322       | 9                     |
| Unfiltered + learned | two_stage_proxy  | RandomForest | 0.09674 | 6.2581     | 1.0910       | 9                     |
| Unfiltered + fixed   | two_stage_proxy  | RandomForest | 0.11032 | 5.9051     | 2.5041       | 9                     |


The main conclusion is that the filtered learned-weight two-stage proxy gives the lowest strict 24-h MAPE. Compared with the filtered fixed-weight counterpart, MAPE decreases by 0.02271, or approximately 20.0%. Compared with the filtered learned historical proxy using the same Random Forest model, MAPE decreases by 0.00888.

Because only 9 complete trajectories are available under the strict 24-h criterion, the text must state that this table is a stringent trajectory-level comparison and should be interpreted together with the horizon-wise results in Table V and Fig. 6.

### B. Proxy-Weight Ablation

**Table III should show fixed vs learned weight deltas.** Build it from `outputs_proxy_impact_filtered/compare_strict_metrics.csv`.

Recommended compact table:


| Strategy         | Model              | Fixed MAPE | Learned MAPE | Delta MAPE | Delta peak | Delta valley |
| ---------------- | ------------------ | ---------- | ------------ | ---------- | ---------- | ------------ |
| two_stage_proxy  | RandomForest       | 0.11367    | 0.09096      | -0.02271   | -0.7651    | -1.3147      |
| historical_proxy | RandomForest       | 0.12376    | 0.09984      | -0.02392   | -0.3704    | -0.9455      |
| two_stage_proxy  | SemiPhysical_Ridge | 0.17856    | 0.13701      | -0.04155   | -0.2393    | -1.1198      |
| two_stage_proxy  | Physical           | 0.19421    | 0.13625      | -0.05796   | -1.5045    | -1.3114      |


The table should be ordered by the final model relevance or by `Delta MAPE`. For the paper narrative, emphasize the Random Forest row because it is the best final predictor, while noting that learned weights also improve the linear/semi-physical models.

**Fig. 4 should visualize this ablation.** Use `outputs_proxy_impact_filtered/fig_strict_best_metrics_compare.png` as the current generated figure. If redrawing for TSG, make a grouped bar chart with three metric groups: MAPE, peak error, and valley error. Use a shared legend (`fixed`, `learned`), keep the y-axis units explicit, and state in the caption that the selected model is the fixed-baseline best model.

### C. Horizon-Wise Error Behavior

**Fig. 6 should present horizon-wise error.** Use:

- `outputs_filter_compare_learned/fig_error_by_horizon.png` for the main error curve;
- `outputs_proxy_impact_filtered/fig_delta_mae_by_horizon.png` for learned-minus-fixed MAE deltas.

For publication, a two-panel figure is recommended:

- Fig. 6(a): MAE versus horizon for the best model under each strategy.
- Fig. 6(b): \Delta MAE = MAE_{learned}-MAE_{fixed}, where negative values indicate improvement.

**Table V should report selected horizons and sample counts.** Build it from `outputs_filter_compare_learned/dayahead_horizon_metrics.csv` by selecting `h=1,6,12,18,24` and the lowest-MAE row at each horizon:


| Horizon | Best strategy    | Best model         | MAE    | RMSE   | MAPE   | Samples | BS  |
| ------- | ---------------- | ------------------ | ------ | ------ | ------ | ------- | --- |
| 1       | two_stage_proxy  | RandomForest       | 2.6155 | 3.8941 | 0.1462 | 2235    | 802 |
| 6       | two_stage_proxy  | RandomForest       | 1.5579 | 2.5762 | 0.0656 | 519     | 402 |
| 12      | historical_proxy | RandomForest       | 3.4591 | 4.9280 | 0.1080 | 117     | 108 |
| 18      | two_stage_proxy  | RandomForest       | 3.4702 | 5.1504 | 0.1132 | 28      | 28  |
| 24      | two_stage_proxy  | SemiPhysical_Ridge | 0.8815 | 1.1530 | 0.0372 | 9       | 9   |


The analysis should not claim that long-horizon prediction is intrinsically easier based only on the low `h=24` MAE, because `h=24` has only 9 samples. The safer interpretation is that sample availability decreases with the horizon, and the figure/table should be read jointly.

### D. Prediction Scatter and Representative Trajectory

**Fig. 5 should show predicted versus actual energy.** Use `outputs_filter_compare_learned/fig_prediction_vs_actual.png`. The figure compares the best model under each proxy strategy. The diagonal line should be retained; for TSG formatting, use equal x/y limits and a legend or panel titles identifying the selected model and MAPE.

**Fig. 7 should show a representative 24-h trajectory.** Use `outputs_filter_compare_learned/fig_dayahead_trajectory.png`. The underlying data are saved as `trajectory_details_B_501@2023-01-02.csv` and `trajectory_summary_B_501@2023-01-02.csv` in the same directory. For the final paper, label the x-axis as forecast horizon and use two curves only: observed energy and predicted energy. The caption should identify that the trajectory is selected from the best strict-trajectory model.

### E. Physical Consistency Analysis

**Fig. 2 should show load-energy relation.** Use `outputs_filter_compare_learned/fig_load_vs_energy.png`. The plot is constructed by sampling up to 12000 panel rows, plotting `load_pmax_weighted` versus `Energy`, and fitting a second-order trend line. It supports the use of the nonlinear proxy feature \widehat{D}_{2}. The filtered main result has Spearman correlation 0.6883.

**Fig. 3 should show ES mode impact.** Use `outputs_filter_compare_learned/fig_es_mode_impact.png`. It plots the mean dynamic-energy difference between active and inactive ES states for each mode. The strongest negative effects are:


| Mode        | Dynamic energy difference |
| ----------- | ------------------------- |
| `S_ESMode5` | -14.1767                  |
| `S_ESMode2` | -13.4676                  |
| `S_ESMode1` | -13.4674                  |


**Table VI should summarize physical checks.** Build it from `outputs_filter_compare_learned/physics_checks.csv` and `es_mode_effects.csv`:


| Check                          | Value  | Interpretation                                           |
| ------------------------------ | ------ | -------------------------------------------------------- |
| Load monotonicity ratio        | 1.0000 | Mean energy is nondecreasing across load bins.           |
| Load-energy Spearman           | 0.6883 | Load and energy have a clear positive rank correlation.  |
| Best ES effect is negative     | 1.0000 | At least one ES mode reduces dynamic energy as expected. |
| p^{base}-`n_cells` correlation | 0.8080 | Base-power estimates align with station scale.           |


### F. Fallback-Chain Diagnostic

The current generated audit in `outputs_fallback_audit/analysis_report.md` states that previous-day same-hour coverage for `load_mean/load_pmax` ranges from approximately 57% to 73%, `roll24` covers roughly 24% to 40%, and `current` contributes about 2.65%. This diagnostic should appear either in an appendix or as a short robustness paragraph.

**Fig. 8 can be constructed from `outputs_fallback_audit/fallback_summary_by_horizon.csv`.** Recommended construction:

1. For `load_mean`, select columns `load_mean_ratio_prevday`, `load_mean_ratio_roll24`, and `load_mean_ratio_current`.
2. Plot them as a stacked bar chart or stacked area chart over `horizon=1,...,24`.
3. Repeat for `load_pmax` only if space permits; otherwise state that `load_pmax` follows the same pattern.
4. Add a horizontal note in the caption that all proportions sum to one up to rounding.

The figure explains why a robust fallback chain is necessary and why proxy-weight learning does not eliminate the need for missing-source diagnostics.

## VI. Discussion

The experiments show that both proxy strategy and proxy weighting matter. The best results are achieved when sparse stations are filtered, the proxy weights are learned, and the two-stage proxy is paired with a nonlinear regressor. The learned weights heavily favor previous-day same-hour covariates, which is consistent with day-ahead periodicity. However, the fallback audit shows that previous-day information is unavailable for a substantial fraction of samples, so rolling and current terms remain important as fallback sources.

The strict 24-h trajectory evaluation is deliberately conservative but has a small sample size. A TSG submission should therefore avoid overclaiming from the strict trajectory table alone. The horizon-wise table and error curves are essential for showing that the method has enough evidence beyond the 9 complete trajectories.

## VII. Conclusion

This paper presents a day-ahead-only base-station energy forecasting framework that combines static base-power decomposition, physics-guided dynamic-energy features, constrained proxy-weight learning, and group-wise validation by station. On the filtered main experiment, the learned two-stage proxy with Random Forest achieves the lowest strict 24-h trajectory MAPE of 0.09096 and improves over the fixed-weight counterpart by approximately 20.0%. Physical consistency checks further support the model design by confirming load-energy monotonicity, negative ES effects for dominant modes, and a strong relationship between static base power and station scale.

Future work should increase the number of complete 24-h trajectories, test cross-date and cross-region generalization, and compare the proxy framework with sequence models under the same no-future-covariate constraint.

## Figure and Table Construction Checklist

This checklist is for preparing the final IEEE TSG manuscript figures and tables.


| Item                         | Source file                                                | Construction rule                                                        | Where to place         |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------- |
| Table I: experiment matrix   | `filter_meta.json`, `compare_meta.json`                    | Summarize filter, BS count, row count, weight mode, purpose              | Experimental Setup     |
| Table II: strict performance | `dayahead_metrics.csv`                                     | Sort by MAPE; keep proposed and key baselines                            | Simulation Results     |
| Table III: ablation          | `compare_strict_metrics.csv`                               | Report fixed, learned, and delta metrics                                 | Simulation Results     |
| Table IV: proxy weights      | `proxy_weights.json`, `proxy_weights_meta.json`            | Show weights and optional covariate RMSE/MAE                             | Method or Results      |
| Table V: horizon metrics     | `dayahead_horizon_metrics.csv`                             | Select representative horizons or place all 24 in appendix               | Simulation Results     |
| Table VI: physical checks    | `physics_checks.csv`, `es_mode_effects.csv`                | Interpret each check in one phrase                                       | Simulation Results     |
| Fig. 1: workflow             | Manual redraw                                              | Use vector diagram: raw data -> panel -> proxies -> models -> evaluation | Methodology            |
| Fig. 2: load vs energy       | `fig_load_vs_energy.png`                                   | Scatter + quadratic trend; label sampled rows                            | Physical consistency   |
| Fig. 3: ES impact            | `fig_es_mode_impact.png`                                   | Bar chart; sort modes by effect                                          | Physical consistency   |
| Fig. 4: strict ablation      | `fig_strict_best_metrics_compare.png`                      | Grouped bars for MAPE, peak, valley                                      | Main results           |
| Fig. 5: prediction scatter   | `fig_prediction_vs_actual.png`                             | Equal axes, diagonal reference line                                      | Main results           |
| Fig. 6: horizon errors       | `fig_error_by_horizon.png`, `fig_delta_mae_by_horizon.png` | Two panels: MAE and delta MAE                                            | Horizon analysis       |
| Fig. 7: trajectory           | `fig_dayahead_trajectory.png`                              | Observed vs predicted over 24 horizons                                   | Case study             |
| Fig. 8: fallback ratios      | `fallback_summary_by_horizon.csv`                          | Stacked bars/area by horizon                                             | Robustness or appendix |


For IEEE-style presentation, redraw line and bar charts as vector graphics where possible, use consistent fonts and line widths, avoid crowded legends, and include sample counts in captions or table notes.

## References

[1] IEEE Power & Energy Society, "IEEE Transactions on Smart Grid," journal scope and paper categories. [https://ieee-pes.org/publications/transactions-on-smart-grid/](https://ieee-pes.org/publications/transactions-on-smart-grid/)

[2] IEEE Power & Energy Society, "Preparation and Submission of Transactions Papers," IEEE PES Author's Kit. [https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/](https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/)

[3] IEEE Author Center, "Create the Text of Your Article." [https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/)