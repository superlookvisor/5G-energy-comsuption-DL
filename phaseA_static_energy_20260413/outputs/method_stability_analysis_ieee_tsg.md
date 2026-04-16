# Stability Analysis of Static-Energy Estimation Methods

Table I reports the cross-window stability of different static-energy estimation strategies over 774 base stations (BSs) with at least two valid observation windows. A smaller value of `mean_std_across_windows`, `median_std_across_windows`, and `mean_cv_across_windows` indicates higher temporal stability of the estimated static-energy component.

**TABLE I  
STABILITY COMPARISON OF STATIC-ENERGY ESTIMATION METHODS**

| Method | No. of BSs | Mean Std. | Median Std. | Mean CV |
|---|---:|---:|---:|---:|
| 5th percentile | 774 | **0.2318** | 0.0615 | **0.01135** |
| 10th percentile | 774 | 0.2343 | 0.0814 | 0.01136 |
| Minimum | 774 | 0.2781 | **0.0000** | 0.01403 |
| Median | 774 | 0.3919 | 0.2815 | 0.01759 |
| Trimmed mean | 774 | 0.4356 | 0.3301 | 0.01939 |
| Mean | 774 | 0.4766 | 0.3807 | 0.02115 |

The results show that the lower-tail quantile estimators provide the most stable estimates of the static-energy component across temporal windows. In particular, the 5th-percentile estimator achieves the lowest mean standard deviation, i.e., 0.2318, and the lowest mean coefficient of variation, i.e., 0.01135. Compared with the conventional mean-based estimator, the 5th-percentile method reduces the mean cross-window standard deviation by 51.37% and the mean coefficient of variation by 46.35%. This indicates that the arithmetic mean is more sensitive to temporal fluctuations and load-dependent variations, and is therefore less suitable for extracting a quasi-static energy component.

The 10th-percentile estimator exhibits a very similar stability profile to the 5th-percentile estimator, with only a 1.07% increase in mean standard deviation. This suggests that both low-quantile estimators are robust choices for suppressing transient high-load effects. Nevertheless, the 5th-percentile estimator is marginally more stable in both mean standard deviation and mean coefficient of variation, making it the preferred option when the objective is to isolate the base-load or static-energy floor.

The minimum-based estimator yields a median standard deviation of zero, implying that for at least half of the BSs the minimum estimate remains unchanged across windows. However, its mean standard deviation is 0.2781 and its mean coefficient of variation is 0.01403, both worse than those of the 5th- and 10th-percentile estimators. This discrepancy suggests that the minimum operator may be overly sensitive to extreme observations or window-level noise for a subset of BSs. Therefore, although the minimum estimator can appear stable in the median sense, it is less reliable from a population-level robustness perspective.

Overall, the stability ranking based on mean standard deviation is:

`5th percentile` > `10th percentile` > `minimum` > `median` > `trimmed mean` > `mean`.

These findings support the use of the 5th-percentile estimator as the default static-energy extraction method in the small-sample BS energy modeling pipeline. It provides the best tradeoff between temporal stability and robustness to outliers, while avoiding the extreme-value sensitivity associated with the minimum operator.
