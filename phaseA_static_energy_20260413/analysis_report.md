# 阶段A：基站静态基础能耗估计与静态特征建模分析

## 1. 代码与运行说明

本阶段面向 5G 基站能耗数据构建静态基础能耗估计流程，并进一步验证仅使用基站静态配置特征预测基础能耗的可行性。主程序为 `run_phaseA_static_energy.py`，输入数据包括 `../data/ECdata.csv`、`../data/CLdata.csv` 和 `../data/BSinfo.csv`。主要输出文件位于 `outputs/` 文件夹，包括窗口筛选统计、静态基础能耗估计、方法稳定性分析、建模数据集、模型性能、预测结果、随机森林特征重要性以及相关可视化图件。

## 2. 关键输出文件说明

- `window_summary.csv`：不同静态窗口定义下的样本数量、样本占比、基站覆盖数量和覆盖率。
- `target_combo_summary.csv`：不同窗口与估计方法组合下的目标变量覆盖情况及方法排序。
- `method_stability_summary.csv`：不同估计方法在多窗口条件下的跨窗口稳定性统计。
- `model_dataset.csv`：用于静态特征监督学习的 BS 级建模数据集。
- `model_metrics.csv`：Linear Regression、RidgeCV、LassoCV 和 Random Forest 的测试集性能。
- `rf_feature_importance.csv`：Random Forest 的静态特征重要性。
- `model_predictions.csv`：不同模型在测试集上的预测结果。
- `fig_window_coverage.png`：不同静态窗口的样本覆盖能力对比。
- `fig_window_stability.png`：不同窗口定义下的基础能耗估计稳定性。
- `fig_method_distribution.png`：不同估计方法得到的 `P_base` 分布对比。
- `fig_scatter_pbase_vs_features.png`：`P_base` 与关键静态特征之间的散点关系。

## 3. 结果分析

### 3.1 静态窗口敏感性分析

本阶段首先在 BS-时间粒度上筛选低负载或近似静态运行样本。由 `window_summary.csv` 可知，不同静态窗口定义对可用样本量和基站覆盖率具有显著影响。

**TABLE I  
STATIC-WINDOW COVERAGE COMPARISON**

| Window | Samples | Sample Ratio | BS Covered | BS Cover Ratio |
|---|---:|---:|---:|---:|
| A_strict_weighted | 15923 | 18.08% | 528 | 57.27% |
| B_mean_all_es_zero | 27668 | 31.42% | 725 | 78.63% |
| C_bs_low5pct | 3808 | 4.32% | 685 | 74.30% |
| D_relaxed | 36452 | 41.40% | 809 | 87.74% |

The results indicate a clear tradeoff between static purity and sample availability. The strictest window, `A_strict_weighted`, retains only 57.27% of the base stations, which may lead to biased downstream modeling because a large fraction of BSs cannot be assigned a static-energy target. In contrast, `D_relaxed` provides the largest number of valid samples and covers 809 BSs, corresponding to 87.74% of all BSs. Although a relaxed window may include a small amount of residual load-dependent variation, its substantially higher coverage makes it more appropriate for constructing a representative BS-level training set.

### 3.2 静态基础能耗估计方法对比

在每个静态窗口内，本文比较了 `min`、`quantile_05`、`quantile_10`、`median`、`trimmed_mean` 和 `mean` 六类聚合估计器。跨窗口稳定性统计见 `method_stability_summary.csv`。其中，`mean_std_across_windows` 和 `mean_cv_across_windows` 越小，说明该估计方法对静态窗口定义越不敏感，估计结果越稳定。

**TABLE II  
STABILITY COMPARISON OF STATIC-ENERGY ESTIMATION METHODS**

| Method | BSs with >=2 Windows | Mean Std. | Median Std. | Mean CV |
|---|---:|---:|---:|---:|
| quantile_05 | 774 | **0.2318** | 0.0615 | **0.01135** |
| quantile_10 | 774 | 0.2343 | 0.0814 | 0.01136 |
| min | 774 | 0.2781 | **0.0000** | 0.01403 |
| median | 774 | 0.3919 | 0.2815 | 0.01759 |
| trimmed_mean | 774 | 0.4356 | 0.3301 | 0.01939 |
| mean | 774 | 0.4766 | 0.3807 | 0.02115 |

The lower-tail quantile estimators provide the most stable estimates across temporal windows. The 5th-percentile estimator achieves the lowest mean standard deviation and mean coefficient of variation, while the 10th-percentile estimator is nearly equivalent, with only a 1.07% increase in mean standard deviation relative to `quantile_05`. Compared with the arithmetic mean, `quantile_05` reduces the mean cross-window standard deviation by 51.37%, and `quantile_10` reduces it by 50.84%.

The `min` estimator has a median standard deviation of zero, indicating that it is unchanged across windows for at least half of the BSs. However, its mean standard deviation and mean coefficient of variation are higher than those of the quantile-based estimators. This implies that the minimum operator is overly sensitive to extreme low observations for a subset of BSs. Conversely, `mean` and `trimmed_mean` are more affected by residual dynamic traffic loads and therefore tend to produce less stable static-energy estimates.

Based on the stability ranking,

`quantile_05` > `quantile_10` > `min` > `median` > `trimmed_mean` > `mean`,

low-quantile estimation is the preferred strategy for constructing a static-energy supervision target. In this implementation, the final target uses `D_relaxed + quantile_10`. This selection is justified by two criteria: first, `D_relaxed` maximizes BS-level coverage and provides 809 supervised samples; second, `quantile_10` retains the robustness of lower-tail estimation while being slightly less extreme than `quantile_05`, thereby reducing the risk of learning from occasional abnormal under-consumption values.

### 3.3 物理合理性验证

The physical consistency of the selected target is evaluated by correlating the estimated `P_base` with static BS configuration features in `model_dataset.csv`. The selected target covers 809 BSs. The estimated `P_base` has a mean of 22.50, a standard deviation of 9.43, a median of 18.45, and a range from 10.01 to 53.90.

Key Pearson correlations are as follows:

- corr(`P_base`, `sum_antennas`) = 0.8293
- corr(`P_base`, `sum_pmax`) = 0.8077
- corr(`P_base`, `n_cells`) = 0.8007
- corr(`P_base`, `sum_bandwidth`) = 0.4806
- corr(`P_base`, `mean_frequency`) = 0.0193
- corr(`P_base`, `high_freq_ratio`) = -0.3517

The strong positive correlations with `sum_antennas`, `sum_pmax`, and `n_cells` are consistent with the physical expectation that base energy consumption increases with radio scale, transmission capacity, and hardware complexity. This confirms that the selected target is not merely a statistical artifact of the windowing procedure, but preserves interpretable engineering relationships. The weak correlation with `mean_frequency` suggests that carrier frequency alone is not the dominant determinant of static energy after BS scale features are considered. The negative correlation with `high_freq_ratio` should be interpreted jointly with deployment type and RU composition, rather than as a standalone causal effect.

### 3.4 统计建模结果

To evaluate whether static configuration features can explain inter-BS variation in `P_base`, four supervised regression models were trained on the BS-level dataset: Linear Regression, RidgeCV, LassoCV, and Random Forest. The training target is the selected `D_relaxed + quantile_10` estimate. The dataset contains 809 BS-level samples, and the train-test split uses an 80/20 holdout protocol with `random_state = 42`.

**TABLE III  
TEST PERFORMANCE OF STATIC-FEATURE-BASED REGRESSION MODELS**

| Model | Test R2 | Test MAE |
|---|---:|---:|
| RandomForest | **0.8412** | **2.9271** |
| RidgeCV | 0.8266 | 3.1546 |
| LinearRegression | 0.8266 | 3.1536 |
| LassoCV | 0.8266 | 3.1555 |

The static-feature-based models achieve strong predictive performance. Even the linear baselines explain approximately 82.66% of the test-set variance, indicating that a large portion of static base-energy heterogeneity can be captured by configuration-level descriptors. Random Forest further improves the test R2 to 0.8412 and reduces the MAE to 2.9271, corresponding to a 7.18% MAE reduction compared with Linear Regression. This improvement suggests that nonlinear interactions among bandwidth, antenna scale, RU composition, and transmit-power capacity contribute additional explanatory power beyond a purely additive linear relationship.

The relatively small performance gap between Linear Regression, RidgeCV, and LassoCV also provides useful evidence. Since all three linear models reach nearly identical R2 values, the static-energy target is largely governed by stable and low-dimensional physical factors. Regularization does not materially improve the linear baseline, implying that the feature set is not severely over-parameterized under the current sample size. The superior but moderate gain of Random Forest indicates that nonlinear modeling is beneficial, but the learned relationship remains physically structured rather than noise dominated.

### 3.5 静态特征训练方法选择依据

The selection of a static-feature training pipeline is supported by three groups of simulation evidence.

First, the label construction is sufficiently representative. `D_relaxed` provides the highest BS coverage among all static windows, covering 809 BSs and 87.74% of the full BS population. This avoids the coverage loss observed under stricter windows and reduces the risk that the supervised model is trained only on a biased subset of lightly loaded or unusually stable BSs.

Second, the target estimator is robust to window perturbations. Lower-tail quantile estimators have the best cross-window stability. Although `quantile_05` is marginally the most stable, `quantile_10` achieves almost the same stability while being less sensitive to occasional abnormal low-energy measurements. Therefore, `D_relaxed + quantile_10` is a reasonable compromise between sample representativeness, temporal stability, and robustness to extreme-value noise.

Third, the static features have both predictive and physical validity. The selected target is strongly correlated with `sum_antennas`, `sum_pmax`, and `n_cells`, and the supervised models achieve a test R2 above 0.84 using only static configuration features. These results jointly demonstrate that the extracted `P_base` can serve as a reliable supervised signal for learning BS-level static energy characteristics. Consequently, the use of static features for the first-stage base-energy model is justified before introducing dynamic traffic-dependent terms in later modeling stages.

### 3.6 Feature Importance and Interpretability

The Random Forest feature importance results further clarify the dominant drivers of static base-energy variation.

**TABLE IV  
TOP RANDOM-FOREST FEATURE IMPORTANCE VALUES**

| Feature | Importance |
|---|---:|
| sum_bandwidth | 0.4501 |
| ru_ratio_Type1 | 0.2946 |
| sum_antennas | 0.1317 |
| sum_pmax | 0.0610 |
| ru_ratio_Type7 | 0.0403 |
| ru_ratio_Type3 | 0.0089 |
| mean_frequency | 0.0040 |
| ru_ratio_Type6 | 0.0032 |

The most influential feature is `sum_bandwidth`, followed by `ru_ratio_Type1`, `sum_antennas`, and `sum_pmax`. This ranking indicates that static base energy is jointly determined by spectrum allocation scale, RU hardware composition, antenna count, and rated transmission capability. The dominance of `sum_bandwidth` does not contradict the correlation analysis; rather, it suggests that bandwidth captures additional configuration information that becomes important in nonlinear partitions of the feature space. Meanwhile, the presence of RU-type ratios among the top features indicates that hardware architecture is an important source of static-energy heterogeneity even for BSs with similar cell counts or transmit-power capacity.

## 4. 关键发现

1. Static-window design directly affects the representativeness of the training target. `D_relaxed` provides the best coverage, with 36452 valid static-window samples and 809 covered BSs.
2. Lower-tail quantile estimators provide more stable static-energy estimates than mean-based or median-based estimators. The 5th- and 10th-percentile estimators reduce the sensitivity to residual dynamic load while avoiding the instability of the minimum operator.
3. The selected `D_relaxed + quantile_10` target balances coverage, stability, and robustness, making it suitable as the supervised label for static-feature training.
4. Static configuration features explain most of the inter-BS variation in `P_base`. Random Forest achieves a test R2 of 0.8412 and a test MAE of 2.9271.
5. The leading explanatory variables are `sum_bandwidth`, RU-type composition, `sum_antennas`, and `sum_pmax`, which are consistent with engineering expectations regarding hardware scale and radio resource configuration.

## 5. 结论

The Phase-A results demonstrate that static base-energy modeling is feasible and statistically well supported under the proposed label-construction pipeline. A relaxed static window combined with a lower-tail quantile estimator provides a reliable supervision target with high BS coverage and strong cross-window stability. The resulting target preserves physically meaningful correlations with radio configuration features and can be accurately predicted from static BS descriptors.

From a modeling perspective, the results support a two-stage energy modeling strategy. The first stage should learn the BS-specific static energy floor from configuration features, using `D_relaxed + quantile_10` as the target construction rule. The second stage can then focus on dynamic load-dependent energy variation after removing or conditioning on the learned static component. This separation is important for improving model interpretability and for avoiding the confounding of hardware-driven base energy with traffic-driven incremental energy consumption.

## 6. Additional Discussion

For BSs without valid static-window samples, `bs_without_static_samples.csv` provides the corresponding missing-sample records under each window definition. A practical extension is to impute their static base energy using configuration-similar BSs. Specifically, nearest-neighbor matching can be conducted in the feature space formed by `sum_pmax`, `n_cells`, `sum_antennas`, `sum_bandwidth`, RU-type ratios, and mode ratios. The imputed value can then be assigned using the lower-tail quantile or median of matched BSs, together with an uncertainty interval derived from the matched set. This would allow the static-energy model to cover BSs that are persistently active and therefore lack sufficiently low-load observations.
