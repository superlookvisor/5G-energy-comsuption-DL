# Phase D 代码说明：通信可靠性约束 BESS 可调度容量模型

## 1. 文件位置

主程序文件：

```text
phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py
```

该脚本已经从原来的“固定后备窗口 reserve 评估”重构为“随机修复时间 + BESS 可支撑时间 + 通信可靠性约束”的 Phase D 可调度容量分析程序。

## 2. 建模目标

新代码的目标是将 Phase C 的日前概率能耗预测转化为通信可靠性约束下的 BESS 可调度容量。

旧逻辑是：

```text
固定 T_backup 小时累计能耗
-> 估计安全 reserve
-> C_disp = usable_bess - reserve
```

新逻辑是：

```text
随机修复时间 D_b
-> 可靠性目标 R_min 对应的最小支撑时间 T_rel
-> Phase C 预测 T_rel 小时累计能耗上界
-> C_disp = usable_bess - reliable_energy_requirement
-> 评估通信不中断概率、期望中断时长和未服务业务量
```

核心可靠性事件为：

$$
D_b \le \tau_{b,t},
$$

其中 $D_b$ 是随机修复时间，$\tau_{b,t}$ 是 BESS 在保留后备能量后的可支撑时间。

## 3. 主要输入文件

脚本读取 Phase C 输出目录中的两个文件：

```text
phaseC_pg_rnp_cqr_20260416/outputs/pg_rnp_predictions.csv
phaseC_pg_rnp_cqr_20260416/outputs/phaseb_baseline_predictions.csv
```

`pg_rnp_predictions.csv` 提供：

| 字段 | 用途 |
|---|---|
| `energy_true` | 真实小时级能耗，用于评估和校准。 |
| `energy_pred` | Phase C 点预测，重命名为 `energy_pred_phaseC`。 |
| `upper_90` | Phase C 小时级 CQR 上界，用于 Hourly Upper baseline。 |
| `p_base` | 用于合成 BESS 容量 `C_b = capacity_hours * p_base`。 |
| `split` | 继承 Phase C 的 train/calibration/test 划分。 |
| `load_mean_hat` | 默认作为业务量 proxy；若不存在则使用 `energy_true`。 |

`phaseb_baseline_predictions.csv` 只用于保留 Phase B 对齐信息和兼容性，不再是新主模型的核心。

## 4. 新增命令行参数

新增的可靠性建模参数包括：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--repair-distribution` | `exponential` | 修复时间分布，支持 `exponential` 和 `weibull`。 |
| `--repair-rate` | `0.7` | 指数修复率 $\mu$，单位为 1/hour。 |
| `--weibull-shape` | `1.5` | Weibull 形状参数。 |
| `--weibull-scale` | `4.0` | Weibull 尺度参数。 |
| `--reliability-target` | `0.99` | 最低通信可靠性 $R_b^{\min}$。 |
| `--energy-risk-epsilon` | `0.10` | Phase C 累计能耗不确定性校准风险水平。 |
| `--outage-rate` | `0.01` | 停电发生率，用于计算 SAIDI 类指标。 |
| `--interruption-penalty` | `10.0` | 期望通信中断时长惩罚系数。 |
| `--unserved-traffic-penalty` | `1.0` | 期望未服务业务量惩罚系数。 |
| `--traffic-column` | `None` | 可选业务量字段；若不指定，优先使用 `load_mean_hat`。 |
| `--max-support-hours` | `24` | 最大可评估 BESS 支撑小时数。 |

原有参数 `--capacity-hours`、`--soc`、`--soc-min`、`--dispatch-price` 和 `--evaluation-split` 仍然保留。

## 5. 关键函数说明

### 5.1 `load_hourly_predictions()`

作用：读取 Phase C 和 Phase B 输出，并构造小时级对齐数据。

主要处理：

1. 检查 `pg_rnp_predictions.csv` 和 `phaseb_baseline_predictions.csv` 是否存在。
2. 检查关键列是否齐全。
3. 按 `BS, trajectory_id, origin_time, target_time, horizon` 对齐 Phase B 和 Phase C。
4. 将 Phase C 的 `energy_pred` 重命名为 `energy_pred_phaseC`。
5. 构造 `traffic_proxy`。

业务量 proxy 的选择顺序为：

```text
用户指定 traffic_column
-> load_mean_hat
-> energy_true
```

### 5.2 `build_reliability_trajectory_dataset()`

作用：从小时级预测构造“停电起点-未来累计能耗”数据集。

每一行表示：

```text
一个基站 b
一个日前预测起点 origin_time
一个假设停电起始时距 outage_start_horizon
```

对每一行，代码为 $T=1,2,...,T_{max}$ 构造累计量：

```text
R_true_T01, R_true_T02, ..., R_true_T24
R_pred_phaseC_T01, ..., R_pred_phaseC_T24
R_hourly_upper_T01, ..., R_hourly_upper_T24
traffic_T01, ..., traffic_T24
```

其中：

$$
R^{\mathrm{true}}_{b,t}(T)=\sum_{j=0}^{T-1}E_{b,t+j},
$$

$$
\widehat{R}^{C}_{b,t}(T)=\sum_{j=0}^{T-1}\widehat{E}^{C}_{b,t+j}.
$$

如果某个停电起点之后剩余 horizon 不足，则对应更长持续时间的字段为 `NaN`。

### 5.3 `calibrate_energy_quantiles_by_duration()`

作用：对每个持续时间 $T$ 单独做 Aggregate-CQR 校准。

校准分数为：

$$
s_i^{(T)}=R_i^{\mathrm{true}}(T)-\widehat{R}_i^C(T).
$$

给定 `--energy-risk-epsilon`，计算：

$$
q^R_{1-\epsilon_E,T}.
$$

输出是一个字典：

```python
{
    1: q_T1,
    2: q_T2,
    ...,
    24: q_T24,
}
```

这些分位数会写入：

```text
energy_cqr_quantiles_by_duration.csv
```

### 5.4 `repair_cdf()` 和 `repair_quantile()`

`repair_cdf()` 计算修复时间分布函数：

指数分布：

$$
F(d)=1-\exp(-\mu d).
$$

Weibull 分布：

$$
F(d)=1-\exp\left[-\left(\frac{d}{\lambda}\right)^k\right].
$$

`repair_quantile()` 计算可靠性目标对应的最小支撑时间：

$$
T^{\mathrm{rel}}=F^{-1}(R^{\min}).
$$

指数分布下：

$$
T^{\mathrm{rel}}=-\frac{\ln(1-R^{\min})}{\mu}.
$$

代码将连续时间分位数向上取整：

```python
T_rel = ceil(T_rel_continuous)
```

### 5.5 `apply_reliability_policies()`

这是新 Phase D 的核心函数。

它计算五种策略：

| 策略 | 含义 |
|---|---|
| `Oracle Reliability` | 用真实未来累计能耗作为理论最优需求。 |
| `Phase C Point Reliability` | 用 Phase C 点预测累计能耗。 |
| `Reliability-Aware CQR` | 用 Phase C 点预测 + Aggregate-CQR 累计裕量。 |
| `Hourly Upper Reliability` | 直接相加 Phase C 小时级上界。 |
| `Full BESS Backup` | 不释放任何容量，全部 BESS 留给通信后备。 |

对主策略 `Reliability-Aware CQR`，可靠能量需求为：

$$
\widehat{R}^{U}_{b,t}(T^{\mathrm{rel}})
=
\widehat{R}^{C}_{b,t}(T^{\mathrm{rel}})+
q^R_{1-\epsilon_E,T^{\mathrm{rel}}}.
$$

可调度容量为：

$$
C^{\mathrm{disp}}_{b,t}=
\left[\overline{C}_{b,t}-\widehat{R}^{U}_{b,t}(T^{\mathrm{rel}})\right]_+.
$$

### 5.6 `autonomy_time_from_row()`

作用：根据真实累计能耗和保留的后备能量计算 BESS 实际可支撑时间。

离散定义为：

$$
\tau_{b,t}
=
\max\left\{m:\sum_{j=0}^{m-1}E_{b,t+j}\le \overline{C}^{\mathrm{bk}}_{b,t}\right\}.
$$

代码会从 $T=1$ 开始逐步检查累计真实能耗是否超过保留后备能量。

### 5.7 `expected_excess_repair_time()`

作用：计算期望通信中断时长：

$$
\mathrm{ECID}=\mathbb{E}[(D-\tau)_+].
$$

指数分布下使用闭式解：

$$
\mathrm{ECID}=\frac{\exp(-\mu\tau)}{\mu}.
$$

Weibull 分布下使用数值积分近似。

### 5.8 `expected_unserved_traffic()`

作用：计算期望未服务业务量的离散近似。

代码使用：

$$
\mathrm{EUT}
\approx
\sum_{j>\tau} L_{b,t+j}\Pr(D_b>j).
$$

其中 $L_{b,t+j}$ 来自 `traffic_proxy`。

### 5.9 `reliability_metric_rows()`

作用：汇总策略级指标。

输出字段包括：

| 字段 | 含义 |
|---|---|
| `mean_dispatchable_capacity` | 平均可调度容量。 |
| `dispatchable_capacity_ratio` | 可调度容量占 BESS 总容量比例。 |
| `mean_comm_reliability` | 平均通信不中断概率。 |
| `reliability_violation_rate` | 通信可靠性低于目标值的比例。 |
| `mean_expected_interruption_duration` | 平均期望通信中断时长。 |
| `mean_saidi_bs` | 基站级 SAIDI 类指标。 |
| `mean_expected_unserved_traffic` | 平均期望未服务业务量。 |
| `mean_reliable_energy_requirement` | 平均可靠后备能量需求。 |
| `mean_autonomy_time_true` | 平均真实可支撑时间。 |
| `net_dispatch_value` | 简化调度净收益。 |
| `capacity_infeasibility_rate` | 可靠能量需求超过 BESS 可用能量的比例。 |
| `horizon_infeasibility_rate` | 所需支撑时间超过可用预测 horizon 的比例。 |

## 6. 输出文件

默认输出目录：

```text
phaseD_bess_dispatchability_20260417/outputs/
```

核心 CSV 输出：

| 文件 | 含义 |
|---|---|
| `aligned_hourly_predictions.csv` | Phase B / Phase C 对齐后的小时级预测。 |
| `reliability_trajectory_dataset.csv` | 停电起点级、多持续时间累计能耗数据集。 |
| `reliability_policy_decisions.csv` | 每个停电起点、每种策略下的可调度容量和可靠性指标。 |
| `reliability_policy_metrics.csv` | 策略级汇总指标。 |
| `reliability_by_horizon.csv` | 按停电起始 horizon 汇总的可靠性和容量指标。 |
| `energy_cqr_quantiles_by_duration.csv` | 各持续时间 $T$ 的 Aggregate-CQR 校准分位数。 |

报告输出：

```text
reliability_aware_dispatchability_report.md
run_meta.json
```

图表输出：

| 图 | 含义 |
|---|---|
| `fig1_repair_distribution_reliability_curve.png` | 修复时间 CDF 与通信可靠性的非线性关系。 |
| `fig2_autonomy_time_distribution.png` | 不同策略下 BESS 可支撑时间分布。 |
| `fig3_dispatchable_capacity_vs_reliability_target.png` | 可靠性目标提高时可调度容量下降趋势。 |
| `fig4_comm_reliability_dispatch_tradeoff.png` | 平均通信可靠性与可调度容量权衡。 |
| `fig5_expected_interruption_duration.png` | 不同策略下期望通信中断时长。 |

## 7. 运行示例

默认运行：

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py
```

设置更高通信可靠性：

```powershell
python run_phaseD_bess_dispatchability.py --reliability-target 0.9
```

使用 Weibull 修复时间：

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --repair-distribution weibull --weibull-shape 1.5 --weibull-scale 4
```

如果设置极高可靠性，例如：

```powershell
python phaseD_bess_dispatchability_20260417/run_phaseD_bess_dispatchability.py --reliability-target 0.99999
```

则可能出现：

```text
T_rel > max_support_hours
```

此时脚本不会伪造结果，而会在输出中将相关样本标记为 `horizon_infeasible=True`，并在指标中体现 `horizon_infeasibility_rate`。

## 8. 注意事项

1. 当前实现使用 Phase C 的 24 小时日前预测，因此 `max_support_hours` 最大为 24。
2. 若可靠性目标过高或修复率过低，所需支撑时间可能超过 24 小时，当前日前预测无法支撑该可靠性评估。
3. `Full BESS Backup` 仍可能达不到目标可靠性，这表示 BESS 装机容量不足，而不是调度策略失败。
4. `Reliability-Aware CQR` 的可靠性由两部分共同决定：修复时间分位数和 Phase C 累计能耗校准裕量。
5. 若没有真实通信业务量字段，代码默认使用 `load_mean_hat` 作为业务量 proxy；若该列也不存在，则使用 `energy_true`。
