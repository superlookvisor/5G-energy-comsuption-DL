# Phase D：基于日前负荷不确定性的基站 BESS 可调度容量评估模型

## 1. 研究目标

本文的核心目标是分析：

```text
基站日前负荷计划/预测的准确度，会如何影响其后备储能系统（BESS）的可调度容量、后备可靠性和调度经济性。
```

既有研究 `Evaluating the Dispatchable Capacity of Base Station Backup Batteries in Distribution Networks` 说明了为什么基站后备电池可以作为电力系统灵活性资源：基站配置 BESS 是为了保障通信供电可靠性，但在满足后备供电要求后，部分剩余容量可以参与电网调度。

本文在此基础上进一步强调：

```text
BESS 的可调度容量不仅与基站所在节点的供电可靠性、后备保障时间和电池总容量有关，
还与日前对基站未来负荷的预测准确度和不确定性刻画有关。
```

在日前调度中，未来基站负荷不可直接观测，BESS 需要保留多少后备能量只能依赖日前预测。如果预测低估未来负荷，会导致后备能量不足，损害通信可靠性；如果预测高估未来负荷，会导致过度保留电量，降低可调度容量和经济收益。因此，准确且校准良好的日前负荷预测是评估 BESS 可调度容量的重要前提。

## 2. 与当前工作的关系

当前项目已经完成了基站能耗预测的三个阶段：

| 阶段 | 作用 | 输出 |
|---|---|---|
| Phase A | 估计基站静态基础功耗 | `p_base` |
| Phase B | 构建仅日前可用信息下的能耗点预测 | `\hat{E}^{B}_{b,t+h}` |
| Phase C | 构建物理引导残差概率预测 | `\hat{E}^{C}_{b,t+h}` 和预测区间 |

Phase D 不再以提升预测精度为唯一目标，而是将 Phase B / Phase C 的预测结果输入 BESS 可调度容量评估模型，研究预测误差如何传导到后备 reserve、可调度容量和调度风险。

整体链路为：

```text
日前负荷预测
    -> 后备能量需求估计
    -> BESS 必须保留的安全 reserve
    -> 可调度容量
    -> 可靠性风险与经济性收益
```

## 3. 建模假设

本文暂不研究以下因素的变化：

```text
1. 基站最小保障时间不变；
2. BESS 总能量容量不变；
3. BESS 安装位置和节点可靠性参数不变；
4. 重点只分析基站负荷预测不确定性的影响。
```

主要假设如下：

| 符号 | 含义 |
|---|---|
| `b` | 基站索引 |
| `t_0` | 日前预测起点 |
| `h` | 预测时距，`h = 1, ..., 24` |
| `t` | 可能发生停电或需要后备保障的时刻 |
| `E_{b,t}` | 基站 `b` 在时刻 `t` 的真实能耗 |
| `\hat{E}_{b,t}` | 基站 `b` 在时刻 `t` 的日前预测能耗 |
| `C_b` | 基站 `b` 的 BESS 固定可用能量容量 |
| `T^{backup}` | 固定最小后备保障时间 |
| `SOC_{b,t}` | BESS 在时刻 `t` 的荷电状态 |
| `SOC^{min}` | BESS 最低允许 SOC |
| `\epsilon` | 后备不足风险容忍水平 |

如果暂不显式模拟 SOC 动态，可先令：

```text
SOC_{b,t} = 1
SOC^{min} = 0
```

此时 `C_b` 可直接理解为可用 BESS 能量容量。

## 4. 后备能量需求

如果基站 `b` 在时刻 `t` 发生供电中断，并且必须由 BESS 支撑固定时长 `T^{backup}`，则真实后备能量需求为：

```math
R^{true}_{b,t}
=
\sum_{k=0}^{T^{backup}-1}
E_{b,t+k}
```

日前计划中无法提前知道真实值，只能使用预测值：

```math
\hat{R}_{b,t}
=
\sum_{k=0}^{T^{backup}-1}
\hat{E}_{b,t+k}
```

其中 `\hat{E}_{b,t+k}` 可以来自不同预测策略，例如 Phase B 点预测、Phase C 点预测或 Phase C 概率预测。

## 5. 预测误差对 BESS reserve 的影响

后备能量预测误差定义为：

```math
\Delta R_{b,t}
=
\hat{R}_{b,t}
-
R^{true}_{b,t}
```

当：

```math
\Delta R_{b,t} < 0
```

表示预测低估负荷，BESS 预留后备能量不足，可能损害通信可靠性。

当：

```math
\Delta R_{b,t} > 0
```

表示预测高估负荷，BESS 过度保留后备容量，可调度容量被低估，经济性下降。

对应定义后备不足量：

```math
S_{b,t}
=
\max
\left(
0,
R^{true}_{b,t}
-
R^{safe}_{b,t}
\right)
```

以及过度预留量：

```math
O_{b,t}
=
\max
\left(
0,
R^{safe}_{b,t}
-
R^{true}_{b,t}
\right)
```

其中 `R^{safe}_{b,t}` 是调度策略实际要求 BESS 保留的安全后备能量。

## 6. 安全后备能量策略

### 6.1 Oracle 策略

Oracle 使用真实未来负荷，现实中不可获得，仅作为理论上界：

```math
R^{oracle}_{b,t}
=
R^{true}_{b,t}
```

对应可调度容量为理想基准。

### 6.2 Phase B 点预测策略

使用 Phase B 的日前点预测：

```math
R^{B}_{b,t}
=
\sum_{k=0}^{T^{backup}-1}
\hat{E}^{B}_{b,t+k}
```

该策略用于衡量传统点预测日前计划对 BESS 可调度容量的影响。

### 6.3 Phase C 点预测策略

使用 Phase C 的物理引导残差模型点预测：

```math
R^{C}_{b,t}
=
\sum_{k=0}^{T^{backup}-1}
\hat{E}^{C}_{b,t+k}
```

该策略用于评估更高预测精度是否能够减少后备不足和过度预留。

### 6.4 Phase C 累计 CQR 安全策略

Phase C 已经提供单小时预测区间，但 BESS 后备约束关心的是连续 `T^{backup}` 小时累计能耗。因此建议对累计后备需求直接进行 conformal calibration，而不是简单相加逐小时上界。

定义校准集上的累计误差：

```math
s_i
=
R^{true}_i
-
\hat{R}^{C}_i
```

取风险水平 `\epsilon` 下的分位数：

```math
q_{1-\epsilon}
=
Quantile_{1-\epsilon}
\left(
\{s_i\}_{i \in \mathcal{I}_{cal}}
\right)
```

则安全后备能量为：

```math
R^{CQR}_{b,t}
=
\hat{R}^{C}_{b,t}
+
q_{1-\epsilon}
```

使其满足近似概率约束：

```math
\mathbb{P}
\left(
R^{true}_{b,t}
\le
R^{CQR}_{b,t}
\right)
\ge
1-\epsilon
```

该策略是本文建议的主方法，因为它直接服务于 BESS 后备保障约束。

### 6.5 逐小时上界相加策略

作为保守基线，也可以使用 Phase C 的逐小时预测上界：

```math
R^{hourly\_upper}_{b,t}
=
\sum_{k=0}^{T^{backup}-1}
\hat{E}^{U}_{b,t+k}
```

但该方法不应作为主方法，因为逐小时 `90%` 上界相加通常不等价于累计能耗的 `90%` 上界，可能导致过度保守，从而低估可调度容量。

## 7. BESS 可调度容量定义

在不显式考虑 SOC 动态时，基站 `b` 在时刻 `t` 的 BESS 可调度容量定义为：

```math
C^{disp}_{b,t}
=
\max
\left(
0,
C_b
-
R^{safe}_{b,t}
\right)
```

如果考虑当前 SOC 和最低 SOC 限制，则可写为：

```math
C^{disp}_{b,t}
=
\max
\left(
0,
SOC_{b,t} C_b
-
SOC^{min} C_b
-
R^{safe}_{b,t}
\right)
```

其中：

```text
R^{safe}_{b,t} 越大，BESS 必须保留的后备能量越多，可调度容量越小；
R^{safe}_{b,t} 越小，可调度容量越大，但后备不足风险可能增加。
```

因此，预测模型的作用不是直接决定 BESS 容量，而是影响 `R^{safe}_{b,t}` 的估计，从而影响 `C^{disp}_{b,t}`。

## 8. 可靠性与经济性指标

### 8.1 后备可靠性指标

后备不足率：

```math
P^{short}_{policy}
=
\frac{1}{N}
\sum_{(b,t)}
\mathbf{1}
\left(
S_{b,t}^{policy} > 0
\right)
```

平均后备不足能量：

```math
\overline{S}_{policy}
=
\frac{1}{N}
\sum_{(b,t)}
S_{b,t}^{policy}
```

可靠性满足率：

```math
A_{policy}
=
1
-
P^{short}_{policy}
```

也可以进一步报告 `p95` 后备不足能量，用于衡量极端风险。

### 8.2 可调度容量指标

平均可调度容量：

```math
\overline{C}^{disp}_{policy}
=
\frac{1}{N}
\sum_{(b,t)}
C^{disp,policy}_{b,t}
```

可调度容量比例：

```math
\rho^{disp}_{policy}
=
\frac{
\sum_{(b,t)}
C^{disp,policy}_{b,t}
}{
\sum_{(b,t)}
C_b
}
```

相对 Oracle 的容量损失：

```math
L^{disp}_{policy}
=
\overline{C}^{disp}_{oracle}
-
\overline{C}^{disp}_{policy}
```

### 8.3 过度预留指标

平均过度预留能量：

```math
\overline{O}_{policy}
=
\frac{1}{N}
\sum_{(b,t)}
O_{b,t}^{policy}
```

该指标反映由于预测过高或安全策略过保守而损失的可调度空间。

### 8.4 简化经济性指标

若给定单位可调度容量收益 `\pi_t` 和后备不足惩罚系数 `\lambda`，可定义净收益：

```math
J_{policy}
=
\sum_{(b,t)}
\pi_t
C^{disp,policy}_{b,t}
-
\lambda
\sum_{(b,t)}
S^{policy}_{b,t}
```

如果暂时没有真实电价，可以使用情景参数：

```text
低收益场景、中收益场景、高收益场景
低惩罚场景、中惩罚场景、高惩罚场景
```

从而分析不同可靠性偏好下的调度经济性。

## 9. 实验策略对比

建议至少比较以下策略：

| 策略 | 后备能量 `R^{safe}` | 作用 |
|---|---|---|
| Oracle | `R^{true}` | 理想上界 |
| Phase B Point | `R^{B}` | 传统日前点预测基线 |
| Phase C Point | `R^{C}` | 高精度点预测策略 |
| Phase C Aggregate-CQR | `R^{C} + q_{1-\epsilon}` | 主推荐策略 |
| Hourly Upper Sum | `\sum \hat{E}^{U}` | 保守区间基线 |

核心比较问题包括：

```text
1. Phase C 相比 Phase B 是否降低了后备不足率？
2. Phase C 相比 Phase B 是否减少了过度预留？
3. Aggregate-CQR 是否能在满足可靠性约束的同时释放更多可调度容量？
4. Hourly Upper Sum 是否过于保守，导致可调度容量明显偏低？
5. 预测误差从单小时能耗传导到 T_backup 小时后备能量时，风险是否被放大？
```

## 10. Phase D 数据集设计

建议构造文件：

```text
phaseD_bess_dispatchability_*/outputs/bess_window_dataset.csv
```

每一行表示一个基站、一个日前预测起点、一个可能停电起点和一个固定后备窗口。

建议字段：

| 字段 | 含义 |
|---|---|
| `BS` | 基站编号 |
| `origin_time` | 日前预测起点 |
| `outage_start_time` | 假设停电开始时刻 |
| `outage_start_horizon` | 停电开始时刻相对 `origin_time` 的 horizon |
| `backup_duration` | 固定后备保障时长 |
| `R_true` | 真实累计后备需求 |
| `R_pred_phaseB` | Phase B 点预测累计需求 |
| `R_pred_phaseC` | Phase C 点预测累计需求 |
| `R_safe_cqr` | Phase C 累计 CQR 安全后备需求 |
| `R_hourly_upper` | 逐小时上界相加得到的保守需求 |
| `C_bess` | 固定 BESS 容量 |
| `C_disp_oracle` | Oracle 可调度容量 |
| `C_disp_phaseB` | Phase B 可调度容量 |
| `C_disp_phaseC` | Phase C 点预测可调度容量 |
| `C_disp_cqr` | Aggregate-CQR 可调度容量 |
| `shortfall_phaseB` | Phase B 后备不足量 |
| `shortfall_phaseC` | Phase C 后备不足量 |
| `shortfall_cqr` | CQR 后备不足量 |
| `overreserve_phaseB` | Phase B 过度预留量 |
| `overreserve_phaseC` | Phase C 过度预留量 |
| `overreserve_cqr` | CQR 过度预留量 |

## 11. Phase D 输出指标表

建议输出：

```text
phaseD_bess_dispatchability_*/outputs/reserve_policy_metrics.csv
```

字段包括：

| 字段 | 含义 |
|---|---|
| `policy` | 策略名称 |
| `mean_dispatchable_capacity` | 平均可调度容量 |
| `dispatchable_capacity_ratio` | 可调度容量占 BESS 总容量比例 |
| `shortfall_rate` | 后备不足率 |
| `mean_shortfall_energy` | 平均后备不足能量 |
| `p95_shortfall_energy` | 95 分位后备不足能量 |
| `mean_overreserve_energy` | 平均过度预留能量 |
| `reliability_satisfaction_rate` | 后备可靠性满足率 |
| `mean_reserve_energy` | 平均保留后备能量 |
| `mean_reserve_error` | 平均后备需求估计误差 |
| `net_dispatch_value` | 简化调度净收益，可选 |

## 12. 推荐图表

建议生成以下图表：

| 图 | 内容 | 目的 |
|---|---|---|
| Fig. 1 | 单小时预测误差与累计 reserve 误差关系 | 说明预测误差传导 |
| Fig. 2 | 不同策略的平均可调度容量 | 比较容量释放能力 |
| Fig. 3 | 后备不足率与可调度容量 trade-off | 展示可靠性-经济性权衡 |
| Fig. 4 | 不同 outage start horizon 下的 reserve 误差 | 分析时段敏感性 |
| Fig. 5 | Point、Aggregate-CQR、Hourly Upper Sum 对比 | 证明累计校准优于简单保守相加 |

## 13. 预期论文结论

本文最终希望回答：

```text
日前计划越准确，BESS 所需保留的后备 reserve 越接近真实需求；
预测低估会增加后备不足风险，预测高估会损失可调度容量；
相比单纯点预测，经过累计后备需求校准的概率预测可以在满足可靠性要求的同时释放更多可调度容量；
因此，基站负荷预测不确定性是评估后备储能可调度容量时不可忽略的因素。
```

## 14. 下一步实现建议

建议按以下顺序推进：

```text
1. 确保 Phase C 完整输出 pg_rnp_predictions.csv；
2. 对齐 Phase B 和 Phase C 的预测结果；
3. 构造 T_backup 小时累计后备窗口样本；
4. 在校准集上对累计后备需求误差进行 conformal calibration；
5. 计算不同 reserve 策略下的 C_disp、shortfall 和 overreserve；
6. 输出 reserve_policy_metrics.csv 和 dispatchability_analysis_report.md；
7. 生成可靠性-经济性 trade-off 图。
```

Phase D 的关键不是重新训练一个更复杂的预测模型，而是将已有日前概率预测转化为可解释的 BESS 调度指标，从而证明：

```text
负荷预测准确度和不确定性校准能力，会直接影响基站后备储能的可调度容量评估和调度策略质量。
```
