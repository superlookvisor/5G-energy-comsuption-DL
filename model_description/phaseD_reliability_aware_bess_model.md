# Phase D 改进模型：考虑随机修复过程的通信可靠性约束 BESS 可调度容量评估

## 1. 建模动机

原始 Phase D 模型将通信后备可靠性近似表示为固定后备窗口内的能量充裕性约束：

$$
\Pr\{R^{\mathrm{true}}_w \le R^{\mathrm{safe}}_w\}\ge 1-\epsilon.
$$

该约束评估的是 BESS 后备能量是否足以覆盖一个给定长度的后备窗口，但并不等价于完整的通信可靠性。其隐含假设是：只要 BESS 能够支撑固定时长 $T^{\mathrm{bk}}$，则通信可靠；否则通信不可靠。

然而，实际基站停电后存在修复过程。通信服务是否中断不仅取决于 BESS 容量，还取决于：

1. 停电持续时间或故障修复时间；
2. BESS 在当前负载轨迹下能够支撑的时间；
3. 停电是否在 BESS 耗尽之前被修复；
4. 中断发生时的业务负载强度。

因此，通信可靠性不应被简单建模为 BESS 后备容量的线性函数，而应通过随机修复时间和 BESS autonomy time 共同刻画。

## 2. 随机停电修复时间

设基站 $b$ 在时刻 $t$ 发生停电。令随机变量

$$
D_b
$$

表示从停电发生到供电恢复或故障修复所需的持续时间。其分布函数记为

$$
F_b(d)=\Pr(D_b\le d).
$$

常见的修复时间模型包括指数分布和 Weibull 分布。

若采用指数修复时间模型：

$$
D_b\sim \mathrm{Exponential}(\mu_b),
$$

其中 $\mu_b$ 为修复率。对应平均修复时间为

$$
\mathrm{MTTR}_b=\frac{1}{\mu_b}.
$$

此时修复时间分布函数为

$$
F_b(d)=1-\exp(-\mu_b d).
$$

若采用 Weibull 修复时间模型：

$$
D_b\sim \mathrm{Weibull}(k_b,\lambda_b),
$$

其中 $k_b$ 为形状参数，$\lambda_b$ 为尺度参数。对应分布函数为

$$
F_b(d)=1-\exp\left[-\left(\frac{d}{\lambda_b}\right)^{k_b}\right].
$$

指数分布适合描述无记忆修复过程；Weibull 分布可描述修复风险随时间变化的情形，因此更灵活。

## 3. BESS 可支撑时间

设基站 $b$ 在停电起始时刻 $t$ 的可用 BESS 能量为

$$
\overline{C}_{b,t}
=
(\mathrm{SOC}_{b,t}-\mathrm{SOC}^{\min})C_b,
$$

其中 $C_b$ 为 BESS 额定容量，$\mathrm{SOC}_{b,t}$ 为当前荷电状态，$\mathrm{SOC}^{\min}$ 为最低允许荷电状态。

若基站未来能耗轨迹为 $E_{b,t+s}$，则 BESS 可支撑时间定义为

$$
\tau_{b,t}
=
\sup
\left\{
 u\ge 0:
 \int_{0}^{u} E_{b,t+s}\,ds
 \le
 \overline{C}_{b,t}
\right\}.
$$

离散小时形式为

$$
\tau_{b,t}
=
\max
\left\{
 m:
 \sum_{j=0}^{m-1}E_{b,t+j}
 \le
 \overline{C}_{b,t}
\right\}.
$$

该变量表示：若从时刻 $t$ 开始停电，当前 BESS 在给定负载轨迹下最多能够维持基站运行多久。

因此，BESS 容量并不直接线性决定通信可靠性。同样的 BESS 容量在高负载时段对应较短的可支撑时间，在低负载时段对应较长的可支撑时间。

## 4. 通信不中断概率

通信不中断的事件为：

$$
D_b\le \tau_{b,t}.
$$

即电力供应或故障在 BESS 耗尽前恢复。于是，基站 $b$ 在时刻 $t$ 的条件通信可靠性定义为

$$
\mathcal{R}_{b,t}
=
\Pr(D_b\le \tau_{b,t}).
$$

若修复时间分布函数为 $F_b(\cdot)$，则

$$
\mathcal{R}_{b,t}=F_b(\tau_{b,t}).
$$

当 $D_b$ 服从指数分布时，有

$$
\mathcal{R}_{b,t}
=
1-\exp(-\mu_b\tau_{b,t}).
$$

该函数具有非线性递增和边际收益递减特征。即当 BESS 可支撑时间较短时，增加备用能量可以显著提升通信可靠性；当 BESS 已能支撑较长时间时，继续增加备用能量对可靠性的边际提升逐渐减小。

## 5. 考虑可调度容量后的通信可靠性

若从 BESS 中释放 $C^{\mathrm{disp}}_{b,t}$ 给电网调度，则剩余用于通信后备的能量为

$$
\overline{C}^{\mathrm{bk}}_{b,t}
=
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}.
$$

在该决策下，BESS 可支撑时间变为

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})
=
\sup
\left\{
 u\ge 0:
 \int_{0}^{u} E_{b,t+s}\,ds
 \le
 \overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}
\right\}.
$$

对应的通信可靠性为

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})
=
F_b\left(\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right).
$$

因此，可调度容量越大，留给通信后备的能量越少，BESS 可支撑时间越短，通信可靠性越低。

## 6. 通信可靠性约束下的 BESS 可调度容量优化

在考虑随机修复过程后，BESS 可调度容量优化问题可写为

$$
\begin{aligned}
\max_{C^{\mathrm{disp}}_{b,t}}\quad
& \pi_t C^{\mathrm{disp}}_{b,t} \\
\mathrm{s.t.}\quad
& \mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})\ge R_b^{\min}, \\
& 0\le C^{\mathrm{disp}}_{b,t}\le \overline{C}_{b,t},
\end{aligned}
$$

其中 $\pi_t$ 为单位可调度容量收益，$R_b^{\min}$ 为基站最低通信可靠性要求。

将可靠性表达式代入可得：

$$
F_b\left(\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right)
\ge
R_b^{\min}.
$$

若 $F_b$ 单调递增，则该约束等价于

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})
\ge
F_b^{-1}(R_b^{\min}).
$$

其中 $F_b^{-1}(R_b^{\min})$ 表示为了满足可靠性要求 $R_b^{\min}$，BESS 至少需要支撑的时间。

于是，通信可靠性约束可转化为能量约束：

$$
\int_{0}^{F_b^{-1}(R_b^{\min})}
E_{b,t+s}\,ds
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}.
$$

离散形式为

$$
\sum_{j=0}^{T_b^{\mathrm{rel}}-1}E_{b,t+j}
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t},
$$

其中

$$
T_b^{\mathrm{rel}}
=
F_b^{-1}(R_b^{\min}).
$$

因此，可调度容量的最大值为

$$
C^{\mathrm{disp},\star}_{b,t}
=
\left[
\overline{C}_{b,t}-
\int_{0}^{F_b^{-1}(R_b^{\min})}E_{b,t+s}\,ds
\right]_+.
$$

这说明可靠性约束下的可调度容量由修复时间分布的分位数和未来负载轨迹共同决定。

## 7. 期望通信中断时长

仅用“是否中断”仍然较粗糙。若停电持续时间超过 BESS 可支撑时间，则通信中断时长为

$$
I_{b,t}
=
[D_b-\tau_{b,t}]_+,
$$

其中 $[x]_+=\max\{x,0\}$。

期望通信中断时长定义为

$$
\mathrm{ECID}_{b,t}
=
\mathbb{E}\left[(D_b-\tau_{b,t})_+\right].
$$

若考虑停电发生率 $\lambda_b^{\mathrm{out}}$，可定义基站级通信中断时长指标：

$$
\mathrm{SAIDI}^{\mathrm{BS}}_b
=
\lambda_b^{\mathrm{out}}
\mathbb{E}\left[(D_b-\tau_{b,t})_+\right].
$$

该指标表示单位时间内，基站因 BESS 耗尽而产生的期望通信服务中断时长。

若 $D_b$ 服从指数分布，则

$$
\mathbb{E}\left[(D_b-\tau_{b,t})_+\right]
=
\int_{\tau_{b,t}}^{\infty}(d-\tau_{b,t})\mu_b e^{-\mu_b d}\,dd
=
\frac{e^{-\mu_b \tau_{b,t}}}{\mu_b}.
$$

因此，BESS 可支撑时间越长，期望通信中断时长按指数形式下降。

## 8. 期望未服务业务量

通信系统中，高峰时段中断和低谷时段中断的影响不同。设 $L_{b,t}$ 表示基站业务负载或通信需求。若 BESS 在 $\tau_{b,t}$ 后耗尽，则未服务业务量可定义为

$$
U_{b,t}
=
\int_{0}^{D_b}
L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}\,ds.
$$

对应的期望未服务业务量为

$$
\mathrm{EUT}_{b,t}
=
\mathbb{E}
\left[
\int_{0}^{D_b}
L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}\,ds
\right].
$$

该指标衡量由于 BESS 未能支撑到修复完成而造成的通信服务损失。相比单纯的后备能量短缺，期望未服务业务量更贴近通信系统服务质量。

## 9. 与原 Phase D 模型的关系

原 Phase D 链路为：

$$
\text{日前能耗预测}
\rightarrow
\text{固定窗口后备能量}
\rightarrow
\text{安全备用 reserve}
\rightarrow
\text{可调度容量}.
$$

改进后的可靠性建模链路为：

$$
\text{日前能耗预测}
\rightarrow
\text{BESS 可支撑时间}
\rightarrow
\text{修复前不中断概率}
\rightarrow
\text{通信可靠性约束}
\rightarrow
\text{可调度容量}.
$$

因此，BESS 可调度容量不再由固定窗口内的能量 reserve 简单决定，而是由随机修复时间分布、未来负载轨迹、SOC 状态和最低通信可靠性要求共同决定。

## 10. 推荐可靠性指标

建议在 Phase D 中至少使用以下三个层次的通信可靠性指标。

| 指标 | 数学表达 | 含义 |
|---|---|---|
| 通信不中断概率 | $\Pr(D_b\le \tau_{b,t})$ | 停电后 BESS 是否能够支撑到修复完成。 |
| 期望通信中断时长 | $\mathbb{E}[(D_b-\tau_{b,t})_+]$ | 若未能撑到修复完成，平均中断多久。 |
| 期望未服务业务量 | $\mathbb{E}\left[\int_0^{D_b}L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}ds\right]$ | 通信中断造成的业务服务损失。 |

其中，最适合作为主优化约束的是通信不中断概率：

$$
\Pr(D_b\le \tau_{b,t})\ge R_b^{\min}.
$$

最适合作为评估指标的是期望通信中断时长和期望未服务业务量：

$$
\mathbb{E}[(D_b-\tau_{b,t})_+],
$$

$$
\mathbb{E}
\left[
\int_0^{D_b}
L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}ds
\right].
$$

## 11. 可替换的论文表述

原表述为：

> 因此，Phase D 将 BESS 调度问题转化为对累计后备能量需求分位数的可靠估计问题。

建议替换为：

> 因此，Phase D 不应仅将 BESS 调度问题表述为固定后备窗口内累计能量需求的分位数估计问题，而应进一步引入停电修复时间分布，将通信可靠性刻画为 BESS 可支撑时间超过随机修复时间的概率。由此，BESS 可调度容量由通信不中断概率约束决定，而非与后备容量呈简单线性关系。

更数学化的英文表达为：

> Let $D_b$ denote the random outage repair time and $\tau_{b,t}$ denote the BESS autonomy time under the retained backup energy. The communication reliability is defined as $\mathcal{R}_{b,t}=\Pr(D_b\le\tau_{b,t})=F_b(\tau_{b,t})$. The dispatchable BESS capacity is therefore determined by the reliability constraint $\mathcal{R}_{b,t}\ge R_b^{\min}$, rather than by a fixed-duration energy-reserve constraint alone.
