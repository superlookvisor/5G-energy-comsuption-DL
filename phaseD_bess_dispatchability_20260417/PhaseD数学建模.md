# Phase D：考虑随机修复过程的基站 BESS 通信可靠性约束可调度容量模型

## A. 问题定义

Phase D 将 Phase A--C 得到的 5G 基站日前能耗预测结果转化为后备电池储能系统（battery energy storage system, BESS）的通信可靠性评估与可调度容量决策模型。与仅以固定后备窗口内累计能量充裕性衡量可靠性的模型不同，本文显式引入停电修复时间随机变量，并将通信可靠性定义为 BESS 可支撑时间超过随机修复时间的概率。

Phase A 给出基站静态基础功耗估计 $p^{\mathrm{base}}_b$，Phase B 给出基于日前可用物理代理的能耗点预测，Phase C 进一步给出物理引导残差概率预测及 calibrated prediction interval。Phase D 的增量贡献在于：不再以提升单小时预测精度为目标，而是将 Phase C 的日前概率预测映射为 BESS autonomy time、通信不中断概率、期望通信中断时长、期望未服务业务量和可靠性约束下的可调度容量。

令 $\mathcal{B}$ 表示基站集合，$b\in\mathcal{B}$ 表示基站索引；$\mathcal{C}_b$ 表示基站 $b$ 下属 cell 集合。对任一日前预测起点 $t_0\in\mathcal{T}_0$，小时级预测时距集合为

$$
\mathcal{H}=\{1,2,\ldots,H\},\quad H=24.
\tag{1}
$$

目标预测时刻为 $t=t_0+h$，$h\in\mathcal{H}$。基站小时级真实能耗继承 Phase A--C 的分解结构：

$$
E_{b,t}=p^{\mathrm{base}}_b+E^{\mathrm{dyn}}_{b,t}(\ell_{b,t})+E^{\mathrm{save}}_{b,t}(\boldsymbol{s}_{b,t})+\xi_{b,t},
\tag{2}
$$

其中 $p^{\mathrm{base}}_b$ 为静态基础功耗，$E^{\mathrm{dyn}}_{b,t}(\cdot)$ 为业务负载驱动的动态能耗项，$E^{\mathrm{save}}_{b,t}(\cdot)$ 为节能策略导致的能耗修正项，$\xi_{b,t}$ 为未被物理代理解释的随机残差。

Phase C 给出的最终日前点预测为

$$
\widehat{E}^{C}_{b,t}=\widehat{E}^{\mathrm{phys}}_{b,t}+\widehat{\mu}_{b,t},
\tag{3}
$$

其中 $\widehat{E}^{\mathrm{phys}}_{b,t}$ 表示由 Phase A 和 Phase B 构造的物理基线能耗预测值，$\widehat{\mu}_{b,t}$ 为 PG-RNP 残差均值估计。Phase C 同时给出置信水平为 $1-\alpha$ 的校准预测区间

$$
\widehat{\mathcal{I}}^{1-\alpha}_{b,t}=[\widehat{L}_{b,t},\widehat{U}_{b,t}],
\tag{4}
$$

其中 $\widehat{L}_{b,t}$ 和 $\widehat{U}_{b,t}$ 分别为校准后的预测区间下界和上界。

本文将停电持续时间建模为随机修复时间 $D_b$，并以 BESS 在保留通信后备能量后的可支撑时间 $\tau_{b,t}$ 作为连接能耗预测与通信可靠性的中间变量。通信服务不中断的事件为

$$
D_b\le \tau_{b,t}.
\tag{5}
$$

因此，BESS 可调度容量不再由固定时长 reserve 线性决定，而由修复时间分布、未来能耗轨迹、BESS 当前 SOC 和通信可靠性约束共同决定。

## B. 符号说明

| 符号 | 含义 |
|---|---|
| $\mathcal{B}$ | 5G 基站集合。 |
| $\mathcal{C}_b$ | 基站 $b$ 下属的 cell 集合。 |
| $t_0$ | 日前预测起点。 |
| $t$ | 停电或调度决策发生时刻。 |
| $h\in\mathcal{H}$ | 小时级预测时距，$\mathcal{H}=\{1,\ldots,H\}$。 |
| $E_{b,t}$ | 基站 $b$ 在时刻 $t$ 的真实小时级能耗。 |
| $\widehat{E}^{C}_{b,t}$ | Phase C 给出的日前点预测能耗。 |
| $\widehat{L}_{b,t},\widehat{U}_{b,t}$ | Phase C 的 CQR 校准预测区间下界与上界。 |
| $D_b$ | 基站 $b$ 发生停电后的随机修复时间或停电持续时间。 |
| $F_b(d)$ | $D_b$ 的分布函数，即 $F_b(d)=\Pr(D_b\le d)$。 |
| $\mu_b$ | 指数修复时间模型中的修复率。 |
| $C_b$ | 基站 $b$ 配置的 BESS 额定能量容量。 |
| $\mathrm{SOC}_{b,t}$ | BESS 在时刻 $t$ 的荷电状态。 |
| $\mathrm{SOC}^{\min}$ | BESS 最低允许荷电状态。 |
| $\overline{C}_{b,t}$ | 扣除最低 SOC 后可用于通信后备与调度的 BESS 可用能量。 |
| $C^{\mathrm{disp}}_{b,t}$ | 时刻 $t$ 可释放给电网调度的 BESS 能量。 |
| $\overline{C}^{\mathrm{bk}}_{b,t}$ | 扣除可调度能量后保留给通信后备的 BESS 能量。 |
| $\tau_{b,t}$ | BESS 在给定后备能量和能耗轨迹下的可支撑时间。 |
| $\mathcal{R}_{b,t}$ | 通信不中断概率。 |
| $R_b^{\min}$ | 基站 $b$ 的最低通信可靠性要求。 |
| $I_{b,t}$ | 通信中断时长。 |
| $L_{b,t}$ | 基站业务负载或通信需求。 |
| $U_{b,t}$ | 未服务业务量。 |
| $\pi_t$ | 单位可调度 BESS 容量收益。 |
| $\lambda_I,\lambda_U$ | 通信中断时长和未服务业务量惩罚系数。 |
| $q^{R}_{1-\epsilon,T}$ | 对持续时间 $T$ 的累计能耗预测误差进行 Aggregate-CQR 校准得到的单侧分位数。 |

BESS 可用能量定义为

$$
\overline{C}_{b,t}=\max\{0,(\mathrm{SOC}_{b,t}-\mathrm{SOC}^{\min})C_b\}.
\tag{6}
$$

若缺少实测 BESS 容量，采用参数化容量情景

$$
C_b=\chi p^{\mathrm{base}}_b,
\tag{7}
$$

其中 $\chi>0$ 表示按静态基础功率折算的可支撑小时数。若不显式模拟 SOC 动态，可取 $\mathrm{SOC}_{b,t}=1$ 且 $\mathrm{SOC}^{\min}=0$。

## C. 数学模型构建

### C.1 随机修复时间模型

设 $D_b$ 表示基站 $b$ 从停电发生到供电恢复或故障修复所需的随机时间，其分布函数为

$$
F_b(d)=\Pr(D_b\le d).
\tag{8}
$$

若采用指数修复模型，则

$$
D_b\sim \mathrm{Exponential}(\mu_b),
\tag{9}
$$

其中 $\mu_b$ 为修复率，平均修复时间为

$$
\mathrm{MTTR}_b=\frac{1}{\mu_b}.
\tag{10}
$$

对应分布函数为

$$
F_b(d)=1-\exp(-\mu_b d).
\tag{11}
$$

更一般地，若采用 Weibull 修复模型，

$$
D_b\sim \mathrm{Weibull}(k_b,\lambda_b),
\tag{12}
$$

则

$$
F_b(d)=1-\exp\left[-\left(\frac{d}{\lambda_b}\right)^{k_b}\right].
\tag{13}
$$

指数模型适用于无记忆修复过程，Weibull 模型可刻画修复率随停电持续时间变化的情形。

### C.2 BESS 可支撑时间

若停电起始时刻为 $t$，并且释放 $C^{\mathrm{disp}}_{b,t}$ 给电网调度，则保留给通信后备的 BESS 能量为

$$
\overline{C}^{\mathrm{bk}}_{b,t}=\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}.
\tag{14}
$$

在连续时间形式下，BESS 可支撑时间定义为

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})
=
\sup\left\{u\ge0:
\int_{0}^{u}E_{b,t+s}\,ds
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}
\right\}.
\tag{15}
$$

在小时级离散调度中，令 $\Delta t=1$ h，则

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})
=
\max\left\{m\in\mathbb{Z}_{\ge0}:
\sum_{j=0}^{m-1}E_{b,t+j}\Delta t
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}
\right\}.
\tag{16}
$$

式 (15)--(16) 表明，同一 BESS 容量在不同业务负载时段对应不同可支撑时间。因此，通信可靠性与 BESS 容量并非线性关系，而是通过未来能耗轨迹的积分或累加间接关联。

### C.3 通信不中断概率

通信不中断事件为修复时间不超过 BESS 可支撑时间：

$$
D_b\le \tau_{b,t}(C^{\mathrm{disp}}_{b,t}).
\tag{17}
$$

因此，给定调度决策 $C^{\mathrm{disp}}_{b,t}$ 时的通信可靠性定义为

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})
=
\Pr\left(D_b\le \tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right).
\tag{18}
$$

若 $D_b$ 的分布函数为 $F_b(\cdot)$，则

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})
=
F_b\left(\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right).
\tag{19}
$$

在指数修复模型下，通信可靠性为

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})
=
1-\exp\left[-\mu_b\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right].
\tag{20}
$$

该函数对 $\tau_{b,t}$ 单调递增且边际增益递减，从而刻画了 BESS 后备能力对通信可靠性的非线性贡献。

### C.4 通信可靠性约束下的可调度容量优化

给定最低通信可靠性要求 $R_b^{\min}\in(0,1)$，Phase D 的基本可调度容量优化问题为

$$
\begin{aligned}
\max_{C^{\mathrm{disp}}_{b,t}}\quad
& \pi_t C^{\mathrm{disp}}_{b,t} \\
\mathrm{s.t.}\quad
& \mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})\ge R_b^{\min}, \\
& 0\le C^{\mathrm{disp}}_{b,t}\le \overline{C}_{b,t}.
\end{aligned}
\tag{21}
$$

由于 $F_b(\cdot)$ 单调递增，可靠性约束等价于

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})
\ge
T_b^{\mathrm{rel}},
\tag{22}
$$

其中

$$
T_b^{\mathrm{rel}}=F_b^{-1}(R_b^{\min})
\tag{23}
$$

表示满足最低通信可靠性要求所需的最小 BESS 可支撑时间。对指数修复模型，有

$$
T_b^{\mathrm{rel}}
=
-\frac{\ln(1-R_b^{\min})}{\mu_b}.
\tag{24}
$$

因此，式 (21) 可进一步转化为能量约束：

$$
\int_0^{T_b^{\mathrm{rel}}}E_{b,t+s}\,ds
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}.
\tag{25}
$$

离散形式为

$$
\sum_{j=0}^{\lceil T_b^{\mathrm{rel}}\rceil-1}E_{b,t+j}\Delta t
\le
\overline{C}_{b,t}-C^{\mathrm{disp}}_{b,t}.
\tag{26}
$$

若未来能耗轨迹已知，则最大可调度容量具有闭式表达：

$$
C^{\mathrm{disp},\star}_{b,t}
=
\left[
\overline{C}_{b,t}-
\int_0^{T_b^{\mathrm{rel}}}E_{b,t+s}\,ds
\right]_+.
\tag{27}
$$

式 (27) 表明，可调度容量由修复时间分布的可靠性分位数和未来负载能耗共同决定。

### C.5 基于 Phase C 预测的可靠性安全调度

实际日前调度中，$E_{b,t+s}$ 未知，只能由 Phase C 的概率预测给出。令

$$
\widehat{R}^{C}_{b,t}(T)
=
\sum_{j=0}^{T-1}\widehat{E}^{C}_{b,t+j}\Delta t
\tag{28}
$$

表示持续时间 $T$ 内的 Phase C 累计点预测能耗。令

$$
R^{\mathrm{true}}_{b,t}(T)
=
\sum_{j=0}^{T-1}E_{b,t+j}\Delta t
\tag{29}
$$

表示同一持续时间内的真实累计能耗。为了避免简单相加逐小时上界导致过度保守，Phase D 对累计预测误差进行 Aggregate-CQR 校准：

$$
s_i^{(T)}=R^{\mathrm{true}}_i(T)-\widehat{R}^{C}_i(T),\quad i\in\mathcal{W}^{\mathrm{cal}}_T.
\tag{30}
$$

给定能耗不确定性风险水平 $\epsilon_E\in(0,1)$，令

$$
q^{R}_{1-\epsilon_E,T}
=\mathrm{Quantile}_{1-\epsilon_E}\left(\{s_i^{(T)}:i\in\mathcal{W}^{\mathrm{cal}}_T\}\right).
\tag{31}
$$

于是，持续时间 $T$ 内的可靠累计能耗上界为

$$
\widehat{R}^{U}_{b,t}(T)=
\widehat{R}^{C}_{b,t}(T)+q^{R}_{1-\epsilon_E,T}.
\tag{32}
$$

将 $T=\lceil T_b^{\mathrm{rel}}\rceil$ 代入，得到考虑能耗预测不确定性的可调度容量：

$$
C^{\mathrm{disp},\star}_{b,t}
=
\left[
\overline{C}_{b,t}-
\widehat{R}^{U}_{b,t}\left(\lceil T_b^{\mathrm{rel}}\rceil\right)
\right]_+.
\tag{33}
$$

式 (33) 同时包含两个不确定性来源：修复时间不确定性通过 $T_b^{\mathrm{rel}}=F_b^{-1}(R_b^{\min})$ 进入模型，能耗预测不确定性通过 Aggregate-CQR 累计上界 $\widehat{R}^{U}_{b,t}(T)$ 进入模型。

若需要联合可靠性保证，可采用风险分配：

$$
\epsilon_D+\epsilon_E\le \epsilon,
\tag{34}
$$

其中 $\epsilon_D=1-R_b^{\min}$ 表示修复时间侧风险，$\epsilon_E$ 表示能耗预测侧风险，总风险水平不超过 $\epsilon$。

## D. 通信可靠性评价指标

### D.1 通信不中断概率

对给定调度决策 $C^{\mathrm{disp}}_{b,t}$，通信不中断概率为

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})
=F_b\left(\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\right).
\tag{35}
$$

在测试集 $\mathcal{T}^{\mathrm{eval}}$ 上的平均通信可靠性为

$$
\overline{\mathcal{R}}
=
\frac{1}{|\mathcal{T}^{\mathrm{eval}}|}
\sum_{(b,t)\in\mathcal{T}^{\mathrm{eval}}}
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t}).
\tag{36}
$$

### D.2 期望通信中断时长

若停电持续时间超过 BESS 可支撑时间，则通信中断时长为

$$
I_{b,t}=[D_b-\tau_{b,t}]_+.
\tag{37}
$$

期望通信中断时长定义为

$$
\mathrm{ECID}_{b,t}
=\mathbb{E}\left[(D_b-\tau_{b,t})_+\right].
\tag{38}
$$

若 $D_b\sim\mathrm{Exponential}(\mu_b)$，则

$$
\mathrm{ECID}_{b,t}
=
\frac{\exp(-\mu_b\tau_{b,t})}{\mu_b}.
\tag{39}
$$

若考虑停电发生率 $\lambda_b^{\mathrm{out}}$，基站级通信中断时长指标可写为

$$
\mathrm{SAIDI}^{\mathrm{BS}}_b
=
\lambda_b^{\mathrm{out}}
\mathbb{E}\left[(D_b-\tau_{b,t})_+\right].
\tag{40}
$$

### D.3 期望未服务业务量

设 $L_{b,t}$ 表示基站业务负载或通信需求。若 BESS 在 $\tau_{b,t}$ 后耗尽，则未服务业务量为

$$
U_{b,t}
=
\int_0^{D_b}L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}\,ds.
\tag{41}
$$

对应的期望未服务业务量为

$$
\mathrm{EUT}_{b,t}
=
\mathbb{E}\left[
\int_0^{D_b}L_{b,t+s}\mathbf{1}\{s>\tau_{b,t}\}\,ds
\right].
\tag{42}
$$

该指标用于区分高峰时段中断与低谷时段中断的通信服务影响。

### D.4 经济性目标

若同时考虑可调度容量收益、通信中断时长惩罚和未服务业务量惩罚，则净收益可定义为

$$
J
=
\sum_{(b,t)\in\mathcal{T}^{\mathrm{eval}}}
\left(
\pi_t C^{\mathrm{disp}}_{b,t}-
\lambda_I\mathrm{ECID}_{b,t}-
\lambda_U\mathrm{EUT}_{b,t}
\right).
\tag{43}
$$

其中 $\lambda_I$ 和 $\lambda_U$ 分别表示单位通信中断时长惩罚和单位未服务业务量惩罚。该目标函数刻画 BESS 调度收益与通信服务可靠性损失之间的权衡。

## E. 模型性质

当修复时间分布 $F_b$ 给定且单调递增时，通信可靠性约束

$$
\mathcal{R}_{b,t}(C^{\mathrm{disp}}_{b,t})\ge R_b^{\min}
\tag{44}
$$

可等价转化为最小可支撑时间约束

$$
\tau_{b,t}(C^{\mathrm{disp}}_{b,t})\ge F_b^{-1}(R_b^{\min}).
\tag{45}
$$

进一步地，在采用累计能耗上界 $\widehat{R}^{U}_{b,t}(T)$ 后，该约束可写为线性能量约束

$$
C^{\mathrm{disp}}_{b,t}+
\widehat{R}^{U}_{b,t}\left(\lceil F_b^{-1}(R_b^{\min})\rceil\right)
\le
\overline{C}_{b,t}.
\tag{46}
$$

因此，在不考虑二进制市场参与变量、充放电互斥约束和网络潮流约束时，可靠性约束下的最大可调度容量具有闭式解 (33)。若加入多时段 SOC 动态，模型可扩展为线性规划：

$$
\mathrm{SOC}_{b,t+1}C_b
=
\mathrm{SOC}_{b,t}C_b
+\eta^{\mathrm{ch}}P^{\mathrm{ch}}_{b,t}\Delta t
-\frac{1}{\eta^{\mathrm{dis}}}P^{\mathrm{dis}}_{b,t}\Delta t,
\tag{47}
$$

并附加

$$
\mathrm{SOC}^{\min}\le \mathrm{SOC}_{b,t}\le \mathrm{SOC}^{\max}.
\tag{48}
$$

若进一步引入充放电互斥、市场参与启停或配电网潮流约束，则模型成为混合整数线性规划，但通信可靠性约束仍可通过式 (46) 以线性形式嵌入。

计算复杂度主要由三部分构成：构造不同 $T$ 下的累计能耗轨迹，复杂度为 $O(|\mathcal{B}||\mathcal{T}_0|H^2)$；对 Aggregate-CQR 校准分数排序，复杂度为 $O(n_{\mathrm{cal}}\log n_{\mathrm{cal}})$；评估通信可靠性和可调度容量，复杂度为 $O(|\mathcal{T}^{\mathrm{eval}}|)$。校准完成后，Phase D 的在线决策只需计算 $F_b^{-1}(R_b^{\min})$、累计能耗上界和式 (33) 的闭式可调度容量。
