# Phase E：通信可靠性约束下的 IEEE-33 配电网-5G 基站 BESS 联合调度模型

## A. 问题定位与增量贡献

Phase D 完成了**单基站级**的通信可靠性约束可调度容量建模，闭式地刻画了 BESS autonomy time、通信不中断概率和可调度容量三者的映射关系。然而 Phase D 尚未回答三个工程上必须解决的问题：

1. **多基站联合调度**。配电网中多个基站共享同一根节点购电功率，局部节点电压和支路容量耦合，基站间 BESS 决策不可独立给出。
2. **经济成本显式刻画**。Phase D 的目标函数仅以可调度容量收益和通信中断惩罚相加，未纳入分时电价下的购电成本和电池的衰减成本。
3. **能耗预测不确定性的分布鲁棒建模**。Phase C 的 CQR 上界直接作为能耗约束会过度保守；Phase D 只在单点约束上校准，未给出分布鲁棒意义下的决策。

Phase E 的核心贡献是：将 Phase A--C 的日前概率预测和 Phase D 的通信可靠性约束一并嵌入**配电网运行约束**下的 BESS 调度问题，提出一套**Conformalized Wasserstein Distributionally Robust Chance Constraint (CW-DRCC)** 框架，使模型在保持 MISOCP 可解性的前提下具备有限样本、分布无关的可靠性保证。

## B. 符号说明

| 符号 | 含义 |
|---|---|
| $\mathcal{N}$ | 配电网节点集合，IEEE-33 系统下 $|\mathcal{N}|=33$，根节点为 $0$。 |
| $\mathcal{L}$ | 配电网支路集合，辐射状结构下 $|\mathcal{L}|=|\mathcal{N}|-1=32$。 |
| $\mathcal{B}$ | 5G 基站集合，本文设定每个非根节点部署一个基站，$|\mathcal{B}|=32$。 |
| $j(b)$ | 基站 $b$ 所在节点。 |
| $b(j)$ | 节点 $j$ 上部署的基站索引。 |
| $\mathcal{T}$ | 调度时段集合，$\mathcal{T}=\{1,\dots,H\}$，$H=24$，$\Delta t=1$ h。 |
| $r_{ij},x_{ij}$ | 支路 $(i,j)\in\mathcal{L}$ 的电阻、电抗（p.u.）。 |
| $\overline{v},\underline{v}$ | 节点电压平方上下限，典型取 $[0.95^2,1.05^2]$。 |
| $\overline{\ell}_{ij}$ | 支路电流平方上限。 |
| $P^{\mathrm{grid}}_{0,t}$ | 根节点从上级电网购入的有功功率（kW）。 |
| $\pi^{\mathrm{buy}}_t$ | 时段 $t$ 的分时购电电价（元/kWh）。 |
| $E_{b,t}$ | 基站 $b$ 在时段 $t$ 的真实小时级能耗。 |
| $\widehat{E}^{C}_{b,t}$ | Phase C 给出的点预测。 |
| $s^{(T)}_{b,t,i}$ | Phase C 第 $i$ 个校准样本的 $T$ 小时累计能耗误差分数。 |
| $C_b$ | 基站 $b$ 的 BESS 额定能量容量（kWh）。 |
| $P^{\mathrm{rate}}_b$ | BESS 额定功率（kW）。 |
| $\eta^{\mathrm{ch}},\eta^{\mathrm{dis}}$ | 充放电效率。 |
| $\mathrm{SOC}_{b,t}$ | 荷电状态，$\in[\underline{\mathrm{SOC}},\overline{\mathrm{SOC}}]$。 |
| $P^{\mathrm{ch}}_{b,t},P^{\mathrm{dis}}_{b,t}$ | 充/放电功率（kW），非负。 |
| $u^{\mathrm{ch}}_{b,t},u^{\mathrm{dis}}_{b,t}$ | 充/放电状态 0-1 变量，互斥。 |
| $P_{ij,t},Q_{ij,t}$ | 支路 $(i,j)$ 的有功、无功潮流。 |
| $\ell_{ij,t}$ | 支路 $(i,j)$ 的电流平方。 |
| $v_{j,t}$ | 节点 $j$ 的电压平方。 |
| $T^{\mathrm{rel}}_b$ | 基站 $b$ 满足最低通信可靠性 $R^{\min}_b$ 所需的最小 BESS autonomy time，由 $F_b^{-1}(R^{\min}_b)$ 给出。 |
| $\widehat{R}^{U}_{b,t}(T)$ | 基站 $b$ 从 $t$ 起 $T$ 小时的鲁棒累计能耗上界（CW-DRCC 定理给出）。 |
| $c^{\mathrm{wear}}_k$ | 第 $k$ 段 DOD 区间的单位吞吐衰减成本（元/kWh）。 |
| $\mathcal{P}_N(\varepsilon_N)$ | 围绕 $N$ 个校准样本的 Wasserstein-1 模糊集，半径 $\varepsilon_N$。 |

## C. BESS 容量与功率设计

### C.1 基于基站最大能耗的容量设计

令 $E^{\max}_b=\max_{t\in\mathcal{T}}\widehat{E}^{C}_{b,t}$ 表示基站 $b$ 在日前预测窗口内的最大小时能耗估计。BESS 额定能量容量取
$$
C_b=3\cdot E^{\max}_b\quad(\mathrm{kWh})
\tag{1}
$$
即 3 个峰值小时的存储能力。BESS 额定功率按 0.5C 配置：
$$
P^{\mathrm{rate}}_b=\tfrac{1}{2}E^{\max}_b\quad(\mathrm{kW})
\tag{2}
$$
此时放完满电需要 6 小时（$C_b/P^{\mathrm{rate}}_b=6$ h），符合 5G 基站后备电池典型 C 率设计。

### C.2 可用能量与 SOC 边界

$$
\underline{\mathrm{SOC}}\le \mathrm{SOC}_{b,t}\le \overline{\mathrm{SOC}}
\tag{3}
$$
典型取 $\underline{\mathrm{SOC}}=0.1,\overline{\mathrm{SOC}}=0.9$。日前调度起末 SOC 等式保证滚动可行性：
$$
\mathrm{SOC}_{b,0}=\mathrm{SOC}_{b,H}=\mathrm{SOC}^{\mathrm{init}}
\tag{4}
$$

## D. BESS 运行约束

### D.1 SOC 动态与功率限额

$$
\mathrm{SOC}_{b,t+1}C_b=\mathrm{SOC}_{b,t}C_b+\eta^{\mathrm{ch}}P^{\mathrm{ch}}_{b,t}\Delta t-\frac{1}{\eta^{\mathrm{dis}}}P^{\mathrm{dis}}_{b,t}\Delta t
\tag{5}
$$
$$
0\le P^{\mathrm{ch}}_{b,t}\le P^{\mathrm{rate}}_b u^{\mathrm{ch}}_{b,t},\quad 0\le P^{\mathrm{dis}}_{b,t}\le P^{\mathrm{rate}}_b u^{\mathrm{dis}}_{b,t}
\tag{6}
$$

### D.2 充放电互斥

$$
u^{\mathrm{ch}}_{b,t}+u^{\mathrm{dis}}_{b,t}\le 1,\quad u^{\mathrm{ch}}_{b,t},u^{\mathrm{dis}}_{b,t}\in\{0,1\}
\tag{7}
$$

### D.3 基站功率平衡（不可中断）

基站自身能耗 $\widetilde{E}_{b,t}$ 必须由电网或 BESS 放电覆盖，不允许削负荷：
$$
P^{\mathrm{bus}}_{j(b),t}+P^{\mathrm{dis}}_{b,t}-P^{\mathrm{ch}}_{b,t}=\widetilde{E}_{b,t}
\tag{8}
$$
其中 $P^{\mathrm{bus}}_{j,t}$ 是节点 $j$ 从配电网获得的有功功率（由潮流约束决定），$\widetilde{E}_{b,t}$ 的具体取值由不确定性建模方式决定（参见 § F）。

## E. 配电网 DistFlow SOCP 松弛约束

### E.1 分支潮流方程

对辐射状配电网采用 Baran-Wu DistFlow 模型。令 $(i,j)\in\mathcal{L}$ 表示父节点 $i$ 到子节点 $j$ 的支路，$\mathcal{C}(j)$ 表示节点 $j$ 的子节点集合。

**有功平衡**
$$
P_{ij,t}-r_{ij}\ell_{ij,t}-p^{\mathrm{inj}}_{j,t}=\sum_{k\in\mathcal{C}(j)}P_{jk,t}
\tag{9}
$$

**无功平衡**
$$
Q_{ij,t}-x_{ij}\ell_{ij,t}-q^{\mathrm{inj}}_{j,t}=\sum_{k\in\mathcal{C}(j)}Q_{jk,t}
\tag{10}
$$

**电压降方程**
$$
v_{j,t}=v_{i,t}-2(r_{ij}P_{ij,t}+x_{ij}Q_{ij,t})+(r_{ij}^2+x_{ij}^2)\ell_{ij,t}
\tag{11}
$$

**SOCP 锥松弛**（将原非凸等式 $\ell_{ij,t}v_{i,t}=P_{ij,t}^2+Q_{ij,t}^2$ 松弛为凸锥）
$$
P_{ij,t}^2+Q_{ij,t}^2\le v_{i,t}\ell_{ij,t}
\tag{12}
$$
对辐射状网络在多数负荷场景下该松弛紧，解满足原始等式。

### E.2 安全运行边界

$$
\underline{v}\le v_{j,t}\le \overline{v},\quad 0\le \ell_{ij,t}\le \overline{\ell}_{ij},\quad v_{0,t}=V_0^2
\tag{13}
$$
根节点电压平方固定为 $V_0^2$（通常 1.0 p.u.）。

### E.3 节点注入功率

$$
p^{\mathrm{inj}}_{j,t}=\mathbb{1}\{j=0\}P^{\mathrm{grid}}_{0,t}-\widetilde{E}_{b(j),t}-P^{\mathrm{ch}}_{b(j),t}+P^{\mathrm{dis}}_{b(j),t}
\tag{14}
$$
$$
q^{\mathrm{inj}}_{j,t}=\mathbb{1}\{j=0\}Q^{\mathrm{grid}}_{0,t}-\widetilde{Q}_{b(j),t}
\tag{15}
$$
基站无功需求 $\widetilde{Q}_{b(j),t}=\widetilde{E}_{b(j),t}\tan\varphi$，功率因数 $\cos\varphi$ 通常取 0.95。

### E.4 购电功率非负

只购不售：
$$
P^{\mathrm{grid}}_{0,t}\ge 0,\quad Q^{\mathrm{grid}}_{0,t}\in\mathbb{R}
\tag{16}
$$

## F. 能耗不确定性的分布鲁棒建模：CW-DRCC

### F.1 不确定性来源与校准数据

基站 $b$ 在时段 $t$ 的真实能耗写为
$$
E_{b,t}=\widehat{E}^{C}_{b,t}+\xi_{b,t}
\tag{17}
$$
其中 $\xi_{b,t}$ 为 Phase C 未能捕获的残差。对任意持续时间 $T$，累计预测误差样本
$$
s^{(T)}_{b,t,i}=\sum_{j=0}^{T-1}E_{b,t+j,i}-\sum_{j=0}^{T-1}\widehat{E}^{C}_{b,t+j,i},\quad i\in\mathcal{W}^{\mathrm{cal}}_T
\tag{18}
$$
由 Phase C 校准集直接提供，记总样本数为 $N=|\mathcal{W}^{\mathrm{cal}}_T|$。经验分布记为 $\widehat{\mathbb{P}}_N^{(T)}=\frac{1}{N}\sum_{i=1}^{N}\delta_{s^{(T)}_{b,t,i}}$。

### F.2 Wasserstein 模糊集与半径的 conformal 校准

定义围绕 $\widehat{\mathbb{P}}_N^{(T)}$ 的 Wasserstein-1 模糊集
$$
\mathcal{P}_N^{(T)}(\varepsilon_N)=\left\{\mathbb{Q}\in\mathcal{M}(\mathbb{R}):W_1(\mathbb{Q},\widehat{\mathbb{P}}_N^{(T)})\le \varepsilon_N\right\}
\tag{19}
$$
其中
$$
W_1(\mathbb{Q}_1,\mathbb{Q}_2)=\inf_{\Pi}\int|s_1-s_2|\,\mathrm{d}\Pi(s_1,s_2)
\tag{20}
$$
Wasserstein 半径基于 Phase C 的 split conformal 分数校准：
$$
\varepsilon_N=\sigma_N\cdot q_{1-\alpha}\!\left(|s^{(T)}_{b,t,i}-\bar{s}^{(T)}_{b,t}|:i\in\mathcal{W}^{\mathrm{cal}}_T\right)
\tag{21}
$$
其中 $\bar{s}^{(T)}_{b,t}$ 为校准样本均值，$\sigma_N=O(N^{-1/(d+2)})$ 为 Fournier-Guillin 浓度系数（在 $d=1$ 维下退化为 $O(N^{-1/3})$），$\alpha$ 为不确定性风险水平。

**命题 1（有限样本保证）**：若 $\xi$ 的真实分布 $\mathbb{P}^\star$ 具有紧支撑和有限 $p$ 阶矩（$p>1$），则以概率至少 $1-\beta$ 有 $\mathbb{P}^\star\in\mathcal{P}_N^{(T)}(\varepsilon_N)$，从而后续 DRCC 的可行解也以概率 $1-\beta$ 满足真实机会约束。

### F.3 分布鲁棒累计能耗约束

基站通信可靠性约束等价要求累计能耗不超过可用能量。定义
$$
A_{b,t}(T)=\left(\mathrm{SOC}_{b,t}-\underline{\mathrm{SOC}}\right)C_b-\sum_{j=0}^{T-1}\widehat{E}^{C}_{b,t+j}
\tag{22}
$$
为扣除点预测后的剩余可用能量。分布鲁棒机会约束写为
$$
\inf_{\mathbb{Q}\in\mathcal{P}_N^{(T)}(\varepsilon_N)}\mathbb{Q}\!\left(\sum_{j=0}^{T-1}\xi_{b,t+j}\le A_{b,t}(T)\right)\ge 1-\alpha
\tag{23}
$$

### F.4 Tractable Reformulation

**定理 1（CW-DRCC 的 SOCP 等价形式）**：在 Wasserstein-1 模糊集下，式 (23) 等价于存在辅助变量 $\lambda\ge 0,\{z_i\}_{i=1}^{N}$ 使得
$$
\begin{aligned}
&\lambda\varepsilon_N+\frac{1}{N}\sum_{i=1}^{N}z_i\le \alpha A_{b,t}(T)\\
&z_i\ge s^{(T)}_{b,t,i}-A_{b,t}(T)-\lambda\cdot 0,\quad\forall i\\
&z_i\ge (1-\alpha)(s^{(T)}_{b,t,i}-A_{b,t}(T))+\alpha\lambda\cdot 0,\quad\forall i\\
&z_i\ge 0,\quad \lambda\ge 0
\end{aligned}
\tag{24}
$$
该表示由 Esfahani-Kuhn (2018) 的 Wasserstein DRO 对偶理论和 Chen et al. (2024) 的机会约束重构合并得到。对可分凸损失函数，最紧 Lipschitz 常数为 1，因此 $\varepsilon_N$ 直接乘以对偶变量 $\lambda$ 即构成鲁棒裕度。

**推论 1（鲁棒能耗上界的闭式表达）**：令 $q^{(T)}_{1-\alpha}=\mathrm{Quantile}_{1-\alpha}\{s^{(T)}_{b,t,i}\}$ 为经验 CQR 分位数。将 (24) 代入 $A_{b,t}(T)$ 的定义，得到等价的鲁棒累计能耗上界
$$
\widehat{R}^{U,\mathrm{CW}}_{b,t}(T)=\sum_{j=0}^{T-1}\widehat{E}^{C}_{b,t+j}+q^{(T)}_{1-\alpha}+\varepsilon_N
\tag{25}
$$
相比 Phase D 的 Aggregate-CQR 上界 $\widehat{R}^{U}_{b,t}(T)=\widehat{R}^{C}_{b,t}(T)+q^{R}_{1-\varepsilon,T}$，CW-DRCC 额外引入 Wasserstein 半径 $\varepsilon_N$ 作为分布鲁棒性修正项，当样本量 $N\to\infty$ 时 $\varepsilon_N\to 0$，CW-DRCC 退化为经典 CQR。

### F.5 通信可靠性硬约束

令 $T^{\mathrm{rel}}_b=\lceil F_b^{-1}(R^{\min}_b)\rceil$ 为满足最低通信可靠性所需的 BESS autonomy 小时数。**通信可靠性约束在整个调度窗口上以硬约束形式嵌入**：对任意 $t\in\mathcal{T}$ 且 $t+T^{\mathrm{rel}}_b\le H$
$$
\boxed{\;\left(\mathrm{SOC}_{b,t}-\underline{\mathrm{SOC}}\right)C_b\ge \widehat{R}^{U,\mathrm{CW}}_{b,t}(T^{\mathrm{rel}}_b)\;}
\tag{26}
$$
该约束等价于：任意时刻 $t$ 的 SOC 都要足以在 CW-DRCC 分布鲁棒意义下支撑 $T^{\mathrm{rel}}_b$ 小时的随机修复过程。这是 Phase D 通信可靠性思想在多时段、多基站联合调度下的自然推广。

## G. 储能衰减成本的 DOD 分段线性近似

### G.1 循环次数-DOD 幂律关系

锂电池实验表明，单次循环的寿命衰减率与该循环深度 $d\in(0,1]$ 大致呈幂律
$$
N_{\mathrm{cyc}}(d)=a\cdot d^{-k},\quad k\in[1.5,2.5]
\tag{27}
$$
对应单位吞吐能量的衰减成本
$$
c^{\mathrm{wear}}(d)=\frac{C^{\mathrm{bat}}}{2\cdot N_{\mathrm{cyc}}(d)\cdot C_b}\propto d^{k}
\tag{28}
$$
即深度循环越深，单位能量衰减成本越高。

### G.2 分段线性化

将 SOC 的可行区间 $[\underline{\mathrm{SOC}},\overline{\mathrm{SOC}}]$ 等距划分为 $K$ 段，第 $k$ 段中心 SOC 为 $\mathrm{SOC}^c_k$，对应 DOD 中心 $d_k=1-\mathrm{SOC}^c_k$，赋单位衰减成本 $c^{\mathrm{wear}}_k=c^{\mathrm{wear}}(d_k)$。引入辅助变量 $P^{\mathrm{ch},k}_{b,t},P^{\mathrm{dis},k}_{b,t}\ge 0$ 和 0-1 指示 $w^{k}_{b,t}$，满足
$$
\sum_{k=1}^{K}w^{k}_{b,t}=1,\quad \mathrm{SOC}_{b,t}\in[\mathrm{SOC}^c_k-\Delta/2,\mathrm{SOC}^c_k+\Delta/2]\Rightarrow w^{k}_{b,t}=1
\tag{29}
$$
$$
P^{\mathrm{ch}}_{b,t}=\sum_{k}P^{\mathrm{ch},k}_{b,t},\quad P^{\mathrm{dis}}_{b,t}=\sum_{k}P^{\mathrm{dis},k}_{b,t}
\tag{30}
$$
$$
P^{\mathrm{ch},k}_{b,t}\le P^{\mathrm{rate}}_b w^{k}_{b,t},\quad P^{\mathrm{dis},k}_{b,t}\le P^{\mathrm{rate}}_b w^{k}_{b,t}
\tag{31}
$$
衰减成本线性聚合为
$$
\mathrm{DegCost}=\sum_{b,t}\sum_{k}c^{\mathrm{wear}}_k\left(P^{\mathrm{ch},k}_{b,t}+P^{\mathrm{dis},k}_{b,t}\right)\Delta t
\tag{32}
$$
实际实现中为了控制 MILP 规模，本文建议 $K=4$（轻循环/中循环/深循环/极深循环）。

### G.3 事后雨流验证

上述 DOD 分段是路径无关近似，严格的循环衰减需通过雨流计数核算。本文实验流程为：

1. 求解 MISOCP 得到 SOC 轨迹 $\{\mathrm{SOC}^\star_{b,t}\}$；
2. 对每条 SOC 轨迹应用 ASTM E1049-85 标准雨流算法，统计半循环 $\{(\Delta d_m,N^{\mathrm{half}}_m)\}$；
3. 按式 (27) 累加真衰减成本 $C^{\mathrm{deg,rf}}_b=\sum_m \frac{C^{\mathrm{bat}}}{2 N_{\mathrm{cyc}}(\Delta d_m)}N^{\mathrm{half}}_m$；
4. 如 $|C^{\mathrm{deg,rf}}-\mathrm{DegCost}|/C^{\mathrm{deg,rf}}>\tau_{\mathrm{tol}}$，调整 $c^{\mathrm{wear}}_k$ 并重解一次。

## H. 目标函数

在调度窗口内最小化购电成本与衰减成本之和：
$$
\boxed{\;\min\;\sum_{t\in\mathcal{T}}\pi^{\mathrm{buy}}_t P^{\mathrm{grid}}_{0,t}\Delta t+\sum_{b\in\mathcal{B}}\sum_{t\in\mathcal{T}}\sum_{k=1}^{K}c^{\mathrm{wear}}_k\left(P^{\mathrm{ch},k}_{b,t}+P^{\mathrm{dis},k}_{b,t}\right)\Delta t\;}
\tag{33}
$$
分时电价 $\pi^{\mathrm{buy}}_t$ 按峰-平-谷-尖峰四段划分，示例参数（华东地区典型）：
- 尖峰（19-21 时）：1.35 元/kWh
- 高峰（8-11, 18-19, 21-22 时）：1.08 元/kWh
- 平段（7-8, 11-18, 22-23 时）：0.72 元/kWh
- 低谷（23-次日 7 时）：0.35 元/kWh

## I. 完整 MISOCP 模型

$$
\begin{aligned}
\min\quad&\sum_{t}\pi^{\mathrm{buy}}_t P^{\mathrm{grid}}_{0,t}\Delta t+\sum_{b,t,k}c^{\mathrm{wear}}_k(P^{\mathrm{ch},k}_{b,t}+P^{\mathrm{dis},k}_{b,t})\Delta t\\
\mathrm{s.t.}\quad&(5)-(16)\quad\text{BESS 与配电网约束}\\
&(26)\quad\text{CW-DRCC 通信可靠性}\\
&(29)-(31)\quad\text{DOD 分段线性化}\\
&(1)-(4)\quad\text{容量设计与 SOC 边界}
\end{aligned}
\tag{34}
$$

**变量规模**（$|\mathcal{B}|=32, H=24, K=4$）：
- 连续变量：约 $32\times 24\times(8+4)+32\times 24+(32+1)\times 24\times 4\approx 1.4\times 10^4$
- 二进制变量：$32\times 24\times(2+4)\approx 4.6\times 10^3$
- SOCP 锥约束：$32\times 24\approx 768$ 个二阶锥

Gurobi 在 Intel i7 典型配置下求解时间 30--120 秒；若启用 Mosek 可进一步加速。

## J. 对比实验设计

为彰显 CW-DRCC 的增益，本文设计五档对比策略：

| 编号 | 策略 | 能耗约束 | 预期表现 |
|---|---|---|---|
| S1 | **No BESS** | $\widetilde{E}=\widehat{E}^C$，$P^{\mathrm{ch}}=P^{\mathrm{dis}}=0$ | 购电成本最高，无衰减成本 |
| S2 | **Backup Only** | BESS 满充作后备，不参与调度 | 衰减接近 0，无经济收益 |
| S3 | **Deterministic** | $\widetilde{E}=\widehat{E}^C$，忽略不确定性 | 成本最低但可靠性违反率高 |
| S4 | **CQR-Robust** | $\widetilde{E}=\widehat{U}_{90}$（Phase C 上界） | 可靠性高但过度保守 |
| S5 | **CW-DRCC**（本文） | 式 (26) | 经济性与可靠性帕累托最优 |

**评估指标**：
- 日购电成本（元）
- 日衰减成本（DOD 分段近似 + 真雨流复核两者）
- 平均节点电压偏离 $(\max_j|v_j-1|)$
- 可靠性违反率（蒙特卡洛模拟 1000 次停电事件）
- 平均可支撑时间 $\bar{\tau}$
- 期望通信中断时长 ECID

## K. 模型性质与求解复杂度

### K.1 凸性

去除 DOD 分段的 0-1 变量后，模型为凸 SOCP（潮流松弛紧时即为原问题最优）。DOD 分段引入的 $K\cdot|\mathcal{B}|\cdot H$ 个 0-1 变量使整体成为 MISOCP，但每个基站的 $w^k$ 在任意时刻满足 SOS1 约束，分支定界效率较高。

### K.2 CW-DRCC 嵌入的可解性

由于 Wasserstein 模糊集的对偶理论给出线性 + SOCP 形式的等价约束，CW-DRCC 不破坏整体 MISOCP 结构。相较于机会约束的大 M 线性化或样本平均近似，CW-DRCC 避免了 $O(N)$ 量级的 0-1 变量膨胀。

### K.3 滚动时域扩展

若实际部署需要实时 MPC，可将调度窗口滚动为 $[t,t+H']$，每小时求解一次并根据实时观测更新 $\widehat{\mathbb{P}}_N$ 和 $\varepsilon_N$（在线 conformal 如 ACI, Gibbs-Candès 2021 给出自适应半径）。本文暂以日前 24 h 单次调度为基线。

## L. 数据来源与参数取值

| 参数 | 取值 | 来源 |
|---|---|---|
| IEEE-33 阻抗 $(r_{ij},x_{ij})$ | 标准测试数据 | Baran-Wu 1989 |
| 电压限值 $[\underline{V},\overline{V}]$ | $[0.95,1.05]$ p.u. | 配电网导则 |
| 基站能耗 $\widehat{E}^C_{b,t}$ | Phase C 输出 `pg_rnp_predictions.csv` | Phase A--C |
| 校准残差 $s^{(T)}_{b,t,i}$ | Phase C 累计误差 | Phase D § C.5 |
| 功率因数 $\cos\varphi$ | 0.95 | 典型基站设备 |
| 充放电效率 $\eta^{\mathrm{ch}},\eta^{\mathrm{dis}}$ | 0.95, 0.95 | 锂电池典型值 |
| SOC 范围 | $[0.1,0.9]$ | 延长寿命设置 |
| 修复分布 | Exponential($\mu=0.7$) | Phase D 默认 |
| 可靠性目标 $R^{\min}$ | 0.99 | 通信 SLA |
| 不确定性风险 $\alpha$ | 0.10 | Phase C CQR 参数 |
| 电池单位成本 $C^{\mathrm{bat}}$ | 1200 元/kWh | 2025 市场价 |
| DOD 幂律指数 $k$ | 2.0 | Wang et al. 2014 |
| 分时电价 | § H 示例 | 华东电网 |

## M. 与 Phase D 的衔接与增量

| 维度 | Phase D | Phase E |
|---|---|---|
| 决策对象 | 单基站可调度容量 $C^{\mathrm{disp}}_{b,t}$ | 多基站联合 $\{P^{\mathrm{ch}},P^{\mathrm{dis}}\}$ |
| 可靠性约束 | 单时刻闭式 $(33)$ | 多时段硬约束 $(26)$ |
| 经济性目标 | 容量收益 + 惩罚 $(43)$ | 购电 + 衰减 $(33)$ |
| 电网耦合 | 无 | DistFlow SOCP |
| 不确定性 | Aggregate-CQR 点分位 | **CW-DRCC 分布鲁棒** |
| 模型类别 | 闭式解 / LP | MISOCP |
| 创新点 | 通信可靠性-autonomy 映射 | 配电网联合调度 + CW-DRCC |

Phase E 继承 Phase D 的通信可靠性约束结构，但在 CW-DRCC 框架下把 Phase C 的 conformal 分数与 Wasserstein 分布鲁棒优化结合，给出有限样本可靠性保证下的经济最优调度，实现从"单点可靠性"到"系统级鲁棒调度"的跃迁。
