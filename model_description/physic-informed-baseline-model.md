Viewed 03_physics_baseline.py:1-407

***

## II. 物理知识驱动的5G基站能耗基准建模 (Physics-Informed Baseline Modeling of 5G Base Stations)

5G基站的总能耗受硬件配置差异、业务负载高频波动以及多种网络节能策略的非线性耦合影响。为准确刻画基站的能量流转机制，避免纯数据驱动模型在样本外场景出现的物理失真，本文首先构建具备严格物理机理约束的基站级（Base Station, BS）基准能耗模型。总能耗 $E_{\text{BS}}$ 被严谨地解耦为静态基础功耗（Static Power）、动态传输功耗（Dynamic Power）及休眠节能功耗削减（Energy Saving Reduction）三部分，其基本物理等式表达如下：

$$
E_{\text{BS}} = P_{\text{static}} + P_{\text{dynamic}} - P_{\text{save}} + \varepsilon \tag{1}
$$

其中，$P_{\text{static}}$ 表征不随网络负载波动的BS运行能耗（如制冷、漏电流等）；$P_{\text{dynamic}}$ 表征随无线资源利用率（Physical Resource Block, PRB）动态波动的射频发射能耗；$P_{\text{save}}$ 表征激活特定节能模式（如通道关断、符号关断等）带来的能耗削减项；$\varepsilon$ 表示未覆盖的微小非线性扰动。

### A. 物理代理特征与机制构造 (Construction of Physics-Informed Proxy Features)

由于实际网络调度是以基站为整体单位，而基础监测数据往往细化至扇区小区级（Cell-level），本模型首先基于各小区 $i (i=1, 2, ... , N_c)$ 的原始参数，构建表征基站整体能量特征的物理代理变量（Proxy Variables）：

**1) 静态基站特征项 (Static Attributes):**
硬件的绝对规模决定了静态功耗的上限。本文构建了三重维度的静态物理代理项：
* $X_{\text{Pmax}}^{\text{tot}} = \sum_{i=1}^{N_c} p_{\text{max}, i}$ 为基站内全部小区的最大额定发射功率总和；
* $X_{\text{ant}}^{\text{tot}} = \sum_{i=1}^{N_c} ant_i$ 为基站域内总天线阵子数；
* $X_{\text{cell}}^{\text{tot}} = N_c$ 为挂载的小区总数量。

基站层面的总静态能耗模型表达为一阶多元线性映射：
$$
\hat{P}_{\text{static}} = \alpha_0 + \alpha_1 X_{\text{Pmax}}^{\text{tot}} + \alpha_2 X_{\text{ant}}^{\text{tot}} + \alpha_3 X_{\text{cell}}^{\text{tot}} \tag{2}
$$
根据物理守恒定律，硬件参数越加码，静态功耗必然增加或维持。因此，须对回归系数施加强非负性物理边界约束：$\alpha_j \ge 0, \forall j \in \{1,2,3\}$（$\alpha_0$ 视作基底维持功耗截距）。

**2) 动态载荷代理项 (Dynamic Load Proxy):**
空口动态能耗对无线资源的利用率 $\rho_i$ 具有高度敏感性，并且需要最大发射功率与天线数量作为加权底座。由此定义基站域内动态负载等效特征为：
$$
X_{\text{dyn}} = \sum_{i=1}^{N_c} \left( \rho_i \times p_{\text{max}, i} \times ant_i \right) \tag{3}
$$
此外，由于大规模MIMO系统中射频功放（Power Amplifier, PA）在非满载工作区带往往存在效率的非线性折回散逸现象（Efficiency Roll-off），故引入二阶项来包容硬件非线性行为：
$$
\hat{P}_{\text{dynamic}} = \beta_1 X_{\text{dyn}} + \beta_2 X_{\text{dyn}}^2 \tag{4}
$$
同理约束载荷增益系数非负：$\beta_1, \beta_2 \ge 0$。

**3) 节能代理项 (Energy-Saving Proxy):**
设基站包含 $K$ 种相异的网络节能演进模式（$K \le 6$）。激活某一模式所节省的绝对能量受该小区原始辐射基盘的加权倍乘效应影响：
$$
X_{\text{ES}}^{(k)} = \sum_{i=1}^{N_c} \left( I_{k, i} \times p_{\text{max}, i} \times ant_i \right) \tag{5}
$$
式中，$I_{k, i} \in \{0, 1\}$ 为布尔激活算子（指示第 $i$ 小区处于第 $k$ 种睡眠模式状态与否）。基站多维节能总削减模型刻画为：
$$
\hat{P}_{\text{save}} = \sum_{k=1}^K \gamma_k X_{\text{ES}}^{(k)} \tag{6}
$$
能耗削减因子需满足降耗事实约束：$\gamma_k \ge 0$。

汇总以上特征公式，构建带约束的多元基准物理方程为：
$$
\hat{P}_{\text{base}}({\theta}) = \hat{P}_{\text{static}}({\alpha}) + \hat{P}_{\text{dynamic}}({\beta}) - \hat{P}_{\text{save}}({\gamma}) \tag{7}
$$
定义全局待辨识参数向量集 ${\theta} = \{ {\alpha}, {\beta}, {\gamma} \}$。

### B. 基于先验的两步约束优化算法 (Two-Step Constrained Optimization for Parameter Identification)

考虑到各物理代理特征（如负载与静态能耗基底）存在时序维度的共线性交互，直接运用全局最小二乘估计极易触发伪相关，导致辨识系数违反前置非负条件。为此，本文设计了基于低负载边界先验的两步边界约束优化（Two-Step Constrained Optimization）求解算法。

**步骤1：基于极低负载子集的静态参数辨识 (Identification of Static Parameters based on Low-Load Subsets)**
在网元极度空闲波谷期（例如深夜时段），动态射频信号收发停止，基站测得的总功耗逼近纯静态运行功耗。引入样本分位数截断：筛选出满足准则 $X_{\text{dyn}} \le Q_{0.20}(X_{\text{dyn}})$ 的时间切片构成低负载子集 $\mathcal{S}_{\text{low}}$。利用附加大拉格朗日L2惩罚的限制性岭回归（Ridge Regression with Non-Negativity）进行最优辨识：
$$
\hat{{\alpha}} = \arg \min_{{\alpha}} \sum_{t \in \mathcal{S}_{\text{low}}} \left( E_{t} - \hat{P}_{\text{static},t}({\alpha}) \right)^2 + \lambda \|{\alpha}_{0}\|_2^2 \tag{8}
$$
$$
s.t. \quad \alpha_j \ge 0, \quad \forall j
$$
式中 $\lambda=10.0$ 为岭抑制系数，籍此固定静态主导参数 $\hat{{\alpha}}$。

**步骤2：全样本广延动态与节能参数联立估计 (Simultaneous Estimation for Dynamic and ES Parameters)**
固定首步辨识求得的硬件底座参量，计算涵盖日间全时段（样本集 $\mathcal{S}_{\text{all}}$）剥离固有静态负荷后的功耗响应残差：
$$
\Delta E_{t} = E_{t} - \hat{P}_{\text{static},t}(\hat{{\alpha}}) \tag{9}
$$
构建非线性凸优化目标，采用处理多变量边界限制问题的有限内存拟牛顿迭代法（L-BFGS-B Algorithm），联立求解发信负载与基站睡态的权重增益：
$$
\hat{{\beta}}, \hat{{\gamma}} = \arg \min_{{\beta}, {\gamma}} \frac{1}{|\mathcal{S}_{\text{all}}|} \sum_{t \in \mathcal{S}_{\text{all}}} \left( \Delta E_t - \big(\hat{P}_{\text{dynamic},t}({\beta}) - \hat{P}_{\text{save},t}({\gamma})\big) \right)^2 \tag{10}
$$
$$
s.t. \quad \beta_1, \beta_2 \ge 0; \quad \gamma_k \ge 0 \,\, (\forall k \in K)
$$
由于推导出的基准预测结果具有明确的量纲机理保障，提取其逼近残差 $\varepsilon_t = E_t - \hat{P}_{\text{base},t}$ 输入至下游物理指导的注意力神经过程（Physics-Informed Attentive Neural Process, PI-ANP），将促使深度网络模型解耦出对复杂非线性环境状态的高度灵敏捕捉，而非在基础物理定律上过度透支泛化能力。

***

