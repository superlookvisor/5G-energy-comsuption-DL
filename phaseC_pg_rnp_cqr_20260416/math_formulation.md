# Phase C 数学建模：物理引导残差神经过程与分时距保形区间预测（PG-RNP-CQR）

> 本节以 IEEE Transactions on Smart Grid（TSG）体例，正式描述 Phase C 所实现的“物理基线 + 残差神经过程 + 分时距 CQR 校准 + 物理可行域投影”日前能耗预测框架，并给出其完整数学形式。行文中，所有公式均以 LaTeX 内联语法 `$...$` 给出。

## 一、问题定义与记号

令 $\mathcal{B}$ 为 5G 基站集合，$b\in\mathcal{B}$ 表示基站索引；$t_0\in\mathcal{T}_0$ 为日前预测起点，本文中固定为每日 00:00；$h\in\mathcal{H}=\{1,2,\ldots,H\}$ 为预测时距（horizon），$H=24$。记 $E_{b,t_0+h}\in\mathbb{R}_{\ge 0}$ 为基站 $b$ 在目标时刻 $t_0+h$ 观测到的站级总能耗。每个 $(b,t_0)$ 对应一条长度为 $H$ 的“日前轨迹”，记为 $\mathcal{E}_{b,t_0}=\{E_{b,t_0+h}\}_{h=1}^{H}$。

由 Phase A 给出的静态基功率估计记为 $p^{\text{base}}_b\in\mathbb{R}_{>0}$；由 Phase B 的 `two_stage_proxy + Physical` 基线给出的动态能耗物理预测记为 $\widehat{y}^{\text{phys}}_{b,t_0+h}$。据此定义物理基线总能耗预测

$$
\widehat{E}^{\text{phys}}_{b,t_0+h} = p^{\text{base}}_b + \widehat{y}^{\text{phys}}_{b,t_0+h},\tag{1}
$$

以及物理基线残差

$$
r_{b,t_0+h} \;=\; E_{b,t_0+h} - \widehat{E}^{\text{phys}}_{b,t_0+h}.\tag{2}
$$

Phase C 的学习目标，是在已知 $\widehat{E}^{\text{phys}}_{b,t_0+h}$ 的前提下，对残差 $r_{b,t_0+h}$ 进行**概率预测**，并最终给出具有物理可行性保证的**点预测**与**区间预测**：

$$
\widehat{E}_{b,t_0+h} \;=\; \widehat{E}^{\text{phys}}_{b,t_0+h} \;+\; \widehat{\mu}_{b,t_0+h},\qquad \text{(point forecast)}\tag{3}
$$

$$
\widehat{\mathcal{I}}_{b,t_0+h}^{\,1-\alpha} \;=\; \bigl[\,\widehat{L}_{b,t_0+h},\,\widehat{U}_{b,t_0+h}\,\bigr]\subseteq\bigl[\,L^{\text{phys}}_{b,t_0+h},\,U^{\text{phys}}_{b,t_0+h}\,\bigr],\qquad \text{(prediction interval)}\tag{4}
$$

其中 $\widehat{\mu}_{b,t_0+h}$ 为残差的后验均值估计，$[L^{\text{phys}},U^{\text{phys}}]$ 为物理可行域，详见第四节；$\alpha\in(0,1)$ 为预设显著性水平，本文取 $\alpha=0.10$。

## 二、数据生成过程与特征构造

### A. 面板对齐

基于 Phase B 的输出面板 `panel_dataset.csv` 与日前预测 `dayahead_predictions.csv`，在 `(BS, trajectory_id, origin_time, target_time, horizon)` 键上对齐，得到 Phase C 训练样本集

$$
\mathcal{D}=\bigl\{\,(b_i,t_{0,i},h_i,\mathbf{x}_i,\widehat{E}^{\text{phys}}_i,E_i,r_i)\,\bigr\}_{i=1}^{N},\tag{5}
$$

其中 $\mathbf{x}_i\in\mathbb{R}^{d}$ 为 Phase C 特征向量，$r_i=E_i-\widehat{E}^{\text{phys}}_i$ 为公式 (2) 给出的残差真值。

### B. 特征向量 $\mathbf{x}_{b,t_0+h}$

特征向量由以下五类子向量级联而成：

1) **时间周期编码**

$$
\mathbf{x}^{\text{time}}_{b,t_0+h}=\bigl[\sin(2\pi h^{\,*}/24),\,\cos(2\pi h^{\,*}/24),\,\sin(2\pi d^{\,*}/7),\,\cos(2\pi d^{\,*}/7)\bigr]^{\top},\tag{6}
$$

其中 $h^{\,*}$ 与 $d^{\,*}$ 分别为目标时刻的小时与星期几。

2) **物理基线相关**：$p^{\text{base}}_b$、$\widehat{y}^{\text{phys}}_{b,t_0+h}$、$\widehat{E}^{\text{phys}}_{b,t_0+h}$。

3) **两阶段代理（Phase B）**：日前负载代理 $\widehat{\ell}^{\,\text{mean}}_{b,t_0+h},\widehat{\ell}^{\,\text{pmax}}_{b,t_0+h},\widehat{\ell}^{\,\text{std}}_{b,t_0+h}$ 及其物理变换

$$
\widehat{D}^{(1)}=sp_b\,\widehat{\ell}^{\,\text{pmax}},\quad
\widehat{D}^{(2)}=sp_b\,(\widehat{\ell}^{\,\text{pmax}})^2,\quad
\widehat{D}^{(3)}=\widehat{\ell}^{\,\text{std}},\tag{7}
$$

其中 $sp_b=\sum_{c\in b}P^{\max}_{b,c}$ 为基站 $b$ 的额定最大发射功率之和；并令节能模式代理 $\widehat{S}^{(m)}_{b,t_0+h}$ 及其交互项 $\widehat{I}^{(m)}=\widehat{S}^{(m)}\cdot \widehat{\ell}^{\,\text{pmax}}$，$m=1,\ldots,6$。

4) **滚动统计**：$\widehat{\ell}^{\,\text{mean}}_{\text{roll24}},\,\widehat{\ell}^{\,\text{pmax}}_{\text{roll24}},\,\widehat{\ell}^{\,\text{std}}_{\text{roll24}}$。

5) **静态硬件与配置比例**：$n^{\text{cells}}_b$、$sp_b$、$\mathrm{sum\_bandwidth}_b$、$\mathrm{sum\_antennas}_b$、$\overline{f}_b$、$\rho^{\text{high}}_b$，以及调制模式比例 $\{\mathrm{mode\_ratio}_{b,\cdot}\}$ 与 RU 类型比例 $\{\mathrm{ru\_ratio}_{b,\cdot}\}$。

将上述各子向量级联，即得到 $\mathbf{x}_{b,t_0+h}\in\mathbb{R}^{d}$，$d\in\mathbb{N}$。

### C. 防泄漏约束

为避免目标信息进入特征空间，Phase C 强制

$$
\{E,\,E^{\text{true}},\,y,\,y^{\text{true}},\,r^{\text{true}}\}\cap\{\text{feature columns}\}=\varnothing,\tag{8}
$$

并在面板级别校验

$$
\text{target\_time}-\text{origin\_time}\equiv h,\qquad
E - \widehat{E}^{\text{phys}} \equiv r^{\text{true}}.\tag{9}
$$

### D. 标准化

记训练集为 $\mathcal{D}_{\text{tr}}$。定义特征均值—标准差标准化器 $\mathcal{S}_{x}(\cdot)$：

$$
\tilde{\mathbf{x}}=\mathcal{S}_{x}(\mathbf{x})=\Sigma_x^{-1/2}(\mathbf{x}-\boldsymbol{\mu}_x),\quad \boldsymbol{\mu}_x,\Sigma_x \text{ 由 } \mathcal{D}_{\text{tr}} \text{ 估计};\tag{10}
$$

定义残差标准化

$$
\tilde{r}=\frac{r-\mu_r}{\sigma_r},\quad \mu_r=\mathbb{E}_{\mathcal{D}_{\text{tr}}}[r],\quad \sigma_r=\sqrt{\mathrm{Var}_{\mathcal{D}_{\text{tr}}}(r)}.\tag{11}
$$

## 三、物理引导残差神经过程（PG-RNP）

### A. Episode 组织与缺失掩码

以 $(b,t_0)$ 为单位组织 episode，每条 episode 的目标序列长度为 $H$，记为 $\mathbf{X}_{b,t_0}\in\mathbb{R}^{H\times d}$，对应标签 $\mathbf{r}_{b,t_0}\in\mathbb{R}^{H}$。由于部分 $(b,t_0,h)$ 样本缺失，引入掩码向量 $\mathbf{m}_{b,t_0}\in\{0,1\}^{H}$，以仅在可用时距上参与训练与评估。

对每条 episode，定义其上下文集合

$$
\mathcal{C}_{b,t_0}=\bigl\{(\mathbf{x}_{b,\tau+h'},r_{b,\tau+h'})\,\big|\,\tau+h'<t_0\bigr\},\tag{12}
$$

即同一基站 $b$ 中发生于 $t_0$ 之前的历史行。训练时从 $\mathcal{C}_{b,t_0}$ 中随机抽取规模不超过 $n_{\max}$ 的子集作为实际输入上下文 $\mathcal{C}^{\,*}_{b,t_0}=\{(\mathbf{x}^{c}_j,r^{c}_j)\}_{j=1}^{n}$，推理时则取时序最近的 $n_{\max}$ 条以减少方差。

### B. 上下文汇聚

设上下文编码器 $\phi_{c}:\mathbb{R}^{d+1}\to\mathbb{R}^{d_h}$ 为多层感知机（MLP），定义上下文摘要

$$
\mathbf{c}_{b,t_0}=\frac{\sum_{j=1}^{n}\phi_c\bigl([\tilde{\mathbf{x}}^{c}_j;\tilde{r}^{c}_j]\bigr)}{\max(n,1)}\in\mathbb{R}^{d_h}.\tag{13}
$$

### C. 潜变量与摊销推断

Phase C 采用隐变量神经过程的摊销变分推断范式。引入潜变量 $\mathbf{z}_{b,t_0}\in\mathbb{R}^{d_z}$，其变分后验为对角高斯

$$
q_\phi(\mathbf{z}_{b,t_0}\mid \mathcal{C}^{\,*}_{b,t_0})=\mathcal{N}\bigl(\mathbf{z};\,\boldsymbol{\mu}_z(\mathbf{c}_{b,t_0}),\,\mathrm{diag}\,\boldsymbol{\sigma}_z^2(\mathbf{c}_{b,t_0})\bigr),\tag{14}
$$

其中 $\boldsymbol{\mu}_z(\cdot)$、$\log\boldsymbol{\sigma}_z^2(\cdot)$ 均为 $\mathbf{c}_{b,t_0}$ 的线性函数；为数值稳定，对 $\log\boldsymbol{\sigma}_z^2$ 施加截断 $[-8,5]$。先验取标准正态 $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I}_{d_z})$。训练阶段使用重参数化采样

$$
\mathbf{z}=\boldsymbol{\mu}_z+\boldsymbol{\sigma}_z\odot\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{d_z}),\tag{15}
$$

推理阶段取 $\mathbf{z}=\boldsymbol{\mu}_z$。

### D. 时序解码与残差分布

目标序列的每个槽位 $h\in\mathcal{H}$ 首先经目标编码器 $\phi_t$ 融合上下文与潜变量：

$$
\mathbf{u}^{(h)}_{b,t_0}=\phi_t\bigl([\tilde{\mathbf{x}}^{(h)}_{b,t_0};\,\mathbf{c}_{b,t_0};\,\mathbf{z}_{b,t_0}]\bigr)\in\mathbb{R}^{d_h}.\tag{16}
$$

记 $\mathbf{U}_{b,t_0}=[\mathbf{u}^{(1)}_{b,t_0},\ldots,\mathbf{u}^{(H)}_{b,t_0}]^\top\in\mathbb{R}^{H\times d_h}$。在 $H$ 个槽位上应用多头自注意力 Transformer 编码器 $\psi_{\text{tmp}}$，得到时序表征

$$
\mathbf{H}_{b,t_0}=\psi_{\text{tmp}}(\mathbf{U}_{b,t_0})\in\mathbb{R}^{H\times d_h}.\tag{17}
$$

为每个槽位并行输出残差分布参数

$$
\widetilde{\mu}^{(h)}=w_\mu^\top\mathbf{H}_{b,t_0}^{(h)}+b_\mu,\tag{18}
$$

$$
\widetilde{\sigma}^{(h)}=\mathrm{softplus}\bigl(w_\sigma^\top\mathbf{H}_{b,t_0}^{(h)}+b_\sigma\bigr)+\varepsilon_\sigma,\tag{19}
$$

其中 $\varepsilon_\sigma=10^{-4}$ 为下界保护。注意 $\widetilde{\mu}^{(h)}$、$\widetilde{\sigma}^{(h)}$ 定义在**标准化残差空间**中。

### E. 似然选择

本文考虑两种可选观测似然，默认采用具有重尾鲁棒性的 Student-$t$ 分布：

$$
p_\theta(\tilde{r}^{(h)}\mid \mathcal{C},\mathbf{x}^{(h)},\mathbf{z})=
\begin{cases}
\mathcal{N}\bigl(\widetilde{\mu}^{(h)},\,(\widetilde{\sigma}^{(h)})^2\bigr), & \text{likelihood}=\text{Gaussian},\\[4pt]
\mathrm{StudentT}\bigl(\nu;\,\widetilde{\mu}^{(h)},\,\widetilde{\sigma}^{(h)}\bigr), & \text{likelihood}=\text{Student-}t,
\end{cases}\tag{20}
$$

其中自由度 $\nu=\mathrm{softplus}(\nu_0)+2$ 为可学习参数，确保 $\nu>2$ 以保证方差有限。

### F. 反标定与点预测

由 (11) 可得标准化残差到原始能耗空间的反变换：

$$
\widehat{\mu}^{(h)}_{b,t_0+h}=\widetilde{\mu}^{(h)}\,\sigma_r+\mu_r,\qquad
\widehat{\sigma}^{(h)}_{b,t_0+h}=\widetilde{\sigma}^{(h)}\,\sigma_r,\tag{21}
$$

$$
\widehat{E}_{b,t_0+h}=\widehat{E}^{\text{phys}}_{b,t_0+h}+\widehat{\mu}^{(h)}_{b,t_0+h}.\tag{22}
$$

## 四、物理可行域

为保证区间预测的工程可解释性，Phase C 对每条样本引入物理可行域 $[L^{\text{phys}}_{b,t_0+h},U^{\text{phys}}_{b,t_0+h}]$：

$$
L^{\text{phys}}_{b,t_0+h}=0,\qquad
U^{\text{phys}}_{b,t_0+h}=\max\!\Bigl\{p^{\text{base}}_b+2\,sp_b+10,\;\widehat{E}^{\text{phys}}_{b,t_0+h}+1\Bigr\}.\tag{23}
$$

上界的构造融合了“静态基功率 + 若干倍额定发射功率 + 常量安全裕度”与“始终不小于物理基线”两项约束，确保可行域对物理基线本身保持包含关系。

## 五、训练目标与正则化

### A. 负对数似然

对单条 episode 的 NLL 项定义为

$$
\mathcal{L}_{\text{NLL}}(\theta)=-\frac{1}{\sum_h m^{(h)}}\sum_{h=1}^{H}m^{(h)}\,\log p_\theta\bigl(\tilde{r}^{(h)}\bigm|\mathcal{C},\mathbf{x}^{(h)},\mathbf{z}\bigr).\tag{24}
$$

### B. KL 正则

潜变量后验向先验收敛：

$$
\mathcal{L}_{\text{KL}}(\phi)=\mathrm{KL}\!\bigl(q_\phi(\mathbf{z}\mid\mathcal{C})\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\bigr)
=-\tfrac{1}{2}\mathbb{E}\!\left[1+\log\boldsymbol{\sigma}_z^{2}-\boldsymbol{\mu}_z^{2}-\boldsymbol{\sigma}_z^{2}\right].\tag{25}
$$

### C. 物理可行域罚项

将残差在物理空间内的点预测写为 $\widehat{\mu}^{(h)}_{\text{raw}}=\widetilde{\mu}^{(h)}\sigma_r+\mu_r$，对应总能耗 $\widehat{E}^{(h)}=\widehat{E}^{\text{phys},(h)}+\widehat{\mu}^{(h)}_{\text{raw}}$。定义越界二次罚项

$$
\mathcal{L}_{\text{bnd}}(\theta)=\frac{1}{\sigma_r^{2}+\varepsilon}\cdot
\frac{\sum_{h=1}^{H}m^{(h)}\Bigl[\bigl(L^{\text{phys},(h)}-\widehat{E}^{(h)}\bigr)_{+}^{2}+\bigl(\widehat{E}^{(h)}-U^{\text{phys},(h)}\bigr)_{+}^{2}\Bigr]}{\sum_h m^{(h)}},\tag{26}
$$

其中 $(x)_{+}=\max(x,0)$；$\sigma_r^{2}$ 归一化使该项与 NLL 位于相近数量级。

### D. 时间平滑罚项

对相邻时距施加二阶差分罚项，以抑制学习到的残差轨迹的高频震荡：

$$
\mathcal{L}_{\text{smo}}(\theta)=\frac{1}{\sigma_r^{2}+\varepsilon}\cdot
\frac{\sum_{h=2}^{H}\widetilde{m}^{(h)}\bigl(\widehat{\mu}^{(h)}_{\text{raw}}-\widehat{\mu}^{(h-1)}_{\text{raw}}\bigr)^{2}}{\sum_{h=2}^{H}\widetilde{m}^{(h)}},\quad \widetilde{m}^{(h)}=m^{(h)}m^{(h-1)}.\tag{27}
$$

### E. 总损失

综合 (24)–(27)，Phase C 的训练损失为

$$
\boxed{\;\mathcal{L}(\theta,\phi)=\mathcal{L}_{\text{NLL}}+\lambda_{\text{KL}}\mathcal{L}_{\text{KL}}+\lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}}+\lambda_{\text{smo}}\mathcal{L}_{\text{smo}}\;}\tag{28}
$$

其中 $\lambda_{\text{KL}},\lambda_{\text{bnd}},\lambda_{\text{smo}}\ge 0$ 为超参数，默认取 $(0.02,\,0.02,\,0.005)$。参数采用 AdamW 优化器、梯度范数裁剪 $\|\nabla\|_2\le 5$ 以及基于校准集 NLL 的早停策略。

### F. 站点级划分

为避免同一基站信息跨集泄漏，Phase C 在**基站粒度**上做不重叠划分：

$$
\mathcal{B}=\mathcal{B}_{\text{tr}}\sqcup\mathcal{B}_{\text{cal}}\sqcup\mathcal{B}_{\text{te}},\quad
|\mathcal{B}_{\text{tr}}|=\lfloor\rho_{\text{tr}}|\mathcal{B}|\rfloor,\ |\mathcal{B}_{\text{cal}}|=\lfloor\rho_{\text{cal}}|\mathcal{B}|\rfloor,\tag{29}
$$

默认 $(\rho_{\text{tr}},\rho_{\text{cal}})=(0.70,\,0.15)$，其余作为测试集 $\mathcal{B}_{\text{te}}$。

## 六、分时距保形分位数校准（Horizon-wise CQR）

### A. 非一致性分数

在校准集 $\mathcal{D}_{\text{cal}}=\{(i)\mid b_i\in\mathcal{B}_{\text{cal}}\}$ 上，按时距 $h$ 分别构造标准化绝对残差分数

$$
s^{(h)}_i=\frac{\bigl|r_i-\widehat{\mu}^{(h)}_i\bigr|}{\max\!\bigl(\widehat{\sigma}^{(h)}_i,\,\epsilon\bigr)},\quad i\in\mathcal{D}_{\text{cal}},\ h_i=h,\tag{30}
$$

其中 $\epsilon>0$ 为下界保护。

### B. 按 horizon 的分位数阈值

给定显著性水平 $\alpha\in(0,1)$，令 $n_h=|\{i:h_i=h,\,b_i\in\mathcal{B}_{\text{cal}}\}|$，取有限样本修正的分位数水平

$$
\beta_h=\min\!\left\{\frac{\lceil(n_h+1)(1-\alpha)\rceil}{\max(n_h,1)},\,1\right\},\tag{31}
$$

并定义

$$
q_h=\mathrm{Quantile}_{\beta_h}\!\bigl(\{s^{(h)}_i\}_{i:h_i=h}\bigr).\tag{32}
$$

若某 $h$ 的校准样本为空，回退到 $q_h=1.645$（对应正态近似下的 90% 分位）。

### C. 原始区间

对测试样本 $(b,t_0,h)$，未投影的**原始区间**定义为

$$
\widehat{L}^{\,\text{raw}}_{b,t_0+h}=\widehat{E}^{\text{phys}}_{b,t_0+h}+\widehat{\mu}^{(h)}_{b,t_0+h}-q_h\,\widehat{\sigma}^{(h)}_{b,t_0+h},\tag{33}
$$

$$
\widehat{U}^{\,\text{raw}}_{b,t_0+h}=\widehat{E}^{\text{phys}}_{b,t_0+h}+\widehat{\mu}^{(h)}_{b,t_0+h}+q_h\,\widehat{\sigma}^{(h)}_{b,t_0+h}.\tag{34}
$$

### D. 物理可行域投影

将原始区间投影到 (23) 定义的可行域内：

$$
\widehat{L}_{b,t_0+h}=\max\!\bigl\{\widehat{L}^{\,\text{raw}}_{b,t_0+h},\,L^{\text{phys}}_{b,t_0+h}\bigr\},\quad
\widehat{U}_{b,t_0+h}=\min\!\bigl\{\widehat{U}^{\,\text{raw}}_{b,t_0+h},\,U^{\text{phys}}_{b,t_0+h}\bigr\},\tag{35}
$$

并在数值上保证 $\widehat{L}\le\widehat{U}$：$(\widehat{L},\widehat{U})\leftarrow(\min(\widehat{L},\widehat{U}),\,\max(\widehat{L},\widehat{U}))$。最终的 $1-\alpha$ 区间即为 $\widehat{\mathcal{I}}^{\,1-\alpha}_{b,t_0+h}=[\widehat{L}_{b,t_0+h},\widehat{U}_{b,t_0+h}]$。

### E. 有限样本覆盖性质

在校准集与测试集样本**可交换**（exchangeability）的假设下，分时距原始区间 $[\widehat{L}^{\,\text{raw}},\widehat{U}^{\,\text{raw}}]$ 具有下述分布无关的覆盖保证：

$$
\Pr\!\left(E_{b,t_0+h}\in\bigl[\widehat{L}^{\,\text{raw}}_{b,t_0+h},\widehat{U}^{\,\text{raw}}_{b,t_0+h}\bigr]\right)\ge 1-\alpha,\quad \forall h\in\mathcal{H}.\tag{36}
$$

物理投影 (35) 只可能在 raw 区间穿越可行域边界时使区间收缩，从而对**可行范围内的条件覆盖**保持可解释性；若需严格保留 (36) 的有效性，可将物理可行域视为先验工程约束，并在评估时一并汇报投影修正幅度（见第八节）。

## 七、评估口径

Phase C 采用与 Phase B 对齐、并补充不确定性度量的三级口径。

### A. 总体点预测指标

令测试集索引为 $\mathcal{T}=\{i:b_i\in\mathcal{B}_{\text{te}}\}$，定义

$$
\mathrm{MAE}=\frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}\bigl|E_i-\widehat{E}_i\bigr|,\quad
\mathrm{RMSE}=\sqrt{\frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}\bigl(E_i-\widehat{E}_i\bigr)^2},\tag{37}
$$

$$
\mathrm{MAPE}=\frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}\frac{|E_i-\widehat{E}_i|}{\max(|E_i|,\epsilon)}.\tag{38}
$$

### B. 严格 24 h 轨迹指标

记完整 24 h 轨迹集合 $\mathcal{J}=\bigl\{(b,t_0):\{h:(b,t_0,h)\in\mathcal{T}\}=\{1,\ldots,24\}\bigr\}$。定义轨迹级 MAPE、峰误差、谷误差：

$$
\mathrm{MAPE}^{\text{traj}}=\frac{1}{|\mathcal{J}|}\sum_{(b,t_0)\in\mathcal{J}}\frac{1}{H}\sum_{h=1}^{H}\frac{|E_{b,t_0+h}-\widehat{E}_{b,t_0+h}|}{\max(|E_{b,t_0+h}|,\epsilon)},\tag{39}
$$

$$
\mathrm{PE}=\frac{1}{|\mathcal{J}|}\sum_{(b,t_0)\in\mathcal{J}}\Bigl|E_{b,t_0+h^{\star}_b}-\widehat{E}_{b,t_0+h^{\star}_b}\Bigr|,\ h^{\star}_b=\arg\max_h E_{b,t_0+h},\tag{40}
$$

$$
\mathrm{VE}=\frac{1}{|\mathcal{J}|}\sum_{(b,t_0)\in\mathcal{J}}\Bigl|E_{b,t_0+h^{\dagger}_b}-\widehat{E}_{b,t_0+h^{\dagger}_b}\Bigr|,\ h^{\dagger}_b=\arg\min_h E_{b,t_0+h}.\tag{41}
$$

### C. 分时距不确定性指标

对每一 $h$ 分别计算经验覆盖率与平均区间宽度：

$$
\mathrm{Cov}_h^{\,1-\alpha}=\frac{1}{|\mathcal{T}_h|}\sum_{i\in\mathcal{T}_h}\mathbf{1}\!\bigl\{\widehat{L}_i\le E_i\le\widehat{U}_i\bigr\},\tag{42}
$$

$$
\mathrm{Width}_h^{\,1-\alpha}=\frac{1}{|\mathcal{T}_h|}\sum_{i\in\mathcal{T}_h}\bigl(\widehat{U}_i-\widehat{L}_i\bigr),\tag{43}
$$

其中 $\mathcal{T}_h=\{i\in\mathcal{T}:h_i=h\}$。

## 八、物理可行性诊断

为定量刻画物理投影 (35) 对 raw 输出的修正程度，定义如下诊断量，分别在 $\{\text{train},\text{cal},\text{te}\}$ 上汇报：

$$
\mathrm{VR}^{\text{pt}}_{\text{low}}=\frac{1}{N}\sum_{i}\mathbf{1}\!\{\widehat{E}_i<L^{\text{phys}}_i\},\quad
\mathrm{VR}^{\text{pt}}_{\text{up}}=\frac{1}{N}\sum_{i}\mathbf{1}\!\{\widehat{E}_i>U^{\text{phys}}_i\},\tag{44}
$$

$$
\mathrm{VR}^{\text{iv}}_{\text{low}}=\frac{1}{N}\sum_{i}\mathbf{1}\!\{\widehat{L}^{\,\text{raw}}_i<L^{\text{phys}}_i\},\quad
\mathrm{VR}^{\text{iv}}_{\text{up}}=\frac{1}{N}\sum_{i}\mathbf{1}\!\{\widehat{U}^{\,\text{raw}}_i>U^{\text{phys}}_i\},\tag{45}
$$

$$
\Delta^{\text{proj}}=\frac{1}{N}\sum_{i}\Bigl(\bigl|\widehat{L}_i-\widehat{L}^{\,\text{raw}}_i\bigr|+\bigl|\widehat{U}_i-\widehat{U}^{\,\text{raw}}_i\bigr|\Bigr).\tag{46}
$$

其中 (44) 衡量点预测的物理越界率，(45) 衡量未经投影的区间越界率，(46) 为投影引入的平均边界位移幅度。

## 九、端到端算法

综合第二至六节，Phase C 的推断流程可概括为算法 1。

> **Algorithm 1** PG-RNP-CQR inference at origin $t_0$ for station $b$
>
> 1. For each $h\in\mathcal{H}$, assemble $\mathbf{x}_{b,t_0+h}$ by (6)–(7) and retrieve $\widehat{E}^{\text{phys}}_{b,t_0+h}$ by (1).
> 2. Construct context set $\mathcal{C}_{b,t_0}$ by (12), compute $\mathbf{c}_{b,t_0}$ by (13), and infer $\mathbf{z}_{b,t_0}=\boldsymbol{\mu}_z$ by (14).
> 3. For each $h$, compute $\widetilde{\mu}^{(h)},\widetilde{\sigma}^{(h)}$ by (17)–(19) and de-standardize via (21).
> 4. Obtain the point forecast $\widehat{E}_{b,t_0+h}$ by (22).
> 5. Apply horizon-wise CQR using $\{q_h\}_{h=1}^{H}$ pre-fitted on $\mathcal{D}_{\text{cal}}$ via (30)–(32), yielding raw intervals (33)–(34).
> 6. Project onto the physical feasibility box (23) by (35) to output the final $1-\alpha$ prediction interval (4).

在训练阶段，依次对 (28) 进行小批量随机梯度下降，并依据校准集 NLL 施加早停。

## 十、讨论

上述建模可从三个层面给出“物理引导”语义：其一，式 (1) 以 Phase A 的静态基功率与 Phase B 的物理动态项构造先验可解释基线，使神经网络仅负责较小范数的残差拟合；其二，式 (26) 以软约束形式将物理可行域 (23) 嵌入训练损失，抑制学习到的残差在幅值上破坏能耗的工程可行性；其三，式 (35) 在推理阶段对保形区间作硬投影，使最终输出的区间与现场可观测的物理量纲一致，同时借助 (44)–(46) 对投影影响进行透明化。

此外，式 (30) 以 $\widehat{\sigma}^{(h)}$ 对非一致性分数进行尺度化，使 $q_h$ 保持与不确定性估计相容；分时距的 CQR 设计进一步缓解了 5G 基站能耗在不同 horizon 上噪声方差异质性对全局保形分位数的偏倚。综合而言，Phase C 在**点预测精度**、**分布式不确定性表达**与**物理一致性**之间提供了显式的数学可控折衷，为小样本情形下的日前基站能耗预测提供了一种可解释且覆盖率可认证的建模方案。
