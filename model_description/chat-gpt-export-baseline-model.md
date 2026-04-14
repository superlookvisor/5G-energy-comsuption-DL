你这个纠正是对的。这里应该按 **“EC 是基站级整站能耗标签；CL/BSinfo 是 cell 级解释变量，需要先聚合到站点-时刻粒度”** 来处理，而不是把 CL/BSinfo 的基站数量差异误解成多出的站点标签。

基于你这三张表，我建议把问题定义成一个**站点级监督、cell级机理聚合、物理约束回归**的灰盒建模问题，而不是直接做端到端黑箱回归。经典基站功耗研究和 EARTH/3GPP 的主线都支持这种分解：基站总供电功率可以写成“静态功耗 + 随发射/负载变化的动态项 + 睡眠/节能导致的降耗项”，并且负载项在工程上常先近似成仿射或弱非线性形式。5G 的先进睡眠模式本质上也是不同持续时间、不同激活/唤醒代价下的分层部件关闭机制，因此节能模式应作为**持续时间占比或激活强度**进入模型，而不是简单 one-hot 状态。([开放研究][1])

先说你这批数据本身，我看下来有几个很关键的事实。

第一，**建模监督粒度必须统一到“BS-小时”**。
EC 是 `BS, Time -> Energy`；CL 是 `BS, CellName, Time -> load + ESMode`；BSinfo 是 `BS, CellName -> 配置参数`。所以正确流程一定是先把 `CL × BSinfo` 在 `(BS, CellName)` 上合并，再聚合成 `(BS, Time)` 特征表，最后和 EC 对齐。

第二，**大多数站是单 cell，但确实存在多 cell 站**。
你这份 BSinfo 里，约 827 个站是 1 cell，191 个站是 2 cell，2 个站是 4 cell。这个结构说明：

* 不能把 cell 级特征直接丢掉；
* 但也不能自由地给“每个 cell 单独拟合一套能耗参数”，因为你没有 cell 级能耗标签，这会出现**不可辨识**。

第三，**EC 不是完整平衡面板**。
你这份 EC 每站只有 1 到 130 个小时不等，中位数大约 110 小时，不是完整 168 小时周序列。这个点很重要：

* 不要先把缺失能耗整段插补后再建模；
* 训练时按“观测到的站点-小时样本”做截面/短面板建模更稳；
* 评估一定要按 **BS 分组切分**，不能随机按行切，否则会严重泄漏。

第四，**节能模式不是互斥状态**。
你这份 CL 里 ESMode1、2、6 是主要模式，ESMode4 全 0，ESMode5 极稀疏；并且有一部分样本里多个 ESMode 同时为正，所以这些字段更像“该小时内该机制的生效占比/强度”，不是独占状态机。这个判断会直接影响你后面特征设计。

第五，**从你的数据做了一个很粗的组别外检验**，结果很说明问题：

* 只用静态配置特征，解释力明显不够；
* 加入负载后，性能显著提升；
* 再加入 ES 特征和负载-配置交互，效果继续提升。
  这恰好说明你的任务确实适合“静态基础项 + 动态负载项 + 节能修正项”的结构。

---

## 一、已有数据该怎么预处理

### 1）先做“站点-小时”物理聚合，而不是简单求平均

对每个基站 (b)、时刻 (t)，先把该站下所有 cell $(i \in \mathcal C_b)$ 聚合。核心不是做普通均值，而是构造**物理等效量**。

最推荐保留下面几类量：

**静态配置量（只跟 BSinfo 有关）**

* $(C_b)$：cell 数
* $(\sum_i P^{\max}_{b,i})$：站点最大总发射功率
* $(\sum_i BW_{b,i})$：总带宽
* $(\sum_i A_{b,i})$：总天线数
* 频段统计：均值频率、最高频率、高频 cell 占比
* RUType、Transmission mode 的站点签名或 config family id

**动态负载量（CL 聚合）**

* 负载均值：$(\bar l_{b,t})$
* 负载和：$(\sum_i l_{b,i,t})$
* 负载最大值：$(\max_i l_{b,i,t})$
* 负载离散度：$(\mathrm{std}(l_{b,i,t}))$ 或 $( \max - \mathrm{mean})$

其中最重要的是一个**等效负载**
$[
L^{eq}_{b,t}=\frac{\sum_i w_{b,i} l_{b,i,t}}{\sum_i w_{b,i}}
]$
权重 $(w_{b,i})$ 推荐先用 $(P^{\max}_{b,i})$。
如果你后面发现带宽差异影响更强，可以试 $(w_{b,i}=P^{\max}_{b,i}\times BW_{b,i})$。

这样做的理由是：经典基站功耗模型本来就是把供电功率和发射功率/负载联系起来，而发射能力更强的 cell 对整站动态功耗的贡献本应更大。([开放研究][1])

**节能模式量**
不要直接保留 cell 级 ESMode 原值，而要构造：
$[
S^{(m)}_{b,t}=\frac{\sum_i P^{\max}_{b,i} \cdot ES^{(m)}_{b,i,t}}{\sum_i P^{\max}_{b,i}}
]$
以及交互项
$[
I^{(m)}_{b,t}=L^{eq}_{b,t}\cdot S^{(m)}_{b,t}
]$

原因很简单：
节能模式本质上影响的是“同样负载下的供电功率”，所以它既有**直接降耗效应**，也有**改变负载斜率**的效应。3GPP/ASM 文献里，不同睡眠层级对应不同部件关闭深度和不同唤醒代价，不能只用一个统一偏移量描述。([arXiv][2])

### 2）把 ESMode 当“机制强度”，不要当 one-hot

你这里 ESMode1/2/6 明显不是互斥离散状态，所以建议：

* ESMode4：直接删掉
* ESMode5：极稀疏，可并入“rare ES”或先删
* ESMode1/2/3/6：保留为连续强度
* 额外加一个 `ES_total = ES1+ES2+ES3+ES6`

但**不要**把 `ES_total` 作为唯一 ES 特征，因为这会丢失不同节能机制的物理差异。

### 3）对于相同BSid,不同CellName的 97 个BS数据
先把每个 cell 的配置和时变状态表示出来，
再通过一个“可学习的聚合规则”把多个 cell 合成为整站输入，最后用整站能耗做监督
对每个 cell 𝑖 ，在时刻 _t_，定义它的输入：
$[
x_{b,i,t}
=
[\text{RUType},\text{Mode},\text{Frequency},
\text{Bandwidth},\text{Antennas},P^{max},load,ES]
]$
定义一个共享的 cell 响应函数：
* $[
g_\theta(x_{b,i,t})
]$表示：
给定一个 cell 的硬件配置、负载、节能状态
这个 cell 对整站能耗贡献多少“发射相关能耗”

然后整站能耗写成：

$
\hat E_{b,t}
=
P^{base}_b + \sum_{i\in \mathcal C_b} g_\theta(x_{b,i,t})
$

这里：
* $P^{base}_b$：整站基座能耗
* $g_\theta$：Cell参数




### 4）异常值和缺失处理

建议：

* `Energy` 不插值，保留真实观测小时
* 对 `load`、`Energy` 做站内 winsorize，例如 0.5%–99.5%
* 对极少观测站点单独打标，比如样本数 < 24 的站先不参与主模型拟合
* 对配置字段做一致性校验：同一 `BS-CellName` 的静态配置应该不随时间变

### 5）评估方式必须按 BS 分组

至少做两套：

* **Seen-BS**：同一站不同小时的预测
* **Unseen-BS / GroupKFold by BS**：新站泛化

你后面如果要发论文，这个划分非常关键。随机打散行做 CV，结果通常会虚高。

---

## 二、怎么做“物理知识驱动的能耗基准模型”

这里我不建议你直接在站点级写一个纯经验式多项式。更好的办法是：

### 先在 cell 层定义共享机理，再聚合到站点层

因为你没有 cell 能耗标签，所以不能给每个 cell 单独拟合自由参数；
但你可以假设**所有 cell 共享一套物理响应函数**，然后把它们求和成整站能耗。

可以这样写：

$
E_{b,t}=P^{site}_b+\sum_{i\in \mathcal C_b} p_{b,i,t}^{cell}+\varepsilon_{b,t}
$

其中

$
p_{b,i,t}^{cell}$=
$P^{stat}(z_{b,i})$+
$P^{dyn}(z_{b,i},l_{b,i,t})$+
$
P^{save}(z_{b,i},l_{b,i,t},\mathbf e_{b,i,t})
$

这里：

* $(z_{b,i})$：cell 静态配置，如 $(P^{\max})$、带宽、天线、频段、RUType
* $(l_{b,i,t})$：负载
* $(\mathbf e_{b,i,t})$：各 ES mode 强度

这是最关键的一步：**共享 cell 响应 + 站点求和**，就把“没有 cell 标签”的不可辨识问题，变成了“共享参数的可估计问题”。

---

## 三、一个适合你数据的可落地基准式

我建议你先从下面这个版本开始：

$
\hat E_{b,t}$
=
$P^{base}_b$+
$\underbrace{\alpha_1 D_{1,b,t}+\alpha_2 D_{2,b,t}+\alpha_3 D_{3,b,t}}
_{\text{动态传输项}}$
-$\underbrace{\sum_{m\in \mathcal M}
\left(
\delta_m M_{m,b,t}+\tau_m I_{m,b,t}
\right)}_{\text{节能修正项}}$
$

其中：

### 1）静态基础项

$
P^{base}_b=
\beta_0
+\beta_1 \sum_i P^{\max}_{b,i}
+\beta_2 \sum_i A_{b,i}
+\beta_3 \sum_i BW_{b,i}
+\beta_4 F_b
+u_{\text{cfg}(b)}
$

这里$ (u_{\text{cfg}(b)}) $是配置族随机效应或固定效应。
因为经典功耗模型里，静态项和 TRX 数、天线链路数、带宽、PA/RF/BB 配置强相关。([开放研究][1])

### 2）动态传输项

定义：
$
D_{1,b,t}=\sum_i P^{\max}_{b,i} l_{b,i,t}
$

$
D_{2,b,t}=\sum_i P^{\max}_{b,i} l_{b,i,t}^2
$

$
D_{3,b,t}=\mathrm{std}(l_{b,i,t})
$

解释：

* $(D_1)$：一阶等效业务强度
* $(D_2)$：负载非线性
* $(D_3)$：多 cell 间负载不均衡，反映“同总负载但调度分布不同”的能耗差异

经典 EARTH/参数化模型认为供电功率与输出功率/负载近似仿射，但在 5G 多天线、多频点、节能并发情况下，加入一个轻量二次项更稳。([arXiv][3])

### 3）节能修正项

定义：
$
M_{m,b,t}=\sum_i P^{\max}_{b,i} ES^{(m)}_{b,i,t}
$

$
I_{m,b,t}= \sum_i P^{\max}_{b,i} l_{b,i,t} ES^{(m)}_{b,i,t}
$

解释：

* $(M_m)$：模式 (m) 的直接降耗效应
* $(I_m)$：模式 (m) 对动态斜率的修正

这个设计和 ASM/睡眠模式物理直觉是一致的：
浅睡眠更像减少某些活动部件的瞬时耗电，深睡眠则更像显著压低底座功耗，并且可能改变“负载增加 1 单位时能耗增长多少”。([arXiv][2])

---

## 四、参数约束要怎么加

这一步很重要，不然你后面再复杂的 PI-ANP 也会被坏基线拖累。

建议至少加这些约束：

$
\beta_1,\beta_2,\beta_3 \ge 0,\quad
\alpha_1,\alpha_2 \ge 0
$

$
\delta_m,\tau_m \ge 0
\quad \text{（对已确认是节能模式的项）}
$

$
\frac{\partial \hat E_{b,t}}{\partial l_{b,i,t}} \ge 0
\quad \text{当所有 ES 关闭时}
$

$
\hat E_{b,t} \ge P^{sleep}_b
$

其中$ (P^{sleep}_b)$ 可以不用直接设成固定常数，而是设成某个较低分位物理下界。
经典参数化模型明确区分了非零发射时的负载仿射功耗和零发射时的睡眠功耗 $(P_{sleep})$。([arXiv][3])

---

## 五、你的模型应该怎么分两阶段来拟合

### 阶段 A：先估静态基础能耗

从低负载、低 ES 强度样本里估计每站基础项。

比如定义一个近静态窗口：
$
L^{eq}_{b,t}<\tau_l,\qquad ES_{b,t}^{total}<\tau_s
$

在这些样本上取：

* 站内 10% 分位数
* 或站内低负载样本中位数

得到一个粗的 $( \tilde P^{base}_b )$。

然后用站点静态配置去回归这个$ ( \tilde P^{base}_b )$，得到：
$
P^{base}_b = f_{\text{static}}(\text{config}_b)
$

这样做的好处是：先把“底座”钉住，后面动态项不容易吸收静态差异。

### 阶段 B：再拟合动态项和节能项

对残差：
$
R_{b,t}=E_{b,t}-P^{base}_b
$
用约束回归拟合
$
R_{b,t}\approx
\alpha_1 D_1+\alpha_2 D_2+\alpha_3 D_3
-\sum_m(\delta_m M_m+\tau_m I_m)
$

拟合方法优先级：

1. 非负最小二乘 / 约束二次规划
2. 带单调约束的 GAMI-Net / XGBoost monotone
3. 分层贝叶斯 / mixed-effects

对你这个数据规模，我第一版会先用 **约束线性/弱非线性模型**，因为最稳、最好解释、最适合论文里作为“物理基准模型”。

---

## 六、为什么我不建议一上来就做纯黑箱

因为你的任务天然存在三个 OOD 风险：

1. **多 cell 站点的组合差异**
2. **节能模式稀疏且并发**
3. **无标签站点配置分布和有标签站点不完全一致**

纯数据驱动模型很容易把“某些站的 ID / 配置 fingerprint”偷学成经验映射。你前面那段论文表述里说“避免样本外场景物理失真”，这个判断是对的。经典 BS 功耗模型本来就强调“负载功率映射”和“组件级供电项”的结构，不能完全丢。([开放研究][1])

---

## 七、后续怎么接 PI-ANP 做不确定集

这里最稳的路线不是让 PI-ANP 直接学总能耗，而是：

### 用 PI-ANP 学“物理基线的残差”

即先有
$
\hat E^{phys}_{b,t}
$
再让 PI-ANP 学
$
r_{b,t}=E_{b,t}-\hat E^{phys}_{b,t}
$

最终
$
\hat E_{b,t}=\hat E^{phys}_{b,t}+\hat r^{PI\text{-}ANP}_{b,t}
$

这样做有三个好处：

* 物理基线负责大结构，避免黑箱胡跑
* ANP 只学剩余复杂耦合，样本效率更高
* 不确定性更容易解释成“物理基线外的剩余不确定性”

Attentive Neural Process 的优势就在于：它学习的是**条件函数分布**，可以用可变大小的 context set 对新任务/新站点做 few-shot 适配，这对“不同基站对应不同能耗函数”这种任务特别合适。([bayesiandeeplearning.org][4])

### PI-ANP 的输入建议

输入 $(x_{b,t})$：

* 站点静态配置 embedding
* 等效负载 $(L^{eq})$
* 负载不均衡 $(D_3)$
* 各 ES mode 强度及交互项
* 时间位置编码（小时、工作日/周末）
* 物理基线输出 $(\hat E^{phys}_{b,t})$
* 物理基线中间量：$(P^{base}_b, D_1, D_2, M_m, I_m)$

输出：

* 残差均值 $(\mu_r(x))$
* 残差方差$ (\sigma_r^2(x))$

### 物理约束怎么塞进 PI-ANP

可以在 loss 里加下面几类项：

**1. 分解一致性约束**
$
\hat E - \hat P^{base} - \hat P^{dyn} + \hat P^{save} \approx 0
$

**2. 单调性约束**
$
\frac{\partial \hat E}{\partial L^{eq}} \ge 0
\quad (\text{ES关闭时})
$

**3. 节能非增约束**
$
\frac{\partial \hat E}{\partial S^{(m)}} \le 0
$
对已明确属于节能的模式成立。

**4. 可行域约束**
$
P^{sleep}_b \le \hat E_{b,t} \le P^{full}_b
$

ProbConserv 这类方法的启发非常适合你：先用一个概率模型给出均值和协方差，再用物理约束做后验更新，从而在保留不确定性的同时让输出满足物理规律。相关工作就是“黑箱概率模型 + 物理约束 Bayesian update”的两步框架。([OpenReview][5])

---

## 八、你的“不确定集”建议怎么定义

我建议不要只给一个方差，而是给**物理可行的不确定区间/集合**。

### 第一步：PI-ANP 先给出概率预测

$
Y_{b,t}\mid x \sim \mathcal N(\mu_{b,t},\sigma_{b,t}^2)
$

### 第二步：用 conformal calibration 做覆盖率校准

在验证站点上计算分数
$
s_j=\frac{|y_j-\mu_j|}{\sigma_j}
$
取分位数$(q_{1-\alpha})$，得到
$
\mathcal U_\alpha(x)=
\left[
\mu(x)-q_{1-\alpha}\sigma(x),;
\mu(x)+q_{1-\alpha}\sigma(x)
\right]
$

这样做的原因是：单纯神经网络输出的方差经常不校准，而 conformal prediction 可以提供有限样本下更可靠的区间覆盖思路。近年的 physics-informed UQ 研究也在往“物理模型 + conformal 校准”这个方向走。([arXiv][6])

### 第三步：把区间投影到物理可行域

$
\mathcal U_\alpha^{phys}(x)
=
\mathcal U_\alpha(x)\cap
[P^{sleep}_b; P^{full}_b]
$

如果你后面做鲁棒优化或异常检测，这个物理投影非常有用，因为它避免出现“负能耗”或者“超过设备能力上界”的假区间。

---

## 九、一个最适合你现在立即推进的技术路线

按优先顺序，我建议你这样做：

**第 1 步**
构造 `BS-Time` 训练面板，至少包含：

* `Energy`
* `n_cells`
* `sum_pmax, sum_bw, sum_ant`
* `load_mean, load_max, load_std, load_pmax_weighted`
* `ES1/2/3/6` 的 pmax 加权强度
* `ES × load` 交互
* `config_family_id`

**第 2 步**
做一个约束灰盒基线：
[
E = P^{base}(\text{static}) + P^{dyn}(\text{load, config}) - P^{save}(\text{ES, load})
]

**第 3 步**
只对残差建 PI-ANP，而不是对总能耗建 PI-ANP。

**第 4 步**
用按 BS 分组的验证集做 conformal calibration，输出物理可行不确定集。

---

## 十、给你一个论文里可直接用的表述方向

你前文那句可以顺着扩展成：

> 鉴于基站整站能耗标签仅在 BS 级可观测，而业务负载与节能状态信息在 cell 级给出，本文将基站能耗建模为“共享 cell 响应函数经站点聚合后的总供电功率”。具体地，首先依据 cell 静态配置构建站点静态基础能耗项，再以加权负载、负载离散度及节能模式强度刻画动态传输能耗与节能修正项，并通过非负性、单调性及睡眠功耗下界等物理约束实现可解释参数辨识。在此基础上，进一步以物理基准模型残差为学习对象，构建 PI-ANP 概率模型，并结合 conformal 校准形成满足物理可行域约束的基站能耗不确定集。

下一步最合适的是先把这三张表整理成 **站点-小时训练面板字段清单**，再把这个物理基准模型写成可训练的数学形式和代码框架。

[1]: https://openresearch.surrey.ac.uk/view/pdfCoverPage?download=true&filePid=13140534450002346&instCode=44SUR_INST "https://openresearch.surrey.ac.uk/view/pdfCoverPage?download=true&filePid=13140534450002346&instCode=44SUR_INST"
[2]: https://arxiv.org/pdf/1909.09011 "https://arxiv.org/pdf/1909.09011"
[3]: https://arxiv.org/pdf/1411.1571 "https://arxiv.org/pdf/1411.1571"
[4]: https://bayesiandeeplearning.org/2018/papers/28.pdf "https://bayesiandeeplearning.org/2018/papers/28.pdf"
[5]: https://openreview.net/pdf?id=TvP3l8-j-K8 "https://openreview.net/pdf?id=TvP3l8-j-K8"
[6]: https://arxiv.org/abs/2405.08111 "https://arxiv.org/abs/2405.08111"
