# 基于磷酸铁锂循环退化成本的基站储能：容量配置与日内调度一体化建模（方案2）

## 1. 问题定义与边界

### 1.1 系统构成

- 负荷（基站用电）\(L_t\)（kW）
- 光伏出力 \(PV_t\)（kW）
- 电池储能系统（BESS）：仅考虑**磷酸铁锂（LFP）循环老化**带来的退化成本
- 与电网连接：仅允许**购电**，不允许反送电（卖电）

### 1.2 优化目标

在分时电价（TOU）条件下，仅考虑基站侧成本，求解：

- **外层**：选择电池能量容量 \(E\)（kWh）
- **内层**：给定 \(E\) 后求解日内最优充放电调度，使 “购电成本 + 循环退化成本” 最小

### 1.3 时间尺度

按天滚动优化，小时粒度：

- 时间步 \(t=1,\dots,T\)，其中 \(T=24\)
- 时间步长 \(\Delta t=1\)（小时）

---

## 2. 两层优化结构（容量配置 + 调度）

### 2.1 外层（容量选择）

给定候选容量集合 \(\mathcal{E}\)（例如 $(\{0,5,10,\dots,E_{\max}\}$) kWh），
对每个$E\in\mathcal{E}$ 解内层调度问题得到日成本 \(J(E)\)，并选择：

$
E^* = \arg\min_{E\in\mathcal{E}} J(E)
$

> 注：由于内层包含退化成本（且退化成本与容量 \(E\) 成正比），在多数场景下 \(J(E)\) 会呈现合理的“先降后升/边际收益递减”的形状，从而获得有限的最优容量。

### 2.2 内层（日内调度，线性规划 LP）

在给定 \(E\) 与其他参数后，内层问题构造成线性规划（LP），
可用 `cvxpy` / `pyomo` / `pulp` 等求解器稳定求解。

---

## 3. 内层调度模型（LP）：变量、参数、约束、目标

### 3.1 变量（对每个小时 \(t\)）

- 购电功率：$g_t \ge 0$（kW）
- 充电功率：$p^{ch}_t \ge 0$（kW）
- 放电功率：$p^{dis}_t \ge 0$（kW）
- 电池能量状态：$soc_t$（kWh）
- （建议）弃光功率：$curt_t \ge 0$（kW），用于处理 “PV 超出需求且不允许外送” 的情况

为构造循环退化的凸分段线性成本，引入归一化 SOC：

- $s_t = soc_t / E$（无量纲）
- $\Delta s_t = s_{t+1} - s_t$
- 分解绝对值的辅助变量：$u_t \ge 0, v_t \ge 0$，使 $|\Delta s_t| = u_t + v_t$

以及分段线性“增量段”变量（见 4.2）：

- $y_{t,k}$（无量纲），$k=1,\dots,K$

### 3.2 参数

- 负荷：$L_t$（kW）
- 光伏：$PV_t$（kW）
- 分时电价：$\pi_t$（元/kWh）
- 充/放电效率：$\eta_{ch}\in(0,1],\ \eta_{dis}\in(0,1]$
- 充/放电功率上限：$P_{\max}$（kW）（由现有数据推断/给定）
- SOC 上下限（归一化）：$s_{\min}, s_{\max}$（例如 0.1 与 0.9）
- 电池更换/折算成本系数：$c_E^{capex}$（元/kWh）
- 电池更换等效成本：$C_{rep}(E) = c_E^{capex}\cdot E$（元）

### 3.3 约束

#### (1) 功率平衡（不允许反送电）

$
g_t + PV_t - curt_t + p^{dis}_t = L_t + p^{ch}_t,\quad \forall t
$

并且：

$
g_t \ge 0,\quad curt_t \ge 0,\quad \forall t
$

> 若你确认永远不会出现 PV 大于负荷且电池无法吸收的情况，可省略 \(curt_t\)，并强制 \(g_t\) 由平衡式确定且非负即可。但为鲁棒性与可行性，建议保留弃光变量。

#### (2) 充放电功率约束

$
0 \le p^{ch}_t \le P_{\max},\quad 0 \le p^{dis}_t \le P_{\max},\quad \forall t
$

#### (3) SOC 动态方程

$
soc_{t+1} = soc_t + \eta_{ch}\, p^{ch}_t\,\Delta t - \frac{1}{\eta_{dis}}\, p^{dis}_t\,\Delta t,\quad \forall t
$

并定义：
$
s_t = \frac{soc_t}{E},\quad \forall t
$

#### (4) SOC 边界（容量缩放）

对所有 \(t\)：

$
s_{\min} \le s_t \le s_{\max}
$

等价于：

$
s_{\min}E \le soc_t \le s_{\max}E
$

#### (5) 日循环一致性（推荐）

避免“借用初始电量”造成不可重复调度：

$
soc_{T+1} = soc_1
$

（若使用 $s_t$ 则为$s_{T+1}=s_1$）

---

## 4. 循环退化成本（方案2）：凸分段线性（PWL）近似 DoD 非线性

### 4.1 从 DoD-寿命曲线到“单位 SOC 摆幅损伤率”

准备一条 LFP 的 DoD-循环寿命曲线：

- $N(d)$：当循环深度（DoD）为 $d\in(0,1]$ 时，到达 EOL（如容量衰减到 80%）的循环次数

采用 Miner 线性累积损伤思想：深度为 $d$ 的完整循环损伤

$
\Delta D(d)=\frac{1}{N(d)}
$
一个深度为 $d$ 的完整循环，归一化 SOC 的总摆幅近似为 $2d$（放电摆幅 $d$ + 充电摆幅 $d$），因此定义“单位 SOC 摆幅”的损伤率：

$
r(d) \triangleq \frac{\Delta D(d)}{2d}=\frac{1}{2d\,N(d)}
$

该 $r(d)$ 随 $d$ 增大通常显著增大，体现“深循环更伤电池”。

### 4.2 用凸 PWL 函数逼近退化损伤

令：

$
\Delta s_t = s_{t+1}-s_t
$

用线性约束得到绝对值：

$
u_t \ge \Delta s_t,\quad v_t \ge -\Delta s_t,\quad u_t\ge 0,\ v_t\ge 0
$
从而：

$
|\Delta s_t| = u_t+v_t
$

定义一个凸分段线性函数 $\phi(x)$ $(x\in[0,1]$），用于把每步的 SOC 摆幅 $x=|\Delta s_t|$ 映射为损伤增量。选择断点：

$0={{b}_{0}}<{{b}_{1}}<\ldots <{{b}_{K}}\le 1$

并给定每段的斜率 $m_k$（满足 $m_1\le m_2\le \dots \le m_K$，以保证凸性与“深循环更贵”）。

采用“增量段法”进行线性化，引入 $y_{t,k}$：

$
0 \le y_{t,k} \le (b_k-b_{k-1}),\quad \forall t,k
$
$
\sum_{k=1}^K y_{t,k} = |\Delta s_t|,\quad \forall t
$

则每步损伤近似为：

$
\phi(|\Delta s_t|)=\sum_{k=1}^K m_k\,y_{t,k}
$

### 4.3 斜率 $m_k$的标定（来自 $N(d)$）

对每一段 $k$，取代表深度 $\bar d_k$（例如区间中点 $\bar d_k = (b_{k-1}+b_k)/2$），计算：

$
m_k \approx r(\bar d_k)=\frac{1}{2\bar d_k\,N(\bar d_k)}
$

为保证数值稳定与凸性，可对 $m_k$  做单调性修正：

$
m_k \leftarrow \max(m_k, m_{k-1})
$

---

## 5. 内层目标函数（购电成本 + 循环退化成本）

### 5.1 购电成本

$
C_{grid}=\sum_{t=1}^{T} \pi_t\, g_t \Delta t
$
### 5.2 循环退化成本（货币化）

将损伤乘以电池更换等效成本：

$
C_{deg}=C_{rep}(E)\cdot \sum_{t=1}^{T} \phi(|\Delta s_t|)
$

其中：

$
C_{rep}(E)=c_E^{capex}\cdot E
$

### 5.3 总目标

$
\min \quad J(E)= C_{grid}+C_{deg}
$

---

## 6. 数据设计建议（便于复现与论文写作）

### 6.1 DoD-寿命曲线

建议文件：`dod_cycle_life.csv`

- `dod`：循环深度（0~1）
- `cycles_to_eol`：对应 DoD 下的循环寿命 \(N(d)\)
- （可选）`source`：文献/数据手册来源标识

### 6.2 分时电价（TOU）

建议采用“两层数据”：

1) 规则表（政策/电价表 -> 规则）：
- `tou_rules.csv`：`effective_from,effective_to,day_type,season,start_hour,end_hour,price_yuan_per_kwh,region,...`

2) 展开后的日序列（优化直接用）：
- `prices_YYYYMMDD.csv`：`timestamp,price_yuan_per_kwh`
或直接提供长度 24 的 $\pi_t$ 数组。

---

## 7. 求解流程（落地执行顺序）

1) 读取一天的$L_t, PV_t, \pi_t$
2) 从 `dod_cycle_life.csv` 生成 PWL 断点 $\{b_k\}$ 与斜率 $\{m_k\}$
3) 外层遍历候选容量 $E\in\mathcal{E}$
4) 对每个 \(E\)：
   - 解内层 LP，得到最优 $g_t,p^{ch}_t,p^{dis}_t,soc_t$
   - 计算并记录 $J(E), C_{grid}(E), C_{deg}(E)$
5) 输出最优容量 $E^*$ 与对应调度结果（可同时输出 Pareto/敏感性分析曲线）

---

## 8. 备注与可选增强（不改变主模型结构）

- **弃光惩罚**：如希望避免无意义弃光，可对 $curt_t$ 加一个很小的惩罚系数（或直接不惩罚，表示弃光无成本）
- **不同充/放功率上限**：将$P_{\max}$拆为 $P^{ch}_{\max},P^{dis}_{\max}$
- **不同退化权重**：如文献支持“充电/放电不对称退化”，可对 $|\Delta s_t|$ 的正负变化分别使用不同 PWL，但会引入更多变量；在只考虑循环老化的入门版中一般不必

