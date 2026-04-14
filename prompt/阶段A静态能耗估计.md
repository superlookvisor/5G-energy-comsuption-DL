你是电信网络能耗建模与机器学习领域的专家，熟悉5G基站能耗建模（EARTH模型、3GPP模型）、以及TSG（IEEE Transactions on Smart Grid）期刊论文写作风格。

现在请你基于我提供的数据，完成“阶段A：基站静态基础能耗估计”的完整研究流程，包括：

---

## 【一、任务背景】

我有三张数据表：

1）ECdata.csv（基站级能耗）

* Time（时间）
* BS（基站ID）
* Energy（整站能耗）

2）CLdata.csv（cell级负载和节能状态）

* Time, BS, CellName
* load（负载率）
* ESMode1~6（节能模式强度）

3）BSinfo.csv（cell静态配置）

* BS, CellName
* RUType, Transimission_mode
* Frequency, Bandwidth, Antennas
* maximum_transimission_power

注意：

* 一个BS可能包含多个cell
* 能耗只在BS级观测
* CL/BSinfo需要聚合到BS-时间粒度

---

## 【二、研究目标（阶段A）】

目标是估计：

P_base(b)：基站b的静态基础能耗

要求：

* 尽量接近“低负载、无节能状态”下的真实基础功耗
* 为后续动态建模提供物理基准

---

## 【三、方法设计要求】

请你实现并对比至少三类方法：

### （1）静态窗口筛选策略（必须对比）

定义“近静态样本”：

尝试不同阈值组合，例如：

方案A：

* load_pmax_weighted < 0.05
* ES_total < 1e-6

方案B：

* load_mean < 0.1
* ESMode全部接近0

方案C：

* 使用分位数：每个BS取 load 最低 5%

方案D（更宽松）：

* load < 0.15 + 无ES约束

👉 要比较不同窗口对结果稳定性的影响

---

### （2）静态能耗估计方法（必须对比）

在筛选出的静态窗口内，对每个BS估计P_base：

至少实现：

方法1：最小值
P_base = min(Energy)

方法2：低分位数
P_base = quantile(Energy, 0.05 / 0.1)

方法3：均值
P_base = mean(Energy in static window)

方法4：稳健估计（推荐）

* median
* 或 trimmed mean

👉 必须分析这些方法的偏差与稳定性

---

### （3）静态特征建模（BS级）

构建基站静态特征：

* n_cells
* sum_pmax
* sum_bandwidth
* sum_antennas
* mean_frequency
* 高频占比（Frequency>500）
* Transmission mode占比
* RUType编码（可one-hot或embedding）

并建立：

P_base(b) = f(static_features)

要求实现：

* 线性回归（baseline）
* Ridge / Lasso
* 可选：随机森林（用于分析特征重要性）

---

---

## 【四、代码要求】

请输出完整可运行Python代码，包括：

1）数据读取与清洗
2）cell→BS聚合（重点）

* 构造 load_pmax_weighted
* 构造 ES_total
  3）静态窗口筛选
  4）多种P_base估计方法
  5）构建BS级数据集
  6）训练静态模型
  7）评估（R² / MAE / 分布分析）
  8）可视化（必须有）：
* 不同方法P_base分布对比
* P_base vs sum_pmax / n_cells散点图
* 不同窗口下P_base稳定性

代码要求：

* 使用 pandas / numpy / sklearn / matplotlib
* 结构清晰（函数化）
* 注释充分（中文）

---

## 【五、结果分析（非常重要）】

请你输出类似TSG期刊“Simulation Results”部分的分析，用中文，包含：

### 1）静态窗口敏感性分析

* 不同阈值对样本数量影响
* 对P_base的影响（偏高/偏低）

### 2）估计方法对比

* min vs quantile vs mean vs median
* 哪种更稳定？为什么？

### 3）物理合理性验证

* P_base是否随：

  * n_cells ↑
  * sum_pmax ↑
  * antennas ↑
    呈单调上升？

### 4）统计分析

* 方差
* 置信区间

### 5）关键发现（像论文那样写）

例如：（中文）

* “The minimum-based estimator tends to underestimate…”
* “Quantile-based estimation provides a robust trade-off…”

---

---

## 【六、输出格式】

请按以下结构输出：

1）代码（完整）
2）关键图表说明（文字描述）
3）结果分析（TSG风格，中文）
4）结论（bullet points）

要求：

* 中文写作
* 学术风格
* 类似IEEE TSG论文

---

## 【七、额外要求（加分项）】

如果可能，请额外分析：

* 哪些BS没有静态窗口样本？
* 如何用“配置相似BS”进行补偿（简单讨论即可）

---

现在请开始。
