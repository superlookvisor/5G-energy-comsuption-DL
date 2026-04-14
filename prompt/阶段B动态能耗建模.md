你是无线通信能耗建模与物理信息机器学习领域专家，熟悉5G基站功耗模型（EARTH模型、3GPP模型）以及IEEE TSG期刊写作规范。

现在请在“阶段A：静态基础能耗估计”基础上，完成：

👉 **阶段B：支持日前预测与日内滚动预测的动态能耗建模框架**

---

## 【一、任务定义（必须区分）】

请明确区分两个预测任务：

### 任务1：日前预测（Day-ahead Forecasting）

在时间 T（前一天），预测未来24小时：

E(b, t+1 ~ t+24)

⚠️ 注意：
未来时刻的以下变量不可用：

* load
* ESMode
* D1/D2/D3

---

### 任务2：日内滚动预测（Intra-day Rolling Forecasting）

在时间 t，预测未来H步（如1~4小时）：

E(b, t+1 ~ t+H)

✅ 当前时刻可观测：

* load(t)
* ESMode(t)
* Energy(t)

---

---

## 【二、统一能耗分解】

使用：

E(b,t) = P_base(b) + P_dynamic(b,t)

其中：
P_base(b) 已由阶段A得到

本阶段建模：
👉 P_dynamic(b,t)

---

## 【三、总体建模框架（核心要求）】

请构建：

👉 一个统一的动态能耗函数：

P_dynamic = f(load, config, ES, time)

并在不同任务中：

* 日内滚动：使用真实 load/ES
* 日前预测：使用预测值或历史代理

---

## 【四、特征设计（分两套）】

---

## ① 日内滚动预测特征（Real-time Features）

可使用真实值：

* load_mean(t)
* load_max(t)
* load_std(t)
* load_pmax_weighted(t)

构造：

D1 = Σ(Pmax × load)
D2 = Σ(Pmax × load²)
D3 = load_std

节能特征：

S_m = Σ(Pmax × ESMode_m)/Σ(Pmax)
I_m = S_m × load_pmax_weighted

时间特征：

* hour
* day-of-week

滞后特征（必须）：

* Energy(t-1), Energy(t-2)
* load(t-1)
* ES(t-1)

---

## ② 日前预测特征（Forecast-time Features）

⚠️ 不能使用未来真实 load/ES

必须构造：

### 方法A（推荐）：两阶段

先预测：

* load_mean_hat(t+h)
* load_pmax_weighted_hat(t+h)

再构造：

* D1_hat, D2_hat, D3_hat
* S_m_hat（可用历史均值或分类预测）

---

### 方法B：历史代理特征

* 同时刻历史平均：
  load_mean(t-24), load_mean(t-168)
* 滚动统计：
  mean(last 24h), std(last 24h)
* ES历史激活概率：
  P(ESMode_m | hour)

---

---

## 【五、模型设计（必须对比）】

---

① 物理驱动模型（核心）

P_dynamic =
α1·D1 + α2·D2 + α3·D3
− Σ(δ_m·S_m + τ_m·I_m)

约束：

* α ≥ 0
* δ ≥ 0

---

② 半物理增强模型

加入：

* load²
* load × n_cells
* load × sum_pmax
* ES × load

模型：

* Ridge / Lasso

---

③ 黑箱模型（对比）

* Random Forest 或 XGBoost

用于验证：
👉 物理模型 vs 纯数据驱动

---

## 【六、训练策略（关键修改）】

### 1）目标变量

y = Energy - P_base

---

### 2）数据划分（必须）

使用：

👉 GroupKFold（按BS划分）

---

### 3）两种训练方式

#### （A）日内模型

使用真实 load/ES

#### （B）日前模型

使用：

* load预测值
  或
* 历史代理特征

⚠️ 禁止使用未来真实值

---

---

## 【七、评估设计（新增重点）】

必须分别评估：

### 1）日内滚动预测性能

* 1-step ahead
* 3-step ahead
* 6-step ahead

指标：

* MAE
* RMSE
* R²

---

### 2）日前预测性能

预测24小时：

* 整体误差（MAPE）
* 峰值误差
* 谷值误差

---

### 3）物理一致性验证

检查：

* 能耗随负载是否单调增加
* 节能模式是否降低能耗
* 多cell是否增加基线能耗

---

---

## 【八、可视化（必须）】

画图：

1）日前 vs 日内预测对比曲线
2）load vs energy
3）ES模式影响
4）预测 vs 真实
5）不同预测步长误差

---

## 【九、TSG论文风格分析（必须）】

写出：

### 1）Dynamic Energy Behavior

* 负载驱动性
* 非线性

### 2）Day-ahead vs Intra-day Comparison

* 哪种更准？
* 为什么？

### 3）Impact of ES Modes

* 哪些有效

### 4）Model Comparison

### 5）Key Findings

---

---

## 【十、输出格式】

请输出：

1）完整代码
2）关键图说明
3）TSG风格分析（中文）
4）总结

---

现在开始执行。
