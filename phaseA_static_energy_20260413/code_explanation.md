# 阶段A（`phaseA_static_energy_20260413`）代码说明：静态基础能耗 `P_base` 估计与静态特征建模

本文档面向阅读与复现 `run_phaseA_static_energy.py` 的实现细节，逐段解释其**做了什么**、**输入是什么**、**输出是什么**、以及各中间产物在目录中的位置。

---

## 1. 目录与文件角色

当前目录核心文件如下：

- `run_phaseA_static_energy.py`
  - **作用**：一键跑完整 Phase-A 流程（数据读取/清洗 → Cell→BS 聚合 → 静态窗口定义 → `P_base` 多方法估计 → 稳定性与覆盖率分析 → 静态特征建模 → 生成图表与报告）。
  - **输入**：`../data/ECdata.csv`、`../data/CLdata.csv`、`../data/BSinfo.csv`（相对项目根目录的 `data/`）。
  - **输出**：`outputs/` 下的 CSV 结果表 + `outputs/figures_emf/` 下的图（EMF）。

- `analysis_report.md`
  - **作用**：一次运行后的“论文风格”结果解读（窗口覆盖、方法稳定性、相关性、模型指标等）。
  - **来源**：由 `run_phaseA_static_energy.py` 的 `build_report()` 自动生成/覆盖写入。

- `outputs/method_stability_analysis_ieee_tsg.md`
  - **作用**：对“静态能耗估计方法跨窗口稳定性”的单独文字输出（偏 IEEE TSG 风格）。
  - **注意**：它不是脚本当前必写入的文件（属于已有成果/备份式输出）。

---

## 2. 运行方式与运行环境假设

### 2.1 一键运行

在项目环境中进入 `phaseA_static_energy_20260413/` 后运行：

```bash
python run_phaseA_static_energy.py
```

### 2.2 关键外部依赖与坑点

- **Python 依赖**：`numpy/pandas/matplotlib/scikit-learn`。
- **保存 EMF 图**：脚本通过“先存 SVG，再用 Inkscape 转 EMF”的方式实现。
  - 若系统未安装 Inkscape 或未加入 PATH，脚本会**自动降级**：保留 `SVG` 并额外导出 `PNG`，同时跳过 EMF 转换（不再阻断主流程）。
  - 若 Inkscape 可用，则会生成 `EMF`，并在转换后删除临时 `SVG`（但 `PNG` 会保留，便于快速预览）。
- **数据目录定位**：默认先找 `energy_model_anp/data/`，若不存在则自动尝试上一级的 `../data/`（例如你当前的数据位置 `G:\5G-energy-comsuption-DL\Small-sample-MLM\data`）。
  - 也可通过环境变量 `ENERGY_MODEL_DATA_DIR` 显式指定数据目录。

---

## 3. 数据输入：三张表的用途、关键字段与对齐方式

脚本固定读取三张 CSV（路径在 `load_data()` 中定义）：

- `data/ECdata.csv`（能耗）
  - **用途**：提供监督目标/观测量 `Energy`，粒度为（BS，Time）。
  - **关键字段**：`Time`, `BS`, `Energy`

- `data/CLdata.csv`（负载/节能模式）
  - **用途**：提供业务负载、节能模式状态，粒度为（BS，CellName，Time），后续会聚合到（BS，Time）。
  - **关键字段**：`Time`, `BS`, `CellName`, `load`, `ESMode1..ESMode6`

- `data/BSinfo.csv`（基站静态配置）
  - **用途**：提供静态硬件/无线配置，用于构造 **BS 级静态特征**，并为 Cell→BS 聚合提供 `Maximum_Trans_Power` 等权重。
  - **关键字段**：`BS`, `CellName`, `Frequency`, `Bandwidth`, `Antennas`, `Maximum_Trans_Power`, `ModeType`, `RUType`
  - **兼容说明**：部分数据版本字段可能写作 `maximum_transimission_power` 与 `Transimission_mode`（脚本已自动重命名兼容）。

### 3.1 `load_data()` 里的清洗与对齐口径

`load_data()` 做了三类关键处理：

- **类型转换与缺失处理**
  - `Time` 统一转 `datetime`（无法解析的置为 NaT 并被丢弃）
  - 能耗/负载/配置的数值列统一转 numeric（非法置为 NaN，并在关键列上 drop）

- **必要字段缺失行剔除**
  - `ec` 必须有 `Time/BS/Energy`
  - `cl` 必须有 `Time/BS/CellName/load`
  - `bsinfo` 必须有 `BS/CellName`

- **三表取交集的基站集合（保证口径一致）**
  - 只保留同时出现在 `ECdata/CLdata/BSinfo` 的 BS，避免后续 merge 后出现大量缺失或偏差。

**输入**：三张原始 DataFrame  
**输出**：清洗后的 `ec, cl, bsinfo`

---

## 4. 代码主流程总览（`main()`）

`main()` 是整个脚本的编排器，按顺序执行：

1. `ensure_dirs()`：创建输出目录
2. `load_data()`：读取并清洗数据
3. `build_bs_static_features(bsinfo)`：构造 BS 级静态特征表 `bs_feat`
4. `aggregate_cell_to_bs_time(cl, bsinfo)`：把 CL 从 Cell 粒度聚合到（BS，Time）得到 `bs_time`
5. `merged = ec.merge(bs_time, on=["BS","Time"])`：把能耗与聚合后的负载/ES 指标对齐
6. `define_windows(merged)`：在（BS，Time）粒度上定义“静态窗口”掩码
7. `estimate_pbase(merged, windows)`：在不同窗口下、用多种估计器估计 `P_base`
8. `summarize_windows(...) / summarize_method_stability(...)`：输出覆盖率、稳定性汇总表
9. `choose_target_combo(...)`：自动选择一个“窗口+估计方法”作为后续建模的监督目标
10. `train_models(...)`：用静态特征预测 `P_base`，输出模型性能与特征重要性
11. 生成图：方法分布、散点关系、窗口稳定性、窗口覆盖率
12. 写出所有 CSV
13. `build_report(...)`：生成 `analysis_report.md`
14. 覆盖写入 `README.md`（交付说明）

---

## 5. 逐函数说明（做什么 / 输入 / 输出）

### 5.1 输出目录管理

#### `ensure_dirs()`
- **做什么**：确保 `outputs/` 与 `outputs/figures_emf/` 存在。
- **输入**：无（使用脚本内常量 `OUTPUT_DIR/FIGURE_DIR`）。
- **输出**：无（副作用：创建目录）。

#### `save_current_figure_emf(filename)`
- **做什么**：把当前 Matplotlib 图保存为 EMF。
  - 先保存为 `SVG` 到 `outputs/figures_emf/`
  - 再调用 `inkscape --export-type=emf` 转换为 EMF
  - 转换成功后删除临时 SVG
- **输入**：
  - `filename`：用于生成输出文件名（实际上会取 `stem`，输出为 `stem.svg/stem.emf`）。
- **输出**：
  - `outputs/figures_emf/<stem>.emf`
  - 失败时抛出异常，并保留 SVG（便于排查）。

---

### 5.2 数据读取与清洗

#### `load_data() -> (ec, cl, bsinfo)`
- **做什么**：读取三张 CSV，转换类型、剔除关键缺失、并对齐 BS 口径（取交集）。
- **输入**：
  - `data/ECdata.csv`
  - `data/CLdata.csv`
  - `data/BSinfo.csv`
- **输出**：
  - `ec`：包含至少 `Time/BS/Energy`
  - `cl`：包含至少 `Time/BS/CellName/load/ESMode1..6`
  - `bsinfo`：包含至少 `BS/CellName/Frequency/Bandwidth/Antennas/Maximum_Trans_Power/ModeType/RUType`

---

### 5.3 构造 BS 级静态特征（监督学习输入 \(X\)）

#### `build_bs_static_features(bsinfo) -> feats`
- **做什么**：把 `BSinfo`（Cell 级静态配置）汇总到 **BS 级**，形成用于预测 `P_base` 的静态特征。
- **输入**：`bsinfo`（至少包含：`BS, CellName, Frequency, Bandwidth, Antennas, Maximum_Trans_Power, ModeType, RUType`）
- **核心特征构造**：
  - **规模类聚合**（按 `BS` groupby）：
    - `n_cells`：小区数量（`CellName.nunique`）
    - `sum_pmax`：最大发射功率总和（`Maximum_Trans_Power.sum`）
    - `sum_bandwidth`：带宽总和（`Bandwidth.sum`）
    - `sum_antennas`：天线数总和（`Antennas.sum`）
    - `mean_frequency`：平均频点（`Frequency.mean`）
    - `high_freq_ratio`：高频占比（`Frequency > 500` 的比例）
  - **组成比例特征**：
    - `mode_ratio_*`：不同 `ModeType` 在该 BS 内的占比（crosstab normalize）
    - `ru_ratio_*`：不同 `RUType` 在该 BS 内的占比
- **输出**：`feats`（BS 级特征表，主键 `BS`）

---

### 5.4 Cell→BS 的时间序列聚合（窗口定义输入）

#### `aggregate_cell_to_bs_time(cl, bsinfo) -> grouped`
- **做什么**：把 `CLdata` 从（BS，Cell，Time）聚合到（BS，Time），并构造用于“静态窗口”判断的聚合指标。
- **输入**：
  - `cl`：负载与节能模式（含 `ESMode1..6`）
  - `bsinfo`：用于补充每个 Cell 的 `Maximum_Trans_Power`（作为权重）
- **关键派生量**：
  - `es_abs_sum = |ESMode1..6|` 的行和：用来衡量该 Cell 在该时刻节能模式活动程度（越大越“非静态”）
  - `load_x_pmax = load * Maximum_Trans_Power`：用于功率加权负载
- **聚合输出字段（BS，Time 粒度）**：
  - `load_mean`：所有 Cell 平均负载
  - `load_pmax_weighted`：按 `Maximum_Trans_Power` 加权的负载平均（若分母为 0 则退化为 `load_mean`）
  - `ES_total`：所有 Cell 的 `es_abs_sum` 求和
  - `ES_max_abs`：所有 Cell 的 `es_abs_sum` 最大值
  - `n_cells_obs`：该时刻观测到的小区数
- **输出**：`grouped`（BS-时间聚合表）

---

### 5.5 静态窗口定义（标签提取的样本筛选）

#### `define_windows(df) -> windows`
- **做什么**：在（BS，Time）粒度上定义四种“近似静态运行窗口”，输出布尔掩码。
- **输入**：`df`（`merged` 后的数据，至少含 `BS, load_pmax_weighted, load_mean, ES_total, ES_max_abs`）
- **输出**：`windows: Dict[str, pd.Series]`，每个窗口一个 mask：
  - `A_strict_weighted`：`load_pmax_weighted < 0.05` 且 `ES_total≈0`
  - `B_mean_all_es_zero`：`load_mean < 0.10` 且 `ES_max_abs≈0`
  - `C_bs_low5pct`：`load_pmax_weighted <= 每BS 5%分位` 且 `ES_total≈0`
  - `D_relaxed`：`load_pmax_weighted < 0.15` 且 `ES_total≈0`

**理解要点**：窗口越严格，越接近“静态纯净”，但样本更少/覆盖 BS 更少；窗口越宽松，覆盖更高，但可能混入残余动态负载影响。

---

### 5.6 `P_base` 的多方法估计（监督学习标签 \(y\) 的候选集合）

#### `trimmed_mean(x, trim_ratio=0.1)`
- **做什么**：计算截尾均值（默认截尾比例 10%）。
- **输入**：`pd.Series`
- **输出**：`float`（可能为 NaN）

#### `estimate_pbase(df, windows) -> est`
- **做什么**：对每个窗口内的样本，按 BS 聚合 `Energy` 得到 `P_base`，并同时输出多种估计器结果用于比较。
- **输入**：
  - `df`：`merged`（含 `BS, Energy`）
  - `windows`：窗口掩码字典
- **估计器集合（每个 BS、每个 window 各算一遍）**：
  - `min`
  - `quantile_05` / `quantile_10`
  - `mean`
  - `median`
  - `trimmed_mean`
- **输出**：长表 `est`，字段：
  - `BS`, `window`, `method`, `p_base`, `n_samples`

---

### 5.7 覆盖率与稳定性汇总（用于选择最终标签构造规则）

#### `summarize_windows(df, windows) -> summary`
- **做什么**：统计每个窗口的样本数、样本占比、覆盖的 BS 数与覆盖率。
- **输入**：`merged` 与 `windows`
- **输出**：窗口汇总表（写到 `outputs/window_summary.csv`）

#### `summarize_method_stability(est) -> stability`
- **做什么**：衡量“同一估计方法在不同窗口下的离散程度”，越小越稳定。
  - 先把 `est` pivot 为：index=BS, columns=window, values=p_base
  - 对每个 BS 计算跨窗口 std，再汇总 mean/median
  - 同时给出 CV（std/mean）
- **输入**：`pbase_estimates_long`（即 `est`）
- **输出**：方法稳定性汇总表（写到 `outputs/method_stability_summary.csv`）

#### `choose_target_combo(est, n_total_bs) -> (chosen_window, chosen_method, combo_summary)`
- **做什么**：在所有（window, method）组合中，选择一个最适合做监督标签的规则。
- **选择依据（排序逻辑）**：
  - 首先最大化 `cover_ratio`（覆盖多少 BS）
  - 再最大化 `avg_n_samples`（平均样本数，越多越稳）
  - 最后用一个“方法偏好顺序”打破平局（优先分位数/中位数这类稳健估计器）
- **输入**：
  - `est`：长表估计结果
  - `n_total_bs`：总 BS 数
- **输出**：
  - `chosen_window`：最终选中的窗口名
  - `chosen_method`：最终选中的估计方法
  - `combo_summary`：各组合覆盖率与样本量汇总（写到 `outputs/target_combo_summary.csv`）

---

### 5.8 静态特征监督学习（预测 `P_base`）

#### `train_models(data, feature_cols, target_col) -> (metrics, preds, rf_feature_importance)`
- **做什么**：用 BS 级静态特征（`feature_cols`）回归 `P_base`（`target_col`），并比较多种模型。
- **输入**：
  - `data`：`model_dataset`（由 `bs_feat` 与最终 `target` merge 得到）
  - `feature_cols`：除 `BS/p_base` 外的全部特征列
  - `target_col`：固定为 `"p_base"`
- **训练/评估协议**：
  - `train_test_split(test_size=0.2, random_state=42)`
  - 指标：`R2` 与 `MAE`
- **模型集合**：
  - `LinearRegression`（数值特征：Median 填补 + 标准化）
  - `RidgeCV`（同上，alpha 网格）
  - `LassoCV`（同上，5-fold CV）
  - `RandomForestRegressor`（仅做 median 填补；不做标准化）
- **输出**：
  - `metrics`：模型指标表（写到 `outputs/model_metrics.csv`）
  - `preds`：测试集预测明细（写到 `outputs/model_predictions.csv`）
  - `rf_feature_importance`：随机森林特征重要性（写到 `outputs/rf_feature_importance.csv`）

---

### 5.9 图表输出（全部写到 `outputs/figures_emf/`）

#### `plot_method_distribution(est, window_name)`
- **做什么**：固定窗口下，各估计方法的 `P_base` 分布箱线图对比。
- **输入**：`est`、`window_name`
- **输出**：`fig_method_distribution.emf`

#### `plot_scatter_relations(model_dataset, target_col)`
- **做什么**：`P_base` 与 `sum_pmax`、`n_cells` 的散点关系，用于检查物理合理性（单调趋势）。
- **输入**：`model_dataset`、`target_col="p_base"`
- **输出**：`fig_scatter_pbase_vs_features.emf`

#### `plot_window_stability(est, method)`
- **做什么**：固定估计方法下，不同窗口得到的 `P_base` 分布箱线图（稳定性直观看）。
- **输入**：`est`、`method`
- **输出**：`fig_window_stability.emf`

#### `plot_window_coverage(summary)`
- **做什么**：各窗口的 BS 覆盖率柱状图。
- **输入**：`window_summary`
- **输出**：`fig_window_coverage.emf`

---

### 5.10 自动生成结果报告

#### `build_report(...)`
- **做什么**：把一次运行的关键统计量（覆盖率、稳定性、相关性、最佳模型指标、RF 重要性 TopN）拼成 Markdown 报告，并写入 `analysis_report.md`。
- **输入**：`merged/window_summary/method_stability/combo_summary/chosen_window/chosen_method/model_metrics/rf_imp/no_sample_df/model_dataset`
- **输出**：覆盖写入 `analysis_report.md`

---

## 6. 输出清单（脚本一次运行会写出什么）

### 6.1 CSV（`phaseA_static_energy_20260413/outputs/`）

- `bs_time_merged.csv`
  - **内容**：`ec` 与聚合后的 `bs_time` 合并后的（BS，Time）数据集（后续窗口筛选与能耗统计的底座）。
- `window_summary.csv`
  - **内容**：各窗口样本数与 BS 覆盖率。
- `pbase_estimates_long.csv`
  - **内容**：各窗口 + 各估计器 的 `P_base` 长表（`BS/window/method/p_base/n_samples`）。
- `method_stability_summary.csv`
  - **内容**：每种估计器跨窗口稳定性统计（mean std / median std / mean CV）。
- `target_combo_summary.csv`
  - **内容**：（window, method）组合的覆盖率与样本量汇总，便于解释“为什么选这个规则”。
- `model_dataset.csv`
  - **内容**：最终 BS 级建模数据（静态特征 + 目标 `p_base`）。
- `model_metrics.csv`
  - **内容**：各模型的 `R2/MAE`。
- `model_predictions.csv`
  - **内容**：测试集逐样本预测（含 `BS/model/y_true/y_pred`）。
- `rf_feature_importance.csv`
  - **内容**：随机森林特征重要性。
- `bs_without_static_samples.csv`
  - **内容**：各窗口下缺少静态样本的 BS 列表（用于后续“相似 BS 回填/迁移”扩展）。

### 6.2 图（`phaseA_static_energy_20260413/outputs/figures_emf/`）

- `fig_method_distribution.emf`
- `fig_scatter_pbase_vs_features.emf`
- `fig_window_stability.emf`
- `fig_window_coverage.emf`

### 6.3 文本报告（`phaseA_static_energy_20260413/`）

- `analysis_report.md`：自动生成的结果分析报告（每次运行会覆盖写入）。
- `README.md`：自动生成的交付说明（每次运行会覆盖写入）。

---

## 7. “输入输出”视角的最短路径理解

如果你只想快速抓住该阶段的 I/O 对应关系：

- **输入（原始）**：`ECdata.csv`（能耗）、`CLdata.csv`（负载/节能模式）、`BSinfo.csv`（静态配置）
- **中间关键表**：
  - `bs_time_merged.csv`：把能耗与负载/ES 聚合在一起（BS-时间）
  - `pbase_estimates_long.csv`：从静态窗口里抽取 `P_base`（多规则候选）
  - `model_dataset.csv`：静态特征 + 最终选定的 `P_base` 标签
- **输出（给后续阶段最关键）**：
  - `model_dataset.csv`：静态特征监督学习的数据基础
  - `outputs/model_metrics.csv`：静态特征是否足以解释 BS 间基础能耗差异

