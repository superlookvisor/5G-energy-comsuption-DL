# Phase B（仅日前预测）

本目录用于**日前（day-ahead）**能耗预测实验：以每天 `00:00` 作为 `origin_time`，预测未来 \(h\in[1,24]\) 小时的目标时刻 `target_time` 的 `dynamic_energy` / `Energy`。

- **主脚本**：`phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py`
- **关键概念**
  - **horizon**：预测提前量（小时）。例如 `horizon=6` 表示从 `origin_time` 预测 `origin_time+6h`。
  - **prevday_samehour**：按时间戳对齐的“前一天同小时”（相对 `target_time`）：`target_time - 24h`。
  - **回退链**：`prevday -> roll24 -> current`（缺失时依次回退）。

---

## 1. 运行前检查

- **工作目录**：在仓库根目录 `energy_model_anp/` 下运行命令。
- **数据**：脚本会读取 `data/` 下的原始 CSV（与 PhaseB rebuild 保持一致）。
- **依赖**：Python 环境需包含脚本依赖（numpy/pandas/sklearn/matplotlib 等；若启用权重学习还需要 scipy）。

---

## 2. 一键运行（不做过滤）

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py
```

默认输出到：`phaseB_dayahead_only_20260415/outputs/`

---

## 3. 运行（启用 BS 过滤）

常用过滤参数：
- `--min-merged-obs-per-bs`：每个 BS 最少保留的 merged 观测行数
- `--exclude-bs-csv`：显式剔除的 BS 列表（CSV 需包含列 `BS`）

示例（剔除稀疏 QC 中少于 24 观测的站）：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv
```

启用任一过滤参数后，默认输出到：`phaseB_dayahead_only_20260415/outputs_filter/`

---

## 4. 代理权重学习与传参（stacking）

two-stage proxy 中 `*_hat` 的组合权重可以：
- **固定权重**（默认）：使用脚本内置的常数权重
- **学习权重**：在训练 BS 上学习全局权重 \(w\ge 0,\ \sum w=1\)，并应用到 `*_hat`
- **加载权重**：直接读取已有 `proxy_weights.json`

### 4.1 学习并应用权重

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --learn-proxy-weights
```

若同时启用过滤：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv \
  --learn-proxy-weights
```

脚本会在输出目录中写入：
- `proxy_weights.json`：学习到的权重
- `proxy_weights_meta.json`：学习过程与拟合误差统计

### 4.2 加载已有权重（传入权重参数）

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --proxy-weights-json phaseB_dayahead_only_20260415/outputs_filter/proxy_weights.json
```

注意：`--learn-proxy-weights` 与 `--proxy-weights-json` **不能同时使用**。

### 4.3 指定输出子目录（避免覆盖）

`--output-subdir` 是相对 `phaseB_dayahead_only_20260415/` 的子目录名：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --learn-proxy-weights \
  --output-subdir outputs_filter_learned
```

---

## 5. 输出结果说明

输出目录（例如 `outputs/` 或 `outputs_filter/`）中常见文件：

- **核心表格**
  - `panel_dataset.csv`：用于训练/评估的 panel 特征数据（含 roll24、hour_prior 等）
  - `dayahead_predictions.csv`：各模型/策略在每个 `(trajectory_id, horizon)` 上的预测与真值
  - `dayahead_metrics.csv`：严格轨迹口径（完整 24 小时轨迹）的汇总指标
  - `dayahead_horizon_metrics.csv`：按 `horizon` 的误差指标（MAE/RMSE/MAPE 等）
  - `model_coefficients.csv`：线性模型系数/特征重要性导出（不同模型会有所差异）
  - `physics_checks.csv`：物理一致性/约束相关检查结果
  - `es_mode_effects.csv`：ES 模式影响分析结果
  - `filter_meta.json`：过滤条件、输出模式（fixed/learned/loaded）等元信息

- **图表**
  - `fig_prediction_vs_actual.png`：预测 vs 真值散点/对比图
  - `fig_error_by_horizon.png`：误差随 `horizon` 的变化
  - `fig_dayahead_trajectory.png`：代表性轨迹的预测曲线
  - `fig_load_vs_energy.png`：负载与能耗关系图
  - `fig_es_mode_impact.png`：ES 模式影响可视化

- **报告**
  - `analysis_report.md`：自动生成的文字报告（包含关键指标与结论摘要）

---

## 6.（可选）回退链触发比例统计

若你想量化 `prevday/roll24/current` 在不同 `horizon` 下的实际占比，可运行：

```bash
python phaseB_dayahead_only_20260415/fallback_audit/analyze_fallback_usage.py \
  --panel phaseB_dynamic_energy_20260414_rebuild/outputs/panel_dataset.csv \
  --outdir phaseB_dayahead_only_20260415/outputs_fallback_audit
```

输出：
- `outputs_fallback_audit/analysis_report.md`
- `outputs_fallback_audit/fallback_summary_by_horizon.csv`
