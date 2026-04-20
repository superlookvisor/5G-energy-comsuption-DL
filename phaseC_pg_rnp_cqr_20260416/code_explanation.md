# 阶段C（`phaseC_pg_rnp_cqr_20260416`）代码说明：PG-RNP-CQR 残差日前预测

本文档解释 Phase C 的每个代码模块**做了什么**、**输入输出是什么**、以及主要产物文件的位置与含义。Phase C 的核心思想是：**保留 Phase B 的物理可解释基线作为 \(\hat E_{\text{phys}}\)**，只让神经过程模型学习“物理基线的残差”，并用 **按预测步长（horizon）分组的 CQR 校准**给出区间预测。

---

## 1. 任务定义（What / Why）

Phase C 学习如下分解：

- 物理基线（来自 Phase B）：
  - $\hat E_{\text{phys}} = p_{\text{base}} + \widehat{dynamic\_phys}$
- 残差真值：
  - $residual_{true} = energy_{true} - E^{phys}_{hat}$
- PG-RNP 输出残差分布参数（均值/尺度）：
  - `residual_mu`, `residual_sigma`
- 总能耗点预测：
  - `energy_pred = E_phys_hat + residual_mu`
- 区间预测（经 horizon-wise CQR 校准 + 物理边界投影）：
  - `lower_90`, `upper_90`

**关键点**：Phase B 的 `two_stage_proxy + Physical` 被视为“可解释/物理一致”的动态项基线；Phase C 只补上该基线难以拟合的部分（残差），并通过保形（conformal）校准控制不确定性覆盖率。

---

## 2. 目录结构与文件角色

本阶段主要文件：

- `run_phaseC_pg_rnp_cqr.py`
  - **作用**：一键运行 Phase C（构建数据集 → 划分 BS → 训练 PG-RNP → 推理 → CQR 校准 → 评估与写报告）。
  - **依赖**：训练需要 `torch`（PyTorch）。若环境缺少 torch 会直接退出并提示安装。

- `build_residual_trajectory_dataset.py`
  - **作用**：从 Phase B 输出目录读取 `panel_dataset.csv` 和 `dayahead_predictions.csv`，重建 two-stage 代理特征，拼出 Phase C 训练所需的“残差轨迹数据集”。
  - **默认 PhaseB 输入目录**：`phaseB_dayahead_only_20260415/outputs_filter_compare_learned`

- `evaluate_pg_rnp_cqr.py`
  - **作用**：提供评估与校准函数（horizon-wise CQR、overall/horizon 指标、物理边界违反统计、分析报告生成）。

此外，`run_phaseC_pg_rnp_cqr.py` 还会 import：

- `pg_rnp_model.py`
  - **作用**：PG-RNP 模型结构、训练、推理、数据标准化、按基站 split 等（该文件在本说明中按“接口行为”解释；若你希望逐行解释，我也可以再补一版）。

---

## 3. 运行方式（How to run）

从仓库根目录 `energy_model_anp/` 运行：

```bash
python phaseC_pg_rnp_cqr_20260416/run_phaseC_pg_rnp_cqr.py
```

建议的 smoke test（README 里给的参数组合）：

```bash
python phaseC_pg_rnp_cqr_20260416/run_phaseC_pg_rnp_cqr.py ^
  --epochs 1 ^
  --batch-size 4 ^
  --hidden-dim 32 ^
  --latent-dim 8 ^
  --max-rows 240 ^
  --output-dir phaseC_pg_rnp_cqr_20260416/outputs_smoke
```

---

## 4. Phase C 输入（Inputs）

Phase C **不直接读原始三表**（`ECdata/CLdata/BSinfo`），而是依赖 Phase B 的“已构建 panel 与预测结果”：

来自 `--phaseb-dir`（默认 `phaseB_dayahead_only_20260415/outputs_filter_compare_learned`）：

- `dayahead_predictions.csv`
  - **用途**：提供 Phase B 各策略各模型的预测；其中 Phase C 必须能找到：
    - `two_stage_proxy + Physical`：作为物理基线 `E_phys_hat`
    - `two_stage_proxy + RandomForest`：作为“强经验基线”对照（用于 ablation 汇总）
  - **关键列**：`BS, trajectory_id, origin_time, target_time, horizon, y_pred, energy_pred, energy_true, y_true, strategy, model`

- `panel_dataset.csv`
  - **用途**：提供构建 two-stage 代理特征所需的所有列（滚动负载、ES、静态硬件、时间编码等）。
  - **关键列**：`BS, Time, Energy, p_base, load_*`, `S_ESMode*`, `*_roll24`, `*_hour_prior`, 静态特征与 ratio 特征等。

- `dayahead_metrics.csv`
  - **用途**：Phase B 的最佳/对照信息会被写入 Phase C 的 `dataset_meta.json` 作为记录与溯源（不是训练必须项）。

---

## 5. Phase C 输出（Outputs）

输出目录由 `--output-dir` 指定，默认：

```text
phaseC_pg_rnp_cqr_20260416/outputs/
```

核心产物（README 列出的）：

- `residual_trajectory_dataset.csv`：Phase C 训练数据（每行对应一个 `(BS, origin_time, horizon)`）。
- `split_bs.json`：基站划分（train / calibration / test）。
- `feature_scaler.pkl`：特征标准化器（用于推理时一致变换）。
- `pg_rnp_predictions.csv`：PG-RNP + CQR 后的预测结果（点预测 + 区间）。
- `cqr_quantiles.json`：每个 horizon 的 CQR 分位数 \(q_h\)。
- `metrics_overall.csv`：总体指标（含 Phase B baselines + Phase C 模型行）。
- `metrics_by_horizon.csv`：按 horizon 的误差与覆盖率/宽度统计。
- `coverage_by_horizon.csv`：同 `metrics_by_horizon.csv` 的拷贝（便于按“覆盖率”命名引用）。
- `ablation_metrics.csv`：消融汇总（与 `metrics_overall.csv` 相同内容，强调对照）。
- `physics_violation_report.csv`：物理边界投影/违反情况统计。
- `analysis_report.md`：自动生成的文本总结报告。
- `run_meta.json`、`dataset_meta.json`：运行与数据集元信息（配置、残差统计、PhaseB 溯源等）。
- `phaseb_baseline_predictions.csv`：PhaseB 原始预测复制一份到 PhaseC 输出目录（便于后续独立分析）。

---

## 6. 逐文件/逐函数说明（做什么 / 输入 / 输出）

### 6.1 `run_phaseC_pg_rnp_cqr.py`（主入口）

#### (1) `parse_args()`
- **做什么**：解析 CLI 参数（PhaseB 输入目录、输出目录、训练超参、likelihood、device、smoke test 行数上限）。
- **输入**：命令行参数
- **输出**：`argparse.Namespace`

关键参数：
- `--phaseb-dir`：PhaseB 输出目录（含 `panel_dataset.csv`、`dayahead_predictions.csv` 等）
- `--output-dir`：PhaseC 输出目录
- `--epochs/--batch-size/--hidden-dim/--latent-dim`
- `--likelihood`：`student_t` 或 `gaussian`
- `--max-rows`：只保留部分 trajectory 用于 smoke test（按 24-slot 轨迹裁剪）
- `--device`：如 `cpu` / `cuda:0`

#### (2) `leakage_checks(df, feature_cols)`
- **做什么**：做三类一致性/泄漏检查，确保训练特征不包含“偷看答案”的列，且残差定义自洽：
  - **目标泄漏列禁止**：`Energy/energy_true/dynamic_energy/y_true/residual_true` 不允许出现在 `feature_cols`
  - **轨迹一致性**：`target_time - origin_time == horizon`
  - **残差一致性**：`energy_true - E_phys_hat == residual_true`
- **输入**：构建好的 dataset 与 feature 列集合
- **输出**：无；失败抛异常

#### (3) `main()`
按顺序完成：

1) `build_residual_dataset(phaseb_dir, output_dir)`
- 产出 `DatasetBuildResult`，包含 dataset、PhaseB baseline predictions、feature columns、metadata

2) （可选）`--max-rows`：按 trajectory 截断，写回裁剪后的 `residual_trajectory_dataset.csv`

3) `split_bs(df, config)`：按 BS 划分 train/calibration/test，并写出 `split_bs.json`

4) `train_model(...)`
- **输入**：dataset、feature_cols、splits、config、device
- **输出**：`model, scaler, residual_stats, history`
- 同时保存 `feature_scaler.pkl`

5) `predict_rows(...)`：对全部行推理得到 raw residual 分布参数

6) `horizonwise_cqr(raw_pred, alpha=0.10)`
- 得到按 horizon 的 \(q_h\)，生成区间并投影到物理边界
- 写出 `pg_rnp_predictions.csv` 与 `cqr_quantiles.json`

7) 评估与报告：
- `baseline_ablation_rows(...)`：把 PhaseB 各 (strategy, model) 在 test_bs 上的指标也算出来，便于对照
- `metric_row(...)`：给 PG-RNP 行生成 overall 指标
- `metrics_by_horizon(...)`：按 horizon 汇总误差 + 覆盖率 + 区间宽度
- `physics_violation_report(...)`：统计预测或区间超出物理边界的比例、投影修正幅度等
- `write_report(...)`：写 `analysis_report.md`
- 写 `run_meta.json`（config/残差统计/训练曲线尾部/数据集 meta）

---

### 6.2 `build_residual_trajectory_dataset.py`（数据构建器）

#### (1) 常量与默认值
- `PHASEB_DEFAULT_DIR`：默认 PhaseB 输入目录
- `BASELINE_STRATEGY="two_stage_proxy"`
- `PHYSICS_MODEL="Physical"`：用于构建 `E_phys_hat`
- `STRONG_BASELINE_MODEL="RandomForest"`：用于保存强基线预测列 `rf_energy_pred`

#### (2) `_load_phaseb_tables(phaseb_dir)`
- **输入**：PhaseB 输出目录
- **输出**：
  - `predictions`：读 `dayahead_predictions.csv`
  - `panel`：读 `panel_dataset.csv`
  - `metrics`：读 `dayahead_metrics.csv`

#### (3) `_build_two_stage_features(panel)`
- **做什么**：用 PhaseB rebuild 脚本复刻 two-stage 特征生成逻辑：
  - 调用 PhaseB rebuild 的 `build_dayahead_dataset(panel)` 得到 raw dataset
  - 再调用 `prepare_dayahead_features(..., "two_stage_proxy")` 得到 two-stage 特征表
  - 再把静态特征（如 `n_cells/sum_pmax/...`）以及 `mode_ratio_*/ru_ratio_*` 合并回 two_stage 表
- **输入**：PhaseB 保存的 `panel_dataset.csv`
- **输出**：PhaseC 需要的特征表（仍然是 row-level 的 `(BS, origin_time, horizon)` 视角）

#### (4) `build_residual_dataset(phaseb_dir, output_dir)`
- **做什么**：把 PhaseB 特征与 PhaseB 预测对齐到同一 key，构造 PhaseC 的“残差监督数据”。
- **关键对齐键**：`["BS", "trajectory_id", "origin_time", "target_time", "horizon"]`
- **构造步骤**：
  - 从 `dayahead_predictions.csv` 取出：
    - `two_stage_proxy + Physical` 的 `energy_pred/y_pred`，重命名为：
      - `E_phys_hat`（物理基线总能耗预测）
      - `dynamic_phys_hat`（物理基线动态项预测）
    - `two_stage_proxy + RandomForest` 的 `energy_pred`，重命名为 `rf_energy_pred`（可选对照列）
  - 将这些预测列 merge 到 two-stage 特征表上（inner join），确保每行都能得到物理基线
  - 计算残差监督标签：`residual_true = energy_true - E_phys_hat`
  - 标注轨迹完整性：
    - `is_complete_24`：该 `trajectory_id` 的 horizon 是否覆盖 24 个槽位（1..24）
  - 构造物理可行域（用于后续区间投影与违规统计）：
    - `physical_lower = 0.0`
    - `physical_upper = max(p_base + 2*sum_pmax + 10, E_phys_hat + 1.0)`（经验上界，保证至少包住物理基线）
  - 生成 feature 列清单 `feature_columns`（显式列 + `S_ESMode*`/`I_ESMode*`/ratio 前缀列）
- **输入**：PhaseB 输出目录（至少包含 `panel_dataset.csv`、`dayahead_predictions.csv`）
- **输出**：`DatasetBuildResult`
  - `dataset`：PhaseC 训练数据（并写 `residual_trajectory_dataset.csv`）
  - `baseline_predictions`：PhaseB 原始预测复制（写 `phaseb_baseline_predictions.csv`）
  - `feature_columns`：训练特征列清单（写入 `dataset_meta.json`）
  - `metadata`：数据规模、PhaseB 溯源信息等（写 `dataset_meta.json`）

---

### 6.3 `evaluate_pg_rnp_cqr.py`（评估与 CQR 校准）

#### (1) `horizonwise_cqr(pred, alpha=0.10)`
- **做什么**：在 calibration split 上，按 horizon 单独学习一个尺度化残差分位数 \(q_h\)，并生成 90% 区间（再投影到物理上下界）。
- **输入**：`pred`（至少包含 `split/horizon/residual_true/residual_mu/residual_sigma/E_phys_hat/physical_lower/physical_upper`）
- **核心计算**：
  - 校准分数：$\text{score}=\frac{|r-\mu|}{\max(\sigma,\epsilon)}$
  - 每个 \(h\) 取分位数 \(q_h\)（样本为空时用默认 1.645 兜底）
  - 区间（raw）：`E_phys_hat + residual_mu ± q_h * residual_sigma`
  - 区间投影：`lower=max(lower_raw, physical_lower)`，`upper=min(upper_raw, physical_upper)`，并修正 `lower<=upper`
- **输出**：
  - `pred` 增强后的 DataFrame（新增 `cqr_q/lower_90_raw/upper_90_raw/lower_90/upper_90/model_name`）
  - `q_by_h`（写 `cqr_quantiles.json`）

#### (2) `metric_row(df, model_name)`
- **做什么**：输出总体误差指标 + 严格 24h 轨迹指标（如果存在完整轨迹）。
- **输入**：某个模型在某 split 上的预测行（含 `energy_true/energy_pred`）
- **输出**：单行 dict（写入 `metrics_overall.csv` / `ablation_metrics.csv`）
  - `MAE/RMSE/MAPE`
  - `strict_24h_trajectory_MAPE`（仅对 `is_complete_24` 的 trajectory 求每轨迹平均 APE，再跨轨迹平均）
  - `peak_error/valley_error`（每条轨迹取真实峰/谷位置的绝对误差再平均）
  - `n_samples/n_bs/n_trajectories`

#### (3) `metrics_by_horizon(df)`
- **做什么**：按 horizon 汇总误差与区间性质：
  - 覆盖率：`coverage_90`
  - 平均宽度：`avg_width_90`
  - `mae/rmse/mape/n_samples/n_bs`
- **输出**：`metrics_by_horizon.csv`（同时拷贝为 `coverage_by_horizon.csv`）

#### (4) `baseline_ablation_rows(phaseb_predictions, test_bs)`
- **做什么**：把 PhaseB 原始 `dayahead_predictions.csv` 在 test_bs 子集上也算一遍指标，形成 ablation 对照表。
- **输出**：若干行 `PhaseB_{strategy}_{model}` 的 `metric_row` 结果。

#### (5) `physics_violation_report(pred)`
- **做什么**：统计“点预测/区间 raw 值”是否超出物理边界、以及投影修正幅度。
- **输出**：`physics_violation_report.csv`

#### (6) `write_report(...)`
- **做什么**：生成 `analysis_report.md`，汇总数据规模、最佳模型、PG-RNP 结果、平均覆盖率/宽度、以及输出清单。

---

## 7. 你最该先看哪些输出（建议阅读顺序）

- **(1) `analysis_report.md`**：快速确认这次运行用的 PhaseB 输入、数据规模、最佳 MAPE，以及 PG-RNP 的覆盖率/区间宽度概览。
- **(2) `ablation_metrics.csv`**：横向对比 PhaseB 各基线与 PhaseC（PG-RNP-CQR）的整体性能。
- **(3) `metrics_by_horizon.csv`**：看每个 horizon 的误差与 coverage 是否均衡（也能看到样本量随 horizon 变化）。
- **(4) `pg_rnp_predictions.csv`**：需要做进一步可视化/诊断时，从这里取 `lower_90/upper_90` 与 `energy_pred`。

---

## 8. `pg_rnp_model.py` 与 `plot_ieee_tsg_figures.py`（补充说明）

### 8.1 `pg_rnp_model.py`（PG-RNP 训练/推理核心）

该文件实现的是“按 BS-日期 episode 的 24 槽位轨迹建模”，并显式支持**mask 掉缺失 horizon**（因为很多 trajectory 不完整）。

核心设计点：

- **数据组织：`ResidualTrajectoryDataset`**
  - 一个 `__getitem__` 返回一条 episode（一个 `BS + origin_time`），其 target 布局固定为 24 个槽位：
    - `target_x`: shape=(24, d)
    - `target_y`: shape=(24,)
    - `target_mask`: shape=(24,) 表示该 horizon 是否存在样本（缺失则 mask=0）
  - 同时返回物理基线与可行域（用于 bound loss/后续投影）：
    - `e_phys`, `physical_lower`, `physical_upper`
  - `context_x/context_y` 来自**该 BS 的历史行**（`target_time < origin_time`），并受 `max_context` 限制；训练时会随机抽样 context 子集增强鲁棒性。

- **模型结构：`PGRNPModel`**
  - 把 context 聚合成 station-specific summary，再参数化 latent \(z\)（`z_mu/z_logvar`）
  - 用 Transformer encoder 在 24 槽位上建模时间结构
  - 输出残差分布参数：
    - `mu`（残差均值，已在 residual 标准化空间）
    - `scale`（残差尺度，softplus 保证正）
  - likelihood 可选 `gaussian` 或 `student_t`（`student_t` 用可学习自由度 `nu()`）

- **损失函数：`compute_loss()`**
  - NLL（mask 后对可用 horizon 求均值）
  - KL 正则（latent 贴近标准正态）
  - **bound loss**：把残差反标定回能耗后，惩罚 `energy_pred` 落在物理区间外
  - **smooth loss**：相邻 horizon 的残差均值变化过大时惩罚（仅对相邻都可用的槽位生效）

- **训练与推理**
  - `train_model()`：早停（patience）、保存 `pg_rnp_model.pt`、`residual_scaler.pt`、`training_config/history.json`
  - `predict_rows()`：输出逐行的 `residual_mu/residual_sigma` 并回 merge 到原始 dataset；再给出点预测 `energy_pred`

### 8.2 `plot_ieee_tsg_figures.py`（生成 IEEE TSG 风格图）

该脚本**只消费 PhaseC 的输出表**，生成 `outputs/ieee_tsg_figures/` 下的 `png+pdf`：

- **输入**（来自某个 PhaseC 输出目录）：
  - `metrics_overall.csv`
  - `coverage_by_horizon.csv`
  - `pg_rnp_predictions.csv`
  - `physics_violation_report.csv`
  - `phaseb_baseline_predictions.csv`
- **输出**（默认 `outputs/ieee_tsg_figures/`）：
  - `fig01_overall_point_accuracy.(png|pdf)`：总体 MAE/MAPE 柱状对比（含 PhaseB 与 PG-RNP-CQR）
  - `fig02_horizon_mae_coverage.(png|pdf)`：按 horizon 的 MAE 与 90% 覆盖率双轴图
  - `fig03_interval_width_by_horizon.(png|pdf)`：区间宽度随 horizon 的变化
  - `fig04_representative_uncertainty_trajectory.(png|pdf)`：代表性轨迹（含区间带；可叠加 Two-stage RF）
  - `fig05_prediction_scatter.(png|pdf)`：测试集预测散点
  - `fig06_residual_distribution.(png|pdf)`：残差分布直方图 + 经验 CDF
  - `fig07_esmode_active_error.(png|pdf)`：部分 ES 模式激活/未激活下 MAE 对比（若列存在且样本足够）
  - `fig08_physical_violation_rates.(png|pdf)`：投影前 raw 区间的物理边界违规率
  - `figure_manifest.json`：图与关键指标的摘要（便于论文引用）


