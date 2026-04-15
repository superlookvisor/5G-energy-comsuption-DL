# Phase B 仅日前

主脚本：`run_phaseB_dayahead_only.py`  
输出：无过滤时写入 `outputs/`；使用 `--min-merged-obs-per-bs` 或 `--exclude-bs-csv` 时写入 `outputs_filter/`，与全量结果并存。

## 一键运行（不做过滤）

如果你在**仓库根目录**（`energy_model_anp/`）运行：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py
```

如果你在**本目录**（`phaseB_dayahead_only_20260415/`）运行，或在编辑器里**直接点运行按钮**（cwd 通常就是当前文件所在目录），请用：

```bash
python run_phaseB_dayahead_only.py
```

### 常见提示/警告

- **sklearn ConvergenceWarning**：运行时可能会看到 `ConvergenceWarning: Objective did not converge`（线性模型优化未完全收敛）。这通常**不影响脚本产出**（仍会生成指标、图与报告）；如需消除，可在后续再考虑做特征缩放/增加迭代次数/调整正则化（属于模型质量优化，不是运行失败）。
- **Windows 控制台中文乱码**：若命令行里中文输出显示为乱码，可在 PowerShell 里先执行：

```bash
$env:PYTHONUTF8=1
```

## 运行（启用 BS 过滤）

（剔除稀疏 QC 中少于 24 观测的站，结果在 `outputs_filter/`）：

在**仓库根目录**运行：

```bash
python run_phaseB_dayahead_only.py 
\--min-merged-obs-per-bs 24 
\--exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv
```


## 代理权重学习（stacking）

学习并应用多源代理权重（满足 `w>=0` 且 `sum(w)=1`），用于替换 two-stage proxy 的固定权重：

```bash
python run_phaseB_dayahead_only.py 
\--min-merged-obs-per-bs 24 
\--exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv 
\--learn-proxy-weights
```

复用既有权重：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv \
  --proxy-weights-json phaseB_dayahead_only_20260415/outputs_filter/proxy_weights.json
```

为避免覆盖输出，可指定输出子目录（相对 `phaseB_dayahead_only_20260415/`）：

```bash
python phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py \
  --min-merged-obs-per-bs 24 \
  --exclude-bs-csv phase_sparse_observed_20260415/outputs/bs_below_hour_threshold.csv \
  --learn-proxy-weights \
  --output-subdir outputs_filter_learned
```
