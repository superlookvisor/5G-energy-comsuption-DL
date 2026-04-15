## 日前多源代理权重学习（全局）

本子目录用于**学习并保存**日前预测中“多源代理（prevday / roll24 / current 等）”的组合权重，
用于替代手工固定权重（例如 `0.55/0.30/0.15`），以提升可解释性与可复现性。

### 目标

- 在不使用未来 `Energy` 的前提下，仅用目标时刻真实 covariates 监督，学习代理权重：
  - `load_mean_hat`：由 `load_mean_prevday_samehour`、`load_mean_roll24`、`load_mean` 线性凸组合
  - `load_pmax_hat`：由 `load_pmax_prevday_samehour`、`load_pmax_roll24`、`load_pmax_weighted` 线性凸组合
  - `load_std_hat`：由 `load_std_prevday_samehour`、`load_std_roll24` 线性凸组合
  - `S_*_hat`：由 `S_*_prevday_samehour`、`S_*_hour_prior` 线性凸组合
- 约束：权重满足 \(w \\ge 0\) 且 \(\nobreak\\sum w = 1\)（凸组合）。

### 输出

- `proxy_weights.json`：学习得到的权重（全局常数）
- `proxy_weights_meta.json`：训练/验证拆分、样本量、拟合误差等元信息

### 使用位置

`phaseB_dayahead_only_20260415/run_phaseB_dayahead_only.py` 将在启用 `--learn-proxy-weights` 时：
1) 学习权重并保存到本目录；
2) 以学习到的权重生成 `*_hat` 特征并进入日前模型训练与评估。

