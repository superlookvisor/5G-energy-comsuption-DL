# 装机容量调整方法三：基站级差异化容量配置（Method 3, Per-BS Sizing）

## 方法说明
- 为每座基站 b 独立选择最小容量倍率 χ_b，使得该基站自身的可靠性违约率不超过阈值 ε_bs。
- 基站装机容量为 \(C_b=\chi_b\cdot p_b^{\mathrm{base}}\)；总装机容量为 \(\sum_b \chi_b\cdot p_b^{\mathrm{base}}\)。
- χ 网格：`4,6,8,10,12,14,16,18,20,24`
- 可靠性目标 R_min：`0.99`；基站违约阈值 ε_bs：`0.01`；全网违约阈值（基线）：`0.01`

## 基线：统一容量倍率 χ★（Method 1 逻辑）
- `Full BESS Backup`：χ★ = `14.00`
- `Hourly Upper Reliability`：χ★ = `None`
- `Oracle Reliability`：χ★ = `14.00`
- `Phase C Point Reliability`：χ★ = `None`
- `Reliability-Aware CQR`：χ★ = `None`

## 总装机容量对比

| policy | n_bs | n_bs_feasible | n_bs_capped_at_grid_max | uniform_chi_star | uniform_total_installed_capacity | uniform_total_installed_capacity_at_grid_max | bs_level_total_installed_capacity | capacity_savings_vs_uniform_pct | capacity_savings_vs_grid_max_pct | mean_chi_b_star | median_chi_b_star | min_chi_b_star | max_chi_b_star | sum_p_base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full BESS Backup | 53 | 53 | 0 | 14.0000 | 16696.9872 | 28623.4066 | 10509.9694 | 37.0547 | 63.2819 | 8.7925 | 8.0000 | 6.0000 | 14.0000 | 1192.6419 |
| Hourly Upper Reliability | 53 | 49 | 4 | nan | nan | 28623.4066 | 11546.4417 | nan | 59.6608 | 8.6531 | 8.0000 | 6.0000 | 12.0000 | 1192.6419 |
| Oracle Reliability | 53 | 53 | 0 | 14.0000 | 16696.9872 | 28623.4066 | 10509.9694 | 37.0547 | 63.2819 | 8.7925 | 8.0000 | 6.0000 | 14.0000 | 1192.6419 |
| Phase C Point Reliability | 53 | 15 | 38 | nan | nan | 28623.4066 | 23467.3531 | nan | 18.0134 | 8.9333 | 8.0000 | 8.0000 | 12.0000 | 1192.6419 |
| Reliability-Aware CQR | 53 | 45 | 8 | nan | nan | 28623.4066 | 13255.0362 | nan | 53.6916 | 8.4000 | 8.0000 | 6.0000 | 12.0000 | 1192.6419 |

## 基站级 χ_b★ 统计（前 20 条，按策略→BS 排序）

| policy | BS | p_base | chi_b_star | feasible | mean_reliability_at_chi_b | violation_at_chi_b | dispatchable_at_chi_b | installed_capacity_bs_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full BESS Backup | B_1 | 16.0389 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 128.3109 |
| Full BESS Backup | B_112 | 52.5859 | 10.0000 | True | 0.9948 | 0.0000 | 0.0000 | 525.8595 |
| Full BESS Backup | B_130 | 11.3004 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 90.4036 |
| Full BESS Backup | B_157 | 17.4888 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 139.9103 |
| Full BESS Backup | B_169 | 18.3857 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 147.0852 |
| Full BESS Backup | B_192 | 18.2063 | 10.0000 | True | 0.9951 | 0.0000 | 0.0000 | 182.0628 |
| Full BESS Backup | B_20 | 19.8804 | 6.0000 | True | 0.9951 | 0.0000 | 0.0000 | 119.2825 |
| Full BESS Backup | B_202 | 17.1898 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 137.5187 |
| Full BESS Backup | B_260 | 31.5396 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 252.3169 |
| Full BESS Backup | B_27 | 15.9940 | 12.0000 | True | 0.9938 | 0.0000 | 0.0000 | 191.9283 |
| Full BESS Backup | B_278 | 27.4888 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 219.9103 |
| Full BESS Backup | B_301 | 38.6398 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 309.1181 |
| Full BESS Backup | B_353 | 28.1016 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 224.8132 |
| Full BESS Backup | B_355 | 35.2429 | 12.0000 | True | 0.9949 | 0.0000 | 0.0000 | 422.9144 |
| Full BESS Backup | B_362 | 44.9925 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 359.9402 |
| Full BESS Backup | B_364 | 14.4993 | 6.0000 | True | 0.9926 | 0.0000 | 0.0000 | 86.9955 |
| Full BESS Backup | B_372 | 10.6129 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 84.9028 |
| Full BESS Backup | B_376 | 11.3602 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 90.8819 |
| Full BESS Backup | B_379 | 18.6099 | 10.0000 | True | 0.9954 | 0.0000 | 0.0000 | 186.0987 |
| Full BESS Backup | B_401 | 17.3543 | 8.0000 | True | 0.9926 | 0.0000 | 0.0000 | 138.8341 |

> 完整 χ_b★ 明细参见 `bs_level_chi_star.csv`；每 χ、每 BS 的原始指标参见 `bs_level_per_bs_metrics.csv`。