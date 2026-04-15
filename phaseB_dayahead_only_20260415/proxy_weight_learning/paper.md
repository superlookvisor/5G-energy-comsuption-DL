## X.X 多源代理权重的约束学习（Constrained Stacking for Day-Ahead Covariates）

### X.X.1 动机与问题定义

日前预测场景下，目标时刻 $t_0+h$ 的真实负载与节能模式强度通常不可得。为避免使用经验设定的固定权重带来“权重来源不可解释”的问题，本文将负载与 ES 的多源代理构造形式化为一个**受约束的 stacking/凸组合学习问题**：在不使用未来能耗标签 `Energy` 的前提下，仅利用**目标时刻真实 covariates**作为监督信号，学习权重 $w$ 以最小化代理误差，并将学习到的权重用于后续两阶段代理（two-stage proxy）的 `*_hat` 特征生成。

### X.X.2 候选信息源与代理形式

以基站 $b$、起报时刻 $t_0$（00:00）和步长 $h\in\{1,\dots,24\}$ 为例，本文定义如下候选信息源（均可由历史面板在起报时刻构造）：

- **昨日同小时项**（prevday same hour）：$x^{(D)}_{b,h}$
- **滚动 24 小时统计项**（roll24）：$x^{(R)}_{b}$
- **起报时刻当前项**（current）：$x^{(C)}_{b}$
- **同小时历史先验**（hour prior，用于 ES）：$x^{(P)}_{b,\,hour(t_0)}$

并采用凸组合得到目标时刻 covariate 的代理：

- **负载均值代理**：

$
\widehat{load\_mean}_{b,h}
= w^{(lm)}_D x^{(D)}_{b,h}+ w^{(lm)}_R x^{(R)}_{b}+ w^{(lm)}_C x^{(C)}_{b},
\quad
w^{(lm)}\ge 0,\ \mathbf{1}^\top w^{(lm)}=1.
$

- **负载加权代理**（与 `load_pmax_weighted` 对齐）：

$
\widehat{load\_pmax}_{b,h}
= w^{(lp)}_D x^{(D)}_{b,h}+ w^{(lp)}_R x^{(R)}_{b}+ w^{(lp)}_C x^{(C)}_{b},
\quad
w^{(lp)}\ge 0,\ \mathbf{1}^\top w^{(lp)}=1.
$

- **负载波动代理**：

$
\widehat{load\_std}_{b,h}
= w^{(ls)}_D x^{(D)}_{b,h}+ w^{(ls)}_R x^{(R)}_{b},
\quad
w^{(ls)}\ge 0,\ \mathbf{1}^\top w^{(ls)}=1.
$

- **ES 模式强度代理（跨模式共享一组权重）**：

$
\widehat{S}_{m,b,h}
= w^{(s)}_D x^{(D)}_{m,b,h}+ w^{(s)}_P x^{(P)}_{m,b,\,hour(t_0)},
\quad
w^{(s)}\ge 0,\ \mathbf{1}^\top w^{(s)}=1.
$

在得到 $\widehat{load\_pmax}_{b,h}$ 与 $\widehat{S}_{m,b,h}$ 后，本文按两阶段代理策略构造物理解释量：
$
\widehat{D1}_{b,h}=sum\_pmax_b\cdot \widehat{load\_pmax}_{b,h},\quad
\widehat{D2}_{b,h}=sum\_pmax_b\cdot \widehat{load\_pmax}_{b,h}^2,\quad
\widehat{D3}_{b,h}=\widehat{load\_std}_{b,h},
$
以及交互项 $\widehat{I}_{m,b,h}=\widehat{S}_{m,b,h}\cdot \widehat{load\_pmax}_{b,h}$。

> 注：在实现中为保证鲁棒性，对缺失项采用与基线一致的回退链（例如 prevday 缺失回退至 roll24，再回退至 current）。

### X.X.3 约束学习目标与避免数据泄漏

**监督信号**选择为目标时刻真实 covariates（不使用未来 `Energy`），例如
$\,load\_mean(b,t_0+h)$、$load\_pmax\_weighted(b,t_0+h)$、$load\_std(b,t_0+h)$ 以及 $S_m(b,t_0+h)$。
对应的权重学习为：

$
\min_{w\in\Delta}
\frac{1}{N}\sum_{i=1}^{N}\left(y_i-\sum_{k}w_k x_{i,k}\right)^2 + \lambda \|w\|_2^2,
\quad
\Delta=\{w\mid w\ge 0,\ \mathbf{1}^\top w = 1\}.
$

为严格避免同站泄漏，本文按 `BS` 分组进行随机拆分（训练比例 `train_bs_frac=0.8`），仅使用训练基站集合构造样本并拟合权重；同时对每个步长 $h\in\{1,\dots,24\}$ 仅保留满足严格整小时间隔的样本对（与连续性校验一致）。本实验在过滤后数据集上共有 `n_bs_total=818` 个基站，其中训练基站 `n_bs_train=654`、验证基站 `n_bs_valid=164`；用于负载权重拟合的有效样本数为 `n=5873`，用于 ES 权重拟合的有效样本数为 `n=28284`（见 `outputs_filter/proxy_weights_meta.json`）。

### X.X.4 优化实现与学习结果

为保证约束始终满足，本文采用 softmax 参数化 $w=\mathrm{softmax}(a)$，从而天然满足 $w\ge0$ 且 $\sum w=1$，并使用 L-BFGS-B 优化自由变量 $a$。学习到的全局权重（`outputs_filter/proxy_weights.json`）如下：

- `load_mean`（prevday, roll24, current）：$[0.0016,\ 0.9556,\ 0.0428]$
- `load_pmax`（prevday, roll24, current）：$[0.0025,\ 0.9486,\ 0.0489]$
- `load_std`（prevday, roll24）：$[0.8391,\ 0.1609]$
- `s_hat`（prevday, hour_prior）：$[0.3082,\ 0.6918]$

从结果可见：在该数据集上，`roll24` 对负载代理贡献显著（约 0.95），而 ES 代理更依赖同小时历史先验（约 0.69）。该结论为后续能耗模型输入提供了清晰解释：负载侧更受近期统计稳定性驱动，而节能模式侧更体现时段习惯性（hour-specific prior）。此外，代理拟合误差（RMSE/MAE）已在 `proxy_weights_meta.json` 中报告，可作为方法复现与附录材料。

> 图/表插入建议：可在此小节末插入一张“学习到的权重柱状图”（将四组权重可视化），并在附录给出 `proxy_weights_meta.json` 的拟合误差表。

### X.X.5 与日前能耗模型的衔接

学习到的权重用于替换 two-stage proxy 中 `load_*_hat` 与 `S_*_hat` 的固定权重，从而使下游日前能耗模型的输入构造过程从“经验加权”升级为“受约束可学习的 stacking 代理”。在实现层面，权重学习与应用通过命令行开关控制（`--learn-proxy-weights` 或 `--proxy-weights-json`），并与过滤版输出目录一致地保存在 `outputs_filter/`，确保主实验可完全复现（见 `filter_meta.json` 中 `proxy_weight_mode=learned` 标记）。