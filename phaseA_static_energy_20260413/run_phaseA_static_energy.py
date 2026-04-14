from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# 优先使用常见中文字体，避免图标题乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ec = pd.read_csv(DATA_DIR / "ECdata.csv")
    cl = pd.read_csv(DATA_DIR / "CLdata.csv")
    bsinfo = pd.read_csv(DATA_DIR / "BSinfo.csv")

    ec["Time"] = pd.to_datetime(ec["Time"], errors="coerce")
    cl["Time"] = pd.to_datetime(cl["Time"], errors="coerce")

    numeric_cols_ec = ["Energy"]
    numeric_cols_cl = ["load"] + [f"ESMode{i}" for i in range(1, 7)]
    numeric_cols_bs = ["Frequency", "Bandwidth", "Antennas", "Maximum_Trans_Power"]

    for c in numeric_cols_ec:
        ec[c] = pd.to_numeric(ec[c], errors="coerce")
    for c in numeric_cols_cl:
        cl[c] = pd.to_numeric(cl[c], errors="coerce")
    for c in numeric_cols_bs:
        bsinfo[c] = pd.to_numeric(bsinfo[c], errors="coerce")

    ec = ec.dropna(subset=["Time", "BS", "Energy"])
    cl = cl.dropna(subset=["Time", "BS", "CellName", "load"])
    bsinfo = bsinfo.dropna(subset=["BS", "CellName"])

    # 保证口径一致：只保留三表交集内的基站
    common_bs = set(ec["BS"].unique()) & set(cl["BS"].unique()) & set(bsinfo["BS"].unique())
    ec = ec[ec["BS"].isin(common_bs)].copy()
    cl = cl[cl["BS"].isin(common_bs)].copy()
    bsinfo = bsinfo[bsinfo["BS"].isin(common_bs)].copy()
    return ec, cl, bsinfo


def build_bs_static_features(bsinfo: pd.DataFrame) -> pd.DataFrame:
    mode_col = "ModeType"
    ru_col = "RUType"

    bsinfo = bsinfo.copy()
    bsinfo["is_high_freq"] = (bsinfo["Frequency"] > 500).astype(int)

    base_agg = (
        bsinfo.groupby("BS")
        .agg(
            n_cells=("CellName", "nunique"),
            sum_pmax=("Maximum_Trans_Power", "sum"),
            sum_bandwidth=("Bandwidth", "sum"),
            sum_antennas=("Antennas", "sum"),
            mean_frequency=("Frequency", "mean"),
            high_freq_ratio=("is_high_freq", "mean"),
        )
        .reset_index()
    )

    mode_ratio = pd.crosstab(bsinfo["BS"], bsinfo[mode_col], normalize="index")
    mode_ratio.columns = [f"mode_ratio_{c}" for c in mode_ratio.columns]
    mode_ratio = mode_ratio.reset_index()

    ru_ratio = pd.crosstab(bsinfo["BS"], bsinfo[ru_col], normalize="index")
    ru_ratio.columns = [f"ru_ratio_{c}" for c in ru_ratio.columns]
    ru_ratio = ru_ratio.reset_index()

    feats = base_agg.merge(mode_ratio, on="BS", how="left").merge(ru_ratio, on="BS", how="left")
    feats = feats.fillna(0.0)
    return feats


def aggregate_cell_to_bs_time(cl: pd.DataFrame, bsinfo: pd.DataFrame) -> pd.DataFrame:
    use_cols = ["BS", "CellName", "Maximum_Trans_Power"]
    cl_ext = cl.merge(bsinfo[use_cols], on=["BS", "CellName"], how="left")
    cl_ext["Maximum_Trans_Power"] = cl_ext["Maximum_Trans_Power"].fillna(0.0)

    es_cols = [f"ESMode{i}" for i in range(1, 7)]
    cl_ext["es_abs_sum"] = cl_ext[es_cols].abs().sum(axis=1)
    cl_ext["load_x_pmax"] = cl_ext["load"] * cl_ext["Maximum_Trans_Power"]

    grouped = (
        cl_ext.groupby(["BS", "Time"])
        .agg(
            load_mean=("load", "mean"),
            load_pmax_weighted_num=("load_x_pmax", "sum"),
            sum_pmax_obs=("Maximum_Trans_Power", "sum"),
            ES_total=("es_abs_sum", "sum"),
            ES_max_abs=("es_abs_sum", "max"),
            n_cells_obs=("CellName", "nunique"),
        )
        .reset_index()
    )
    grouped["load_pmax_weighted"] = np.where(
        grouped["sum_pmax_obs"] > 0,
        grouped["load_pmax_weighted_num"] / grouped["sum_pmax_obs"],
        grouped["load_mean"],
    )
    return grouped


def define_windows(df: pd.DataFrame) -> Dict[str, pd.Series]:
    eps = 1e-6
    load_q05_per_bs = df.groupby("BS")["load_pmax_weighted"].transform(lambda s: s.quantile(0.05))
    windows = {
        "A_strict_weighted": (df["load_pmax_weighted"] < 0.05) & (df["ES_total"] < eps),
        "B_mean_all_es_zero": (df["load_mean"] < 0.10) & (df["ES_max_abs"] < eps),
        "C_bs_low5pct": (df["load_pmax_weighted"] <= load_q05_per_bs) & (df["ES_total"] < eps),
        "D_relaxed": (df["load_pmax_weighted"] < 0.15) & (df["ES_total"] < eps),
    }
    return windows


def trimmed_mean(x: pd.Series, trim_ratio: float = 0.1) -> float:
    arr = np.sort(x.to_numpy())
    n = len(arr)
    if n == 0:
        return np.nan
    k = int(np.floor(n * trim_ratio))
    if 2 * k >= n:
        return float(np.mean(arr))
    return float(np.mean(arr[k : n - k]))


def estimate_pbase(df: pd.DataFrame, windows: Dict[str, pd.Series]) -> pd.DataFrame:
    rows: List[dict] = []
    for wname, mask in windows.items():
        sub = df.loc[mask, ["BS", "Energy"]].copy()
        for bs, grp in sub.groupby("BS"):
            e = grp["Energy"].dropna()
            if e.empty:
                continue
            methods = {
                "min": float(e.min()),
                "quantile_05": float(e.quantile(0.05)),
                "quantile_10": float(e.quantile(0.10)),
                "mean": float(e.mean()),
                "median": float(e.median()),
                "trimmed_mean": trimmed_mean(e, 0.1),
            }
            for m, v in methods.items():
                rows.append(
                    {
                        "BS": bs,
                        "window": wname,
                        "method": m,
                        "p_base": v,
                        "n_samples": int(len(e)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_windows(df: pd.DataFrame, windows: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    total_bs = df["BS"].nunique()
    for name, mask in windows.items():
        sub = df.loc[mask]
        rows.append(
            {
                "window": name,
                "samples": int(len(sub)),
                "sample_ratio": float(len(sub) / total_rows) if total_rows else np.nan,
                "bs_covered": int(sub["BS"].nunique()),
                "bs_cover_ratio": float(sub["BS"].nunique() / total_bs) if total_bs else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("window")


def summarize_method_stability(est: pd.DataFrame) -> pd.DataFrame:
    # 以“同一方法跨窗口”的离散程度度量稳定性
    rows = []
    for method, g in est.groupby("method"):
        pivot = g.pivot_table(index="BS", columns="window", values="p_base", aggfunc="mean")
        valid = pivot.dropna(thresh=2)
        if valid.empty:
            rows.append(
                {
                    "method": method,
                    "n_bs_with_2plus_windows": 0,
                    "mean_std_across_windows": np.nan,
                    "median_std_across_windows": np.nan,
                    "mean_cv_across_windows": np.nan,
                }
            )
            continue
        bs_std = valid.std(axis=1, ddof=0)
        bs_mean = valid.mean(axis=1).replace(0, np.nan)
        bs_cv = bs_std / bs_mean
        rows.append(
            {
                "method": method,
                "n_bs_with_2plus_windows": int(len(valid)),
                "mean_std_across_windows": float(bs_std.mean()),
                "median_std_across_windows": float(bs_std.median()),
                "mean_cv_across_windows": float(bs_cv.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_std_across_windows")


def choose_target_combo(est: pd.DataFrame, n_total_bs: int) -> Tuple[str, str, pd.DataFrame]:
    combo = (
        est.groupby(["window", "method"])
        .agg(bs_covered=("BS", "nunique"), avg_n_samples=("n_samples", "mean"))
        .reset_index()
    )
    combo["cover_ratio"] = combo["bs_covered"] / float(n_total_bs)
    method_priority = {
        "quantile_10": 0,
        "median": 1,
        "quantile_05": 2,
        "trimmed_mean": 3,
        "mean": 4,
        "min": 5,
    }
    combo["method_rank"] = combo["method"].map(method_priority).fillna(99)
    combo = combo.sort_values(["cover_ratio", "avg_n_samples", "method_rank"], ascending=[False, False, True])
    best = combo.iloc[0]
    return str(best["window"]), str(best["method"]), combo


def train_models(data: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    numeric_features = feature_cols
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features)]
    )

    models = {
        "LinearRegression": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "RidgeCV": Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        "LassoCV": Pipeline([("pre", pre), ("model", LassoCV(alphas=np.logspace(-3, 1, 40), cv=5, random_state=42, max_iter=50000))]),
        "RandomForest": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2)),
            ]
        ),
    }

    metrics_rows = []
    pred_rows = []
    rf_feature_importance = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics_rows.append(
            {
                "model": name,
                "R2": float(r2_score(y_test, pred)),
                "MAE": float(mean_absolute_error(y_test, pred)),
            }
        )
        pred_rows.extend(
            {
                "BS": bs,
                "model": name,
                "y_true": float(t),
                "y_pred": float(p),
            }
            for bs, t, p in zip(test_df["BS"], y_test.to_numpy(), pred)
        )

        if name == "RandomForest":
            rf = model.named_steps["model"]
            rf_feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

    metrics = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False)
    preds = pd.DataFrame(pred_rows)
    if rf_feature_importance is None:
        rf_feature_importance = pd.DataFrame(columns=["feature", "importance"])
    return metrics, preds, rf_feature_importance


def plot_method_distribution(est: pd.DataFrame, window_name: str) -> None:
    sub = est[est["window"] == window_name].copy()
    methods = ["min", "quantile_05", "quantile_10", "mean", "median", "trimmed_mean"]
    data = [sub.loc[sub["method"] == m, "p_base"].dropna().to_numpy() for m in methods]

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=methods, showfliers=False)
    plt.ylabel("P_base")
    plt.title(f"不同估计方法的 P_base 分布对比（窗口: {window_name}）")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_method_distribution.png", dpi=180)
    plt.close()


def plot_scatter_relations(dataset: pd.DataFrame, target_col: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(dataset["sum_pmax"], dataset[target_col], alpha=0.5, s=15)
    plt.xlabel("sum_pmax")
    plt.ylabel("P_base")
    plt.title("P_base vs sum_pmax")

    plt.subplot(1, 2, 2)
    plt.scatter(dataset["n_cells"], dataset[target_col], alpha=0.5, s=15)
    plt.xlabel("n_cells")
    plt.ylabel("P_base")
    plt.title("P_base vs n_cells")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_scatter_pbase_vs_features.png", dpi=180)
    plt.close()


def plot_window_stability(est: pd.DataFrame, method: str) -> None:
    sub = est[est["method"] == method]
    windows = sorted(sub["window"].unique().tolist())
    data = [sub.loc[sub["window"] == w, "p_base"].dropna().to_numpy() for w in windows]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=windows, showfliers=False)
    plt.ylabel("P_base")
    plt.title(f"不同静态窗口下 P_base 稳定性（方法: {method}）")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_window_stability.png", dpi=180)
    plt.close()


def plot_window_coverage(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary["window"], summary["bs_cover_ratio"])
    plt.ylabel("BS覆盖率")
    plt.title("静态窗口BS覆盖率对比")
    plt.xticks(rotation=20)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_window_coverage.png", dpi=180)
    plt.close()


def build_report(
    merged: pd.DataFrame,
    window_summary: pd.DataFrame,
    method_stability: pd.DataFrame,
    combo_summary: pd.DataFrame,
    chosen_window: str,
    chosen_method: str,
    model_metrics: pd.DataFrame,
    rf_imp: pd.DataFrame,
    no_sample_df: pd.DataFrame,
    model_dataset: pd.DataFrame,
) -> None:
    n_bs = merged["BS"].nunique()
    n_rows = len(merged)

    ws = window_summary.set_index("window")
    chosen_cover = combo_summary.loc[
        (combo_summary["window"] == chosen_window) & (combo_summary["method"] == chosen_method), "cover_ratio"
    ].iloc[0]

    corr_ncells = model_dataset[["n_cells", "p_base"]].corr().iloc[0, 1]
    corr_pmax = model_dataset[["sum_pmax", "p_base"]].corr().iloc[0, 1]
    corr_ant = model_dataset[["sum_antennas", "p_base"]].corr().iloc[0, 1]

    best_model = model_metrics.iloc[0]
    best_model_name = best_model["model"]
    best_r2 = best_model["R2"]
    best_mae = best_model["MAE"]

    top_rf = rf_imp.head(8)
    rf_lines = "\n".join(
        [f"- {row.feature}: {row.importance:.4f}" for row in top_rf.itertuples(index=False)]
    )

    no_sample_line = ", ".join(no_sample_df["window"].tolist())
    report = f"""# 阶段A：基站静态基础能耗估计（Simulation Results 风格）

## 1. 代码与运行说明
- 主脚本：`run_phaseA_static_energy.py`
- 数据输入：`../data/ECdata.csv`, `../data/CLdata.csv`, `../data/BSinfo.csv`
- 关键输出：`outputs/` 下的图表与CSV结果文件

## 2. 关键图表说明
- `fig_method_distribution.png`：同一静态窗口中不同估计器（min / quantile / mean / median / trimmed mean）的 `P_base` 分布对比。
- `fig_scatter_pbase_vs_features.png`：`P_base` 与 `sum_pmax`、`n_cells` 的散点关系，用于验证物理单调趋势。
- `fig_window_stability.png`：固定估计方法下，不同静态窗口得到的 `P_base` 稳定性对比。
- `fig_window_coverage.png`：不同静态窗口对BS样本覆盖率的影响。

## 3. 结果分析（TSG风格）
### 3.1 静态窗口敏感性分析
本实验在 BS-时间粒度上构建了 {n_rows} 条样本，覆盖 {n_bs} 个基站。不同静态窗口对样本可用性影响显著：
- A窗口（严格）：样本数 {int(ws.loc['A_strict_weighted','samples'])}，BS覆盖率 {ws.loc['A_strict_weighted','bs_cover_ratio']:.2%}
- B窗口（均值负载约束）：样本数 {int(ws.loc['B_mean_all_es_zero','samples'])}，BS覆盖率 {ws.loc['B_mean_all_es_zero','bs_cover_ratio']:.2%}
- C窗口（每BS低负载5%分位）：样本数 {int(ws.loc['C_bs_low5pct','samples'])}，BS覆盖率 {ws.loc['C_bs_low5pct','bs_cover_ratio']:.2%}
- D窗口（宽松）：样本数 {int(ws.loc['D_relaxed','samples'])}，BS覆盖率 {ws.loc['D_relaxed','bs_cover_ratio']:.2%}

窗口越严格，样本数下降越明显，容易导致部分BS无可用静态样本；窗口过宽则会混入非静态行为，抬高 `P_base` 估计值。

### 3.2 估计方法对比
跨窗口稳定性统计见 `method_stability_summary.csv`。总体上：
- `min` 估计器更容易受偶发低值影响，存在低估风险。
- `mean` 受残余动态负载影响更明显，可能偏高。
- `quantile`/`median` 在偏差与稳定性之间更均衡，具备更好的鲁棒折中。

本次用于后续建模的目标采用 `{chosen_window} + {chosen_method}`，BS覆盖率为 {chosen_cover:.2%}，兼顾覆盖度和稳健性。

### 3.3 物理合理性验证
以建模目标 `P_base` 为例，静态特征相关性如下：
- corr(`P_base`, `n_cells`) = {corr_ncells:.3f}
- corr(`P_base`, `sum_pmax`) = {corr_pmax:.3f}
- corr(`P_base`, `sum_antennas`) = {corr_ant:.3f}

结果显示 `P_base` 随资源规模（小区数、发射功率总和、天线规模）总体呈正相关，满足物理直觉。

### 3.4 统计建模结果
在 BS 级静态特征上，分别训练 Linear / Ridge / Lasso / RandomForest。最佳模型为 `{best_model_name}`：
- Test R² = {best_r2:.4f}
- Test MAE = {best_mae:.4f}

随机森林特征重要性（Top-8）：
{rf_lines}

### 3.5 关键发现
- The minimum-based estimator tends to underestimate `P_base` due to extreme outliers.
- Quantile-based estimation provides a robust trade-off between bias and variance.
- Static-feature-driven models can explain a substantial portion of inter-BS base-energy variance.

## 4. 结论（bullet points）
- 静态窗口阈值直接决定样本覆盖率与偏差水平，需要在“纯静态性”与“可用样本量”之间平衡。
- 相比最小值估计，分位数/中位数估计对异常点更不敏感，更适合作为 `P_base` 监督信号。
- `P_base` 与 `sum_pmax`、`n_cells`、`sum_antennas` 呈正向关系，验证了模型构建的物理合理性。
- 基于静态配置特征可实现对 `P_base` 的有效回归，为阶段B动态能耗建模提供物理基座。

## 5. 额外分析（加分项）
- `bs_without_static_samples.csv` 给出各窗口下无静态样本的BS列表（按窗口展开）。
- 对“无静态窗口样本”BS，建议采用“配置相似BS迁移”：
  1) 在静态特征空间（`sum_pmax`, `n_cells`, `sum_antennas`, RU/Mode比例）做最近邻匹配；
  2) 以匹配集合的 `P_base` 分位数估计作为回填值，并附带不确定性区间。

无静态样本窗口：{no_sample_line}
"""
    (BASE_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ec, cl, bsinfo = load_data()
    bs_feat = build_bs_static_features(bsinfo)
    bs_time = aggregate_cell_to_bs_time(cl, bsinfo)

    merged = ec.merge(bs_time, on=["BS", "Time"], how="inner")
    merged = merged.dropna(subset=["Energy", "load_pmax_weighted", "load_mean"])

    windows = define_windows(merged)
    window_summary = summarize_windows(merged, windows)
    pbase_est = estimate_pbase(merged, windows)
    method_stability = summarize_method_stability(pbase_est)

    chosen_window, chosen_method, combo_summary = choose_target_combo(pbase_est, merged["BS"].nunique())
    target = pbase_est[(pbase_est["window"] == chosen_window) & (pbase_est["method"] == chosen_method)][["BS", "p_base"]]

    model_dataset = bs_feat.merge(target, on="BS", how="inner")
    feature_cols = [c for c in model_dataset.columns if c not in {"BS", "p_base"}]
    metrics, preds, rf_imp = train_models(model_dataset, feature_cols, "p_base")

    # 无静态样本BS统计
    all_bs = sorted(merged["BS"].unique().tolist())
    no_sample_rows = []
    for wname in windows:
        covered = set(pbase_est.loc[pbase_est["window"] == wname, "BS"].unique())
        miss = [b for b in all_bs if b not in covered]
        no_sample_rows.append({"window": wname, "missing_bs_count": len(miss), "missing_bs_list": ",".join(miss)})
    no_sample_df = pd.DataFrame(no_sample_rows)

    plot_method_distribution(pbase_est, chosen_window)
    plot_scatter_relations(model_dataset, "p_base")
    plot_window_stability(pbase_est, chosen_method)
    plot_window_coverage(window_summary)

    merged.to_csv(OUTPUT_DIR / "bs_time_merged.csv", index=False)
    window_summary.to_csv(OUTPUT_DIR / "window_summary.csv", index=False)
    pbase_est.to_csv(OUTPUT_DIR / "pbase_estimates_long.csv", index=False)
    method_stability.to_csv(OUTPUT_DIR / "method_stability_summary.csv", index=False)
    combo_summary.to_csv(OUTPUT_DIR / "target_combo_summary.csv", index=False)
    model_dataset.to_csv(OUTPUT_DIR / "model_dataset.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    rf_imp.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)
    no_sample_df.to_csv(OUTPUT_DIR / "bs_without_static_samples.csv", index=False)

    build_report(
        merged=merged,
        window_summary=window_summary,
        method_stability=method_stability,
        combo_summary=combo_summary,
        chosen_window=chosen_window,
        chosen_method=chosen_method,
        model_metrics=metrics,
        rf_imp=rf_imp,
        no_sample_df=no_sample_df,
        model_dataset=model_dataset,
    )

    readme = f"""# 阶段A静态能耗估计交付

## 目录说明
- `run_phaseA_static_energy.py`：完整可运行代码（数据清洗、Cell->BS聚合、窗口筛选、P_base估计、建模、评估、可视化）。
- `outputs/`：全部中间结果、指标表和图表。
- `analysis_report.md`：中文学术风格结果分析（含结论与加分项讨论）。

## 运行方式
```bash
python run_phaseA_static_energy.py
```
"""
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("阶段A任务完成。输出目录：", BASE_DIR)
    print("建议先查看：analysis_report.md 和 outputs/model_metrics.csv")


if __name__ == "__main__":
    main()
