"""
策略 A（稀疏观测）：不对 Energy 做插补；监督与质检仅基于真实出现的 (BS, Time)。

注意：在 pandas 中对按时间排序的行做 ``shift(1)`` / ``rolling(24)`` 时，语义是
「相邻观测行」，不是「日历上的上一小时」。若 EC 采样不规则，滞后与滚动窗口
可能跨过多个真实小时；日内预测里若使用 ``make_continuity_mask`` 一类校验，
则标签对齐的是「整小时间隔」的相邻行，与上述 panel 特征仍可能不一致。
"""

from __future__ import annotations

import pandas as pd


def hours_since_prev_obs(times: pd.Series) -> pd.Series:
    """
    对**已按时间升序**的单个序列，返回与 ``times`` 同索引的「距上一观测点的小时数」；
    第一个有效时间点为 NaN。
    """
    t = pd.to_datetime(times, errors="coerce")
    delta = t.diff()
    return delta.dt.total_seconds() / 3600.0


def contiguous_run_id(times: pd.Series, gap: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.Series:
    """
    对**已按时间升序**的单个序列，当相邻时间差 *严格大于* ``gap`` 时开始新的一段，返回段编号 0,1,2,...
    """
    t = pd.to_datetime(times, errors="coerce")
    delta = t.diff()
    new_run = (delta > gap).fillna(False)
    return new_run.astype(int).cumsum()
