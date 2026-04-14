from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_ieee_like_matplotlib_style() -> None:
    """
    An IEEE TSG-like (IEEE Transactions) plotting style:
    - Serif font (Times New Roman if available)
    - Small, consistent font sizes
    - Thin grid, modest line widths
    - High DPI export suitable for papers
    """

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 3,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_ecdata(ec_path: Path) -> pd.DataFrame:
    ec = pd.read_csv(ec_path)
    ec["Time"] = pd.to_datetime(ec["Time"], errors="coerce")
    ec["Energy"] = pd.to_numeric(ec["Energy"], errors="coerce")
    ec = ec.dropna(subset=["BS", "Time", "Energy"]).copy()
    return ec


def plot_bs_daily_energy(
    ec: pd.DataFrame,
    bs_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Plot one BS energy curves per day (x=hour-of-day, y=Energy).
    This is typically the most paper-friendly way to show "daily profiles".
    """

    df = ec.loc[ec["BS"].astype(str) == str(bs_id), ["Time", "Energy"]].copy()
    if df.empty:
        raise ValueError(f"No rows found for BS={bs_id!r} in ECdata.")

    df["date"] = df["Time"].dt.date
    df["hour"] = df["Time"].dt.hour

    if start_date is not None:
        df = df[df["Time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["Time"] <= pd.Timestamp(end_date)]
    if df.empty:
        raise ValueError("No data left after applying date filters.")

    # If there are multiple samples per hour, average them (robust for duplicates).
    prof = df.groupby(["date", "hour"], as_index=False)["Energy"].mean()
    pivot = prof.pivot(index="hour", columns="date", values="Energy").sort_index()

    # Ensure hours are 0..23 for a clean x-axis; allow missing hours as NaN.
    pivot = pivot.reindex(np.arange(24))

    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        date_span = f"{min(pivot.columns)}_{max(pivot.columns)}" if len(pivot.columns) else "unknown_dates"
        out_path = OUT_DIR / f"fig_bs_{bs_id}_daily_energy_{date_span}.png"

    set_ieee_like_matplotlib_style()

    # Figure size: single-column-ish or 1.5-column-ish; adjust as needed.
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    cmap = plt.get_cmap("tab10")

    dates = list(pivot.columns)
    for i, d in enumerate(dates):
        y = pivot[d].to_numpy()
        ax.plot(
            pivot.index.to_numpy(),
            y,
            marker="o",
            color=cmap(i % 10),
            alpha=0.95,
            label=str(d),
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy")
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.grid(True, which="major", axis="both")

    # Compact legend for up to ~7-8 days; more days gets unreadable.
    if len(dates) <= 8:
        ax.legend(loc="best", frameon=False, ncol=1, handlelength=2.0)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncol=1, title="Date")

    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-day energy curves for a specific BS (IEEE-like style).")
    parser.add_argument("--bs", required=True, help="Base station id (string match against ECdata.BS).")
    parser.add_argument("--ec", default=str(DATA_DIR / "ECdata.csv"), help="Path to ECdata.csv")
    parser.add_argument("--start-date", default=None, help="Optional ISO date/time filter, e.g. 2026-04-01")
    parser.add_argument("--end-date", default=None, help="Optional ISO date/time filter, e.g. 2026-04-08")
    parser.add_argument("--out", default=None, help="Output image path. Default: phaseB outputs/")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    ec = load_ecdata(Path(args.ec))
    out_path = plot_bs_daily_energy(
        ec=ec,
        bs_id=str(args.bs),
        start_date=args.start_date,
        end_date=args.end_date,
        out_path=Path(args.out) if args.out else None,
        show=bool(args.show),
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

