from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatindex.utils import masked_to_nan, yyyymmdd_to_datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot accumulated HSCI/AHSCI from an HSCI NetCDF file."
    )
    p.add_argument("--hsci-nc", default="HI_EXCD_MJJAS_HWdays_90.nc")
    p.add_argument("--out-csv", default="annual_accumulated_AHSCI.csv")
    p.add_argument("--out-png", default="annual_accumulated_AHSCI.png")
    p.add_argument(
        "--group-by",
        choices=("year", "month", "month-of-year"),
        default="year",
        help="Aggregation for the AHSCI plot. Default is annual sum by year.",
    )
    p.add_argument("--title", default="Annual Accumulated HSCI")
    return p.parse_args()


def read_hsci_series(path: Path) -> pd.Series:
    with netCDF4.Dataset(path) as ds:
        if "time" not in ds.variables or "HSCI" not in ds.variables:
            raise KeyError(f"{path} must contain variables named 'time' and 'HSCI'.")
        dates = pd.to_datetime([d.date() for d in yyyymmdd_to_datetime(ds["time"][:])])
        hsci = masked_to_nan(ds["HSCI"][:]).astype(np.float64)

    series = pd.Series(hsci, index=pd.DatetimeIndex(dates), name="HSCI").sort_index()
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def aggregate_hsci(series: pd.Series, group_by: str) -> pd.DataFrame:
    if group_by == "year":
        grouped = series.groupby(series.index.year)
        out = grouped.agg(AHSCI="sum", heatwave_days=lambda x: int(np.count_nonzero(np.asarray(x) > 0))).reset_index()
        out = out.rename(columns={"index": "year"})
        return out[["year", "AHSCI", "heatwave_days"]]

    if group_by == "month":
        grouped = series.groupby(series.index.to_period("M"))
        out = grouped.agg(AHSCI="sum", heatwave_days=lambda x: int(np.count_nonzero(np.asarray(x) > 0))).reset_index()
        out["month"] = out["index"].dt.to_timestamp()
        return out[["month", "AHSCI", "heatwave_days"]]

    grouped = series.groupby(series.index.month)
    out = grouped.agg(AHSCI="sum", heatwave_days=lambda x: int(np.count_nonzero(np.asarray(x) > 0))).reset_index()
    out = out.rename(columns={"index": "month_number"})
    out["month"] = pd.to_datetime(out["month_number"], format="%m").dt.strftime("%b")
    return out[["month_number", "month", "AHSCI", "heatwave_days"]]


def plot_ahsci(summary: pd.DataFrame, out_png: Path, title: str, group_by: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)

    if group_by == "year":
        ax.plot(summary["year"], summary["AHSCI"], color="#1f5a8a", linewidth=1.8)
        ax.scatter(summary["year"], summary["AHSCI"], color="#1f5a8a", s=20, zorder=3)
        ax.set_xlabel("Year")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    elif group_by == "month":
        ax.plot(summary["month"], summary["AHSCI"], color="#1f5a8a", linewidth=1.8)
        ax.scatter(summary["month"], summary["AHSCI"], color="#1f5a8a", s=14, zorder=3)
        ax.set_xlabel("Month")
        fig.autofmt_xdate()
    else:
        ax.bar(summary["month"], summary["AHSCI"], color="#1f5a8a", width=0.72)
        ax.set_xlabel("Calendar month")

    ax.set_title(title)
    ax.set_ylabel("Accumulated HSCI (AHSCI)")
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8)
    ax.margins(x=0.01)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hsci_nc = Path(args.hsci_nc)
    if not hsci_nc.is_file():
        raise FileNotFoundError(hsci_nc)

    hsci = read_hsci_series(hsci_nc)
    if hsci.empty:
        raise ValueError(f"{hsci_nc} contains no finite HSCI values.")

    summary = aggregate_hsci(hsci, args.group_by)

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    summary.to_csv(out_csv, index=False)
    plot_ahsci(summary, out_png, args.title, args.group_by)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    print(f"Used {hsci_nc}; group_by={args.group_by}; rows={len(summary)}")


if __name__ == "__main__":
    main()
