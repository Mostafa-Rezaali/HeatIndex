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
        description="Plot monthly accumulated HSCI from an HSCI NetCDF file."
    )
    p.add_argument("--hsci-nc", default="HI_EXCD_MJJAS_HWdays_90.nc")
    p.add_argument("--out-csv", default="monthly_accumulated_HSCI_standard_p90_grace1.csv")
    p.add_argument("--out-png", default="monthly_accumulated_HSCI_standard_p90_grace1.png")
    p.add_argument("--min-duration", type=int, default=3)
    p.add_argument("--grace-days", type=int, default=1)
    p.add_argument("--title", default="Monthly Accumulated HSCI")
    return p.parse_args()


def full_mjjas_dates(start_year: int, end_year: int) -> pd.DatetimeIndex:
    parts = []
    for year in range(start_year, end_year + 1):
        days = pd.date_range(f"{year}-05-01", f"{year}-09-30", freq="D")
        parts.append(days)
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def keep_persistent_events(
    is_hot: np.ndarray,
    dates: pd.DatetimeIndex,
    min_duration: int,
    grace_days: int,
) -> np.ndarray:
    keep = np.zeros(is_hot.shape, dtype=bool)
    years = np.unique(dates.year)

    for year in years:
        idx = np.nonzero(dates.year == year)[0]
        if idx.size == 0:
            continue

        pos = 0
        while pos < idx.size:
            if not is_hot[idx[pos]]:
                pos += 1
                continue

            run_positions = []
            hot_count = 0
            gap_count = 0
            j = pos

            while j < idx.size:
                day_idx = idx[j]
                if is_hot[day_idx]:
                    run_positions.append(day_idx)
                    hot_count += 1
                    gap_count = 0
                    j += 1
                    continue

                if gap_count < grace_days and j + 1 < idx.size and is_hot[idx[j + 1]]:
                    gap_count += 1
                    j += 1
                    continue

                break

            if hot_count >= min_duration:
                keep[np.asarray(run_positions, dtype=np.int64)] = True
            pos = max(j, pos + 1)

    return keep


def read_hsci_series(path: Path) -> pd.Series:
    with netCDF4.Dataset(path) as ds:
        if "time" not in ds.variables or "HSCI" not in ds.variables:
            raise KeyError(f"{path} must contain variables named 'time' and 'HSCI'.")
        dates = pd.to_datetime([d.date() for d in yyyymmdd_to_datetime(ds["time"][:])])
        hsci = masked_to_nan(ds["HSCI"][:]).astype(np.float64)

    return pd.Series(hsci, index=pd.DatetimeIndex(dates), name="HSCI").sort_index()


def plot_monthly(monthly: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ax.plot(
        monthly["month"],
        monthly["accumulated_HSCI"],
        color="#1f5a8a",
        linewidth=1.8,
    )
    ax.scatter(
        monthly["month"],
        monthly["accumulated_HSCI"],
        color="#1f5a8a",
        s=12,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Accumulated HSCI")
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
    ax.margins(x=0.01)
    fig.autofmt_xdate()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hsci_nc = Path(args.hsci_nc)
    if not hsci_nc.is_file():
        raise FileNotFoundError(hsci_nc)

    sparse = read_hsci_series(hsci_nc)
    if sparse.empty:
        raise ValueError(f"{hsci_nc} contains no HSCI values.")

    full_dates = full_mjjas_dates(int(sparse.index.year.min()), int(sparse.index.year.max()))
    daily = pd.Series(0.0, index=full_dates, name="HSCI")
    daily.loc[sparse.index.intersection(daily.index)] = sparse.loc[sparse.index.intersection(daily.index)]
    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    is_hot = daily.to_numpy() > 0
    keep = keep_persistent_events(
        is_hot,
        daily.index,
        min_duration=args.min_duration,
        grace_days=args.grace_days,
    )
    filtered = daily.where(keep, 0.0)

    monthly = (
        filtered.groupby(filtered.index.to_period("M"))
        .agg(accumulated_HSCI="sum", heatwave_days=lambda x: int(np.count_nonzero(np.asarray(x) > 0)))
        .reset_index()
    )
    monthly["month"] = monthly["index"].dt.to_timestamp()
    monthly = monthly[["month", "accumulated_HSCI", "heatwave_days"]]

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    monthly.to_csv(out_csv, index=False)
    plot_monthly(monthly, out_png, args.title)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    print(
        f"Used {hsci_nc}; min_duration={args.min_duration}; "
        f"grace_days={args.grace_days}; months={len(monthly)}"
    )


if __name__ == "__main__":
    main()
