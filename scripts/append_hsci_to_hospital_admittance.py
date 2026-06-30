from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import re

import netCDF4
import numpy as np
import pandas as pd

from heatindex.utils import ZipGridMask, haversine_km, masked_to_nan, yyyymmdd_to_datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append HSCI and ZIP exposure metrics to hospital-admission CSV.")
    p.add_argument("--hw-nc-t", default="EXCD_MJJAS_HWdays.nc")
    p.add_argument("--hw-nc-hi", default="HI_EXCD_MJJAS_HWdays_90.nc")
    p.add_argument("--mag-nc-hi", default="HI_EXCDMAG_daily_1981_2025_90.nc")
    p.add_argument("--patient-csv", default="Hospital_Admittancecsv.csv")
    p.add_argument("--zip-csv", default="USZipsWithLatLon_20231227.csv")
    p.add_argument("--mask-cache", default="zip_grid_masks.pkl")
    p.add_argument("--zip-buffer-cells", type=int, default=1)
    p.add_argument("--out-csv", default="Hospital_Admittance_with_HSCI.csv")
    p.add_argument("--out-pickle", default="Hospital_Admittance_with_HSCI.pkl")
    return p.parse_args()


def parse_date_safe(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    s = str(value).strip()
    if not s or s.lower() in {"na", "nan", "n/a", "null"}:
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%-m/%-d/%Y", "%m/%d/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt).date())
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")


def matlab_weekday(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return np.nan
    return ((int(ts.weekday()) + 1) % 7) + 1


def read_time_and_hsci(path: str | Path):
    with netCDF4.Dataset(path) as ds:
        t = np.asarray(ds["time"][:], dtype=np.float64)
        hsci = masked_to_nan(ds["HSCI"][:]) if "HSCI" in ds.variables else np.full(t.shape, np.nan)
    dates = pd.to_datetime([d.date() for d in yyyymmdd_to_datetime(t)])
    return t, dates, hsci


def read_lat_lon(path: str | Path):
    with netCDF4.Dataset(path) as ds:
        lat_name = "lat" if "lat" in ds.variables else "y"
        lon_name = "lon" if "lon" in ds.variables else "x"
        lat = np.asarray(ds[lat_name][:], dtype=np.float64)
        lon = np.asarray(ds[lon_name][:], dtype=np.float64)
    return lat, lon


def find_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalize_zip_series(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return vals.str.zfill(5)


def build_zip_masks(args, needed_zips, lat_grid, lon_grid):
    cache = Path(args.mask_cache)
    if cache.exists():
        with cache.open("rb") as f:
            payload = pickle.load(f)
        cached_zips = set(payload.get("cached_zips", []))
        if cached_zips.issuperset(set(needed_zips)) and payload.get("cached_zip_buffer_cells") == args.zip_buffer_cells:
            print(f"Loading cached ZIP masks from {cache}")
            return payload["zcta_masks"]
        print("Cached ZIP masks missing requested ZIPs or using different neighborhood size; rebuilding.")

    print(f"Loading ZIP centroid CSV: {args.zip_csv}")
    z = pd.read_csv(args.zip_csv)
    zip_col = find_column(z.columns, {"zip", "zipcode", "zip_code", "zip5", "zcta", "zcta5", "postalcode"})
    lat_col = find_column(z.columns, {"lat", "latitude"})
    lon_col = find_column(z.columns, {"lon", "lng", "long", "longitude"})
    if zip_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify ZIP / lat / lon columns in {args.zip_csv}.")

    zip_vals = normalize_zip_series(z[zip_col])
    lat_vals = pd.to_numeric(z[lat_col], errors="coerce")
    lon_vals = pd.to_numeric(z[lon_col], errors="coerce")
    lookup = {zc: i for i, zc in enumerate(zip_vals)}
    print(f"  {len(z)} ZIP centroid rows loaded.")

    masks: dict[str, ZipGridMask] = {}
    for zc in needed_zips:
        if zc not in lookup:
            print(f"Warning: Zip {zc} not found in ZIP centroid CSV.")
            continue
        i = lookup[zc]
        zip_lat = float(lat_vals.iloc[i])
        zip_lon = float(lon_vals.iloc[i])
        if not np.isfinite(zip_lat) or not np.isfinite(zip_lon):
            print(f"Warning: Zip {zc} has missing lat/lon in ZIP centroid CSV.")
            continue
        r_near = int(np.argmin(np.abs(lat_grid - zip_lat)))
        c_near = int(np.argmin(np.abs(lon_grid - zip_lon)))
        r_start = max(0, r_near - args.zip_buffer_cells)
        r_end = min(len(lat_grid) - 1, r_near + args.zip_buffer_cells)
        c_start = max(0, c_near - args.zip_buffer_cells)
        c_end = min(len(lon_grid) - 1, c_near + args.zip_buffer_cells)
        r_count = r_end - r_start + 1
        c_count = c_end - c_start + 1
        mask = np.ones((r_count, c_count), dtype=bool)
        masks["z" + zc] = ZipGridMask(r_start, r_count, c_start, c_count, mask, int(mask.sum()), zip_lat, zip_lon)
        print(
            f"  Zip {zc}: centroid [{zip_lat:.4f}, {zip_lon:.4f}] -> "
            f"grid cell ({r_near + 1},{c_near + 1}), using {r_count}x{c_count} neighborhood ({int(mask.sum())} cells)"
        )

    with cache.open("wb") as f:
        pickle.dump({"zcta_masks": masks, "cached_zips": list(needed_zips), "cached_zip_buffer_cells": args.zip_buffer_cells}, f)
    print(f"Saved ZIP masks to {cache}")
    return masks


def make_date_index(dates: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}


def read_zip_avg(nc_file, var_name, date_index, zm: ZipGridMask, target_date, zero_if_date_missing: bool):
    target = pd.Timestamp(target_date).normalize()
    ti = date_index.get(target)
    if ti is None:
        return 0.0 if zero_if_date_missing else np.nan
    with netCDF4.Dataset(nc_file) as ds:
        slab = masked_to_nan(
            ds[var_name][
                zm.r_start : zm.r_start + zm.r_count,
                zm.c_start : zm.c_start + zm.c_count,
                ti,
            ]
        )
    slab = np.asarray(slab, dtype=np.float64)
    slab[~zm.mask] = np.nan
    good = np.isfinite(slab)
    if not np.any(good):
        return np.nan
    val = float(np.nanmean(slab[good]))
    return 0.0 if np.isnan(val) else val


def find_backward_nearest_heatwave_cell(nc_file, var_name, date_index, dates_vec, zm, target_date, lat_grid, lon_grid):
    if pd.isna(target_date):
        return np.nan, np.nan, np.nan
    target = pd.Timestamp(target_date).normalize()
    valid_dates = [pd.Timestamp(d).normalize() for d in dates_vec if pd.Timestamp(d).normalize() < target]
    if not valid_dates:
        return np.nan, np.nan, np.nan
    max_back = int((target - max(valid_dates)).days)
    if max_back < 1:
        return np.nan, np.nan, np.nan

    with netCDF4.Dataset(nc_file) as ds:
        var = ds[var_name]
        for d in range(1, int((target - min(valid_dates)).days) + 1):
            dd = target - pd.Timedelta(days=d)
            ti = date_index.get(dd)
            if ti is None:
                continue
            slab = masked_to_nan(
                var[
                    zm.r_start : zm.r_start + zm.r_count,
                    zm.c_start : zm.c_start + zm.c_count,
                    ti,
                ]
            )
            slab = np.asarray(slab, dtype=np.float64)
            slab[~zm.mask] = np.nan
            pos_mask = np.isfinite(slab) & (slab > 0)
            if not np.any(pos_mask):
                continue
            rr_local, cc_local = np.nonzero(pos_mask)
            r_abs = zm.r_start + rr_local
            c_abs = zm.c_start + cc_local
            cand_lats = lat_grid[r_abs]
            cand_lons = lon_grid[c_abs]
            cand_vals = slab[pos_mask]
            dists = haversine_km(zm.zip_lat, zm.zip_lon, cand_lats, cand_lons)
            idx_min = int(np.argmin(dists))
            return float(d), float(dists[idx_min]), float(cand_vals[idx_min])
    return np.nan, np.nan, np.nan


def coerce_nan_max_to_zero(x):
    return 0.0 if np.isnan(x) else x


def safe_quantile_positive(x, p):
    vals = np.asarray(x, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.inf
    return float(np.quantile(vals, p))


def make_exposure_category(x):
    vals = np.asarray(x, dtype=np.float64)
    cat = np.full(vals.shape, "missing", dtype=object)
    good = np.isfinite(vals)
    cat[good] = "none"
    pos = vals[good & (vals > 0)]
    if pos.size == 0:
        return cat
    q1, q2 = np.quantile(pos, [1 / 3, 2 / 3])
    cat[good & (vals > 0) & (vals <= q1)] = "low"
    cat[good & (vals > q1) & (vals <= q2)] = "moderate"
    cat[good & (vals > q2)] = "high"
    return cat


def main() -> None:
    args = parse_args()

    p = pd.read_csv(args.patient_csv, dtype={"zip5": str, "ADMIT_DATE": str, "DISCHARGE_DATE": str})
    if "zip5" not in p.columns:
        raise KeyError("Patient CSV must contain a zip5 column.")
    n_p = len(p)
    print(f"Loaded {n_p} patient records.")

    p["zip5"] = p["zip5"].astype(str).str.strip()
    admit_dt = p["ADMIT_DATE"].map(parse_date_safe) if "ADMIT_DATE" in p.columns else pd.Series(pd.NaT, index=p.index)
    discharge_dt = p["DISCHARGE_DATE"].map(parse_date_safe) if "DISCHARGE_DATE" in p.columns else pd.Series(pd.NaT, index=p.index)
    needed_zips = sorted(p["zip5"].dropna().astype(str).str.strip().unique().tolist())
    print(f"Unique zip codes needed: {len(needed_zips)}")

    lat_grid, lon_grid = read_lat_lon(args.mag_nc_hi)
    with netCDF4.Dataset(args.mag_nc_hi) as ds:
        t_daily_hi = np.asarray(ds["time"][:], dtype=np.float64)
    dates_daily_hi = pd.to_datetime([d.date() for d in yyyymmdd_to_datetime(t_daily_hi)])

    hw_time_t, hw_dates_t, hw_hsci_t = read_time_and_hsci(args.hw_nc_t)
    hw_time_hi, hw_dates_hi, hw_hsci_hi = read_time_and_hsci(args.hw_nc_hi)
    print(f"Daily HI file : {len(t_daily_hi)} MJJAS days")
    print(f"HW days T     : {len(hw_time_t)} days")
    print(f"HW days HI    : {len(hw_time_hi)} days")

    masks = build_zip_masks(args, needed_zips, lat_grid, lon_grid)
    idx_daily_hi = make_date_index(dates_daily_hi)
    idx_hw_t = make_date_index(hw_dates_t)
    idx_hw_hi = make_date_index(hw_dates_hi)

    hsci_t_by_date = {pd.Timestamp(d).normalize(): float(v) for d, v in zip(hw_dates_t, hw_hsci_t)}
    hsci_hi_by_date = {pd.Timestamp(d).normalize(): float(v) for d, v in zip(hw_dates_hi, hw_hsci_hi)}

    cols = {
        "HSCI_T_admit": np.full(n_p, np.nan),
        "HSCI_HI_admit": np.full(n_p, np.nan),
        "zcta_EXCD_T_admit": np.full(n_p, np.nan),
        "zcta_EXCD_HI_admit": np.full(n_p, np.nan),
        "HSCI_T_3d_prior": np.full(n_p, np.nan),
        "HSCI_HI_3d_prior": np.full(n_p, np.nan),
        "zcta_EXCD_T_3d_prior": np.full(n_p, np.nan),
        "zcta_EXCD_HI_3d_prior": np.full(n_p, np.nan),
        "HSCI_T_7d_prior": np.full(n_p, np.nan),
        "HSCI_HI_7d_prior": np.full(n_p, np.nan),
        "zcta_EXCD_T_7d_prior": np.full(n_p, np.nan),
        "zcta_EXCD_HI_7d_prior": np.full(n_p, np.nan),
        "days_excd_T_3d_prior": np.zeros(n_p),
        "days_excd_HI_3d_prior": np.zeros(n_p),
        "days_excd_T_7d_prior": np.zeros(n_p),
        "days_excd_HI_7d_prior": np.zeros(n_p),
        "max_excd_T_3d_prior": np.full(n_p, np.nan),
        "max_excd_HI_3d_prior": np.full(n_p, np.nan),
        "max_excd_T_7d_prior": np.full(n_p, np.nan),
        "max_excd_HI_7d_prior": np.full(n_p, np.nan),
        "miss_excd_T_admit": np.zeros(n_p),
        "miss_excd_HI_admit": np.zeros(n_p),
        "nearest_hw_back_days_HI_admit_nan": np.full(n_p, np.nan),
        "nearest_hw_space_km_HI_admit_nan": np.full(n_p, np.nan),
        "nearest_hw_excd_HI_admit_nan": np.full(n_p, np.nan),
    }

    for i in range(n_p):
        zc = str(p.at[i, "zip5"]).strip()
        zm = masks.get("z" + zc)
        d0 = admit_dt.iloc[i]
        if zm is None or pd.isna(d0):
            continue
        d0 = pd.Timestamp(d0).normalize()

        cols["HSCI_T_admit"][i] = hsci_t_by_date.get(d0, np.nan)
        cols["HSCI_HI_admit"][i] = hsci_hi_by_date.get(d0, np.nan)
        cols["zcta_EXCD_T_admit"][i] = read_zip_avg(args.hw_nc_t, "EXCD", idx_hw_t, zm, d0, True)
        cols["zcta_EXCD_HI_admit"][i] = read_zip_avg(args.mag_nc_hi, "HI_EXCDMAG", idx_daily_hi, zm, d0, False)
        cols["miss_excd_T_admit"][i] = float(np.isnan(cols["zcta_EXCD_T_admit"][i]))
        cols["miss_excd_HI_admit"][i] = float(np.isnan(cols["zcta_EXCD_HI_admit"][i]))

        if cols["miss_excd_HI_admit"][i]:
            back_days, space_km, excd_val = find_backward_nearest_heatwave_cell(
                args.mag_nc_hi, "HI_EXCDMAG", idx_daily_hi, dates_daily_hi, zm, d0, lat_grid, lon_grid
            )
            cols["nearest_hw_back_days_HI_admit_nan"][i] = back_days
            cols["nearest_hw_space_km_HI_admit_nan"][i] = space_km
            cols["nearest_hw_excd_HI_admit_nan"][i] = excd_val

        acc_hsci_t = acc_hsci_hi = 0.0
        acc_excd_t = acc_excd_hi = 0.0
        acc_hsci_t_3 = acc_hsci_hi_3 = 0.0
        acc_excd_t_3 = acc_excd_hi_3 = 0.0
        cnt_excd_t_7 = cnt_excd_hi_7 = 0
        cnt_excd_t_3 = cnt_excd_hi_3 = 0
        mx_excd_t_7 = mx_excd_hi_7 = np.nan
        mx_excd_t_3 = mx_excd_hi_3 = np.nan

        for offset in range(-7, 0):
            dd = d0 + pd.Timedelta(days=offset)
            vt = read_zip_avg(args.hw_nc_t, "EXCD", idx_hw_t, zm, dd, True)
            if not np.isnan(vt):
                acc_excd_t += vt
                if vt > 0:
                    cnt_excd_t_7 += 1
                    mx_excd_t_7 = vt if np.isnan(mx_excd_t_7) else max(mx_excd_t_7, vt)
                if offset >= -3:
                    acc_excd_t_3 += vt
                    if vt > 0:
                        cnt_excd_t_3 += 1
                        mx_excd_t_3 = vt if np.isnan(mx_excd_t_3) else max(mx_excd_t_3, vt)

            v = hsci_t_by_date.get(dd)
            if v is not None and np.isfinite(v):
                acc_hsci_t += v
                if offset >= -3:
                    acc_hsci_t_3 += v

            v = hsci_hi_by_date.get(dd)
            if v is not None and np.isfinite(v):
                acc_hsci_hi += v
                if offset >= -3:
                    acc_hsci_hi_3 += v

            vhi = read_zip_avg(args.mag_nc_hi, "HI_EXCDMAG", idx_daily_hi, zm, dd, False)
            if not np.isnan(vhi):
                acc_excd_hi += vhi
                if vhi > 0:
                    cnt_excd_hi_7 += 1
                    mx_excd_hi_7 = vhi if np.isnan(mx_excd_hi_7) else max(mx_excd_hi_7, vhi)
                if offset >= -3:
                    acc_excd_hi_3 += vhi
                    if vhi > 0:
                        cnt_excd_hi_3 += 1
                        mx_excd_hi_3 = vhi if np.isnan(mx_excd_hi_3) else max(mx_excd_hi_3, vhi)

        cols["HSCI_T_3d_prior"][i] = acc_hsci_t_3
        cols["HSCI_HI_3d_prior"][i] = acc_hsci_hi_3
        cols["zcta_EXCD_T_3d_prior"][i] = acc_excd_t_3
        cols["zcta_EXCD_HI_3d_prior"][i] = acc_excd_hi_3
        cols["HSCI_T_7d_prior"][i] = acc_hsci_t
        cols["HSCI_HI_7d_prior"][i] = acc_hsci_hi
        cols["zcta_EXCD_T_7d_prior"][i] = acc_excd_t
        cols["zcta_EXCD_HI_7d_prior"][i] = acc_excd_hi
        cols["days_excd_T_3d_prior"][i] = cnt_excd_t_3
        cols["days_excd_HI_3d_prior"][i] = cnt_excd_hi_3
        cols["days_excd_T_7d_prior"][i] = cnt_excd_t_7
        cols["days_excd_HI_7d_prior"][i] = cnt_excd_hi_7
        cols["max_excd_T_3d_prior"][i] = coerce_nan_max_to_zero(mx_excd_t_3)
        cols["max_excd_HI_3d_prior"][i] = coerce_nan_max_to_zero(mx_excd_hi_3)
        cols["max_excd_T_7d_prior"][i] = coerce_nan_max_to_zero(mx_excd_t_7)
        cols["max_excd_HI_7d_prior"][i] = coerce_nan_max_to_zero(mx_excd_hi_7)

        if i == 0 or (i + 1) % 100 == 0 or i + 1 == n_p:
            print(f"  Processed {i + 1}/{n_p}  (zip {zc}, {zm.n_cells} cells, admit {d0.date()})")

    for name, values in cols.items():
        p[name] = values

    p["admit_year"] = admit_dt.dt.year
    p["admit_month"] = admit_dt.dt.month
    p["admit_dayofyear"] = admit_dt.dt.dayofyear
    p["admit_weekday"] = admit_dt.map(matlab_weekday)
    los = (discharge_dt - admit_dt).dt.days.astype(float)
    los[pd.isna(admit_dt) | pd.isna(discharge_dt)] = np.nan
    p["length_of_stay_days"] = los
    p["long_stay_3plus"] = (p["length_of_stay_days"] >= 3).astype(float)
    p["any_heat_T_admit"] = (p["HSCI_T_admit"] > 0).astype(float)
    p["any_heat_HI_admit"] = (p["HSCI_HI_admit"] > 0).astype(float)
    p["any_excd_T_admit"] = (p["zcta_EXCD_T_admit"] > 0).astype(float)
    p["any_excd_HI_admit"] = (p["zcta_EXCD_HI_admit"] > 0).astype(float)

    p["excd_T_recent_intensity"] = p["zcta_EXCD_T_3d_prior"] / 3.0
    p["excd_HI_recent_intensity"] = p["zcta_EXCD_HI_3d_prior"] / 3.0
    p["excd_T_week_intensity"] = p["zcta_EXCD_T_7d_prior"] / 7.0
    p["excd_HI_week_intensity"] = p["zcta_EXCD_HI_7d_prior"] / 7.0
    p["excd_T_acceleration"] = p["excd_T_recent_intensity"] - p["excd_T_week_intensity"]
    p["excd_HI_acceleration"] = p["excd_HI_recent_intensity"] - p["excd_HI_week_intensity"]
    p["excd_T_admit_vs_3d"] = p["zcta_EXCD_T_admit"] - p["excd_T_recent_intensity"]
    p["excd_HI_admit_vs_3d"] = p["zcta_EXCD_HI_admit"] - p["excd_HI_recent_intensity"]
    p["excd_T_admit_vs_7d"] = p["zcta_EXCD_T_admit"] - p["excd_T_week_intensity"]
    p["excd_HI_admit_vs_7d"] = p["zcta_EXCD_HI_admit"] - p["excd_HI_week_intensity"]

    thr_t_admit = safe_quantile_positive(p["zcta_EXCD_T_admit"], 0.90)
    thr_hi_admit = safe_quantile_positive(p["zcta_EXCD_HI_admit"], 0.90)
    p["extreme_excd_T_admit"] = ((p["zcta_EXCD_T_admit"] >= thr_t_admit) & np.isfinite(p["zcta_EXCD_T_admit"])).astype(float)
    p["extreme_excd_HI_admit"] = ((p["zcta_EXCD_HI_admit"] >= thr_hi_admit) & np.isfinite(p["zcta_EXCD_HI_admit"])).astype(float)
    p["excd_T_admit_cat"] = make_exposure_category(p["zcta_EXCD_T_admit"])
    p["excd_HI_admit_cat"] = make_exposure_category(p["zcta_EXCD_HI_admit"])
    p["excd_T_7d_prior_cat"] = make_exposure_category(p["zcta_EXCD_T_7d_prior"])
    p["excd_HI_7d_prior_cat"] = make_exposure_category(p["zcta_EXCD_HI_7d_prior"])

    p.to_csv(args.out_csv, index=False)
    with Path(args.out_pickle).open("wb") as f:
        pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDone. Saved {args.out_csv}")


if __name__ == "__main__":
    main()
