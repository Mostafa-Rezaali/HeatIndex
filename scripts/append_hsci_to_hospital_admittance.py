from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
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
    p.add_argument("--hi-pcts", default="90,95")
    p.add_argument("--hi-mag-template", default="HI_EXCDMAG_daily_1981_2025_{pct}.nc")
    p.add_argument("--hi-hw-template", default="HI_EXCD_MJJAS_HWdays_{pct}.nc")
    p.add_argument("--patient-csv", default="Hospital_Admittancecsv.csv")
    p.add_argument("--zip-csv", default="USZipsWithLatLon_20231227.csv")
    p.add_argument("--mask-cache", default="")
    p.add_argument("--zip-buffer-cells", type=int, default=1)
    p.add_argument("--out-csv", default="Hospital_Admittance_with_HSCI.csv")
    p.add_argument("--out-pickle", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=100)
    return p.parse_args()


def parse_pcts(value: str) -> list[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("--hi-pcts must include at least one percentile")
    return out


def pct_suffix(pct: int) -> str:
    return f"p{pct}"


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


def load_hi_context(pct: int, args):
    mag_nc = Path(args.hi_mag_template.format(pct=pct))
    hw_nc = Path(args.hi_hw_template.format(pct=pct))
    if not mag_nc.is_file():
        raise FileNotFoundError(f"Missing HI MAG file for P{pct}: {mag_nc}")
    if not hw_nc.is_file():
        raise FileNotFoundError(f"Missing HSCI-H file for P{pct}: {hw_nc}")

    with netCDF4.Dataset(mag_nc) as ds:
        t_daily = np.asarray(ds["time"][:], dtype=np.float64)
    dates_daily = pd.to_datetime([d.date() for d in yyyymmdd_to_datetime(t_daily)])
    _, hw_dates, hw_hsci = read_time_and_hsci(hw_nc)
    return {
        "pct": pct,
        "suffix": pct_suffix(pct),
        "mag_nc": str(mag_nc),
        "hw_nc": str(hw_nc),
        "dates_daily": dates_daily,
        "idx_daily": make_date_index(dates_daily),
        "hw_dates": hw_dates,
        "hsci_by_date": {pd.Timestamp(d).normalize(): float(v) for d, v in zip(hw_dates, hw_hsci)},
    }


def read_lat_lon(path: str | Path):
    with netCDF4.Dataset(path) as ds:
        lat_name = "lat" if "lat" in ds.variables else "y"
        lon_name = "lon" if "lon" in ds.variables else "x"
        lat = np.asarray(ds[lat_name][:], dtype=np.float64)
        lon = np.asarray(ds[lon_name][:], dtype=np.float64)
    return lat, lon


def find_column(columns, candidates):
    def norm(value):
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    lower = {str(c).strip().lower(): c for c in columns}
    compact = {norm(c): c for c in columns}
    for cand in candidates:
        cand_lc = str(cand).strip().lower()
        if cand_lc in lower:
            return lower[cand_lc]
        cand_norm = norm(cand)
        if cand_norm in compact:
            return compact[cand_norm]
    return None


def normalize_zip_series(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return vals.str.zfill(5)


def grid_signature(lat_grid, lon_grid):
    lat = np.asarray(lat_grid, dtype=np.float64)
    lon = np.asarray(lon_grid, dtype=np.float64)
    return {
        "nlat": int(lat.size),
        "nlon": int(lon.size),
        "lat0": float(lat[0]) if lat.size else np.nan,
        "lat1": float(lat[-1]) if lat.size else np.nan,
        "lon0": float(lon[0]) if lon.size else np.nan,
        "lon1": float(lon[-1]) if lon.size else np.nan,
    }


def same_grid_signature(a, b):
    if not a or not b:
        return False
    if a.get("nlat") != b.get("nlat") or a.get("nlon") != b.get("nlon"):
        return False
    for key in ("lat0", "lat1", "lon0", "lon1"):
        if not np.isclose(float(a.get(key, np.nan)), float(b.get(key, np.nan)), equal_nan=True):
            return False
    return True


def build_zip_masks(args, needed_zips, lat_grid, lon_grid):
    import pickle

    sig = grid_signature(lat_grid, lon_grid)
    cache = Path(args.mask_cache)
    if args.mask_cache and cache.exists():
        with cache.open("rb") as f:
            payload = pickle.load(f)
        cached_zips = set(payload.get("cached_zips", []))
        if (
            cached_zips.issuperset(set(needed_zips))
            and payload.get("cached_zip_buffer_cells") == args.zip_buffer_cells
            and same_grid_signature(payload.get("grid_signature"), sig)
        ):
            print(f"Loading cached ZIP masks from {cache}")
            return payload["zcta_masks"]
        print("Cached ZIP masks missing requested ZIPs, using a different neighborhood size, or built on a different grid; rebuilding.")

    print(f"Loading ZIP centroid CSV: {args.zip_csv}")
    z = pd.read_csv(args.zip_csv)
    zip_col = find_column(z.columns, {"zip", "zipcode", "zip_code", "zip5", "zcta", "zcta5", "postal code", "postalcode"})
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

    if args.mask_cache:
        with cache.open("wb") as f:
            pickle.dump(
                {
                    "zcta_masks": masks,
                    "cached_zips": list(needed_zips),
                    "cached_zip_buffer_cells": args.zip_buffer_cells,
                    "grid_signature": sig,
                },
                f,
            )
        print(f"Saved ZIP masks to {cache}")
    return masks


def make_date_index(dates: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}


def spatial_time_slab(var, ti: int, zm: ZipGridMask) -> np.ndarray | None:
    dims = tuple(str(d).lower() for d in var.dimensions)
    try:
        time_axis = dims.index("time")
    except ValueError:
        time_axis = len(dims) - 1

    row_axis = next((i for i, d in enumerate(dims) if d in {"lat", "y"}), None)
    col_axis = next((i for i, d in enumerate(dims) if d in {"lon", "x"}), None)
    if row_axis is None or col_axis is None or row_axis == col_axis:
        candidates = [i for i in range(len(dims)) if i != time_axis]
        if len(candidates) < 2:
            return None
        row_axis, col_axis = candidates[:2]

    if ti >= var.shape[time_axis] or zm.r_start >= var.shape[row_axis] or zm.c_start >= var.shape[col_axis]:
        return None

    r_stop = min(zm.r_start + zm.r_count, var.shape[row_axis])
    c_stop = min(zm.c_start + zm.c_count, var.shape[col_axis])
    slices = [slice(None)] * len(dims)
    slices[time_axis] = ti
    slices[row_axis] = slice(zm.r_start, r_stop)
    slices[col_axis] = slice(zm.c_start, c_stop)

    slab = masked_to_nan(var[tuple(slices)])
    remaining_axes = [axis for axis in range(len(dims)) if axis != time_axis]
    row_pos = remaining_axes.index(row_axis)
    col_pos = remaining_axes.index(col_axis)
    slab = np.asarray(slab, dtype=np.float64)
    if slab.ndim != 2:
        slab = np.squeeze(slab)
    if slab.ndim != 2:
        return None
    if (row_pos, col_pos) != (0, 1):
        slab = np.moveaxis(slab, (row_pos, col_pos), (0, 1))
    return slab


def read_zip_avg(nc_file, var_name, date_index, zm: ZipGridMask, target_date, zero_if_date_missing: bool):
    target = pd.Timestamp(target_date).normalize()
    ti = date_index.get(target)
    if ti is None:
        return 0.0 if zero_if_date_missing else np.nan
    cache_key = zip_avg_cache_key(nc_file, var_name, ti, zm, zero_if_date_missing)
    cached = _ZIP_AVG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ds = cached_dataset(nc_file)
    var = ds[var_name]
    slab = spatial_time_slab(var, ti, zm)
    if slab is None:
        _ZIP_AVG_CACHE[cache_key] = np.nan
        return np.nan
    mask = zm.mask[: slab.shape[0], : slab.shape[1]]
    slab[~mask] = np.nan
    good = np.isfinite(slab)
    if not np.any(good):
        _ZIP_AVG_CACHE[cache_key] = np.nan
        return np.nan
    val = float(np.nanmean(slab[good]))
    val = 0.0 if np.isnan(val) else val
    _ZIP_AVG_CACHE[cache_key] = val
    return val


def sum_hsci_prior(hsci_by_date, admit_date, days: int) -> float:
    total = 0.0
    for offset in range(-days, 0):
        dd = pd.Timestamp(admit_date).normalize() + pd.Timedelta(days=offset)
        v = hsci_by_date.get(dd)
        if v is not None and np.isfinite(v):
            total += v
    return total


def count_zip_heat_days(ctx, zm: ZipGridMask, admit_date, days: int) -> int:
    count = 0
    for offset in range(-days, 0):
        dd = pd.Timestamp(admit_date).normalize() + pd.Timedelta(days=offset)
        v = read_zip_avg(ctx["mag_nc"], "HI_EXCDMAG", ctx["idx_daily"], zm, dd, False)
        if np.isfinite(v) and v > 0:
            count += 1
    return count


def anchored_heat_duration_with_grace(ctx, zm: ZipGridMask, admit_date, max_back_days: int = 30, grace_days: int = 1) -> int:
    heat_days = 0
    grace_used = 0
    for offset in range(0, -max_back_days - 1, -1):
        dd = pd.Timestamp(admit_date).normalize() + pd.Timedelta(days=offset)
        v = read_zip_avg(ctx["mag_nc"], "HI_EXCDMAG", ctx["idx_daily"], zm, dd, False)
        is_heat = np.isfinite(v) and v > 0
        if is_heat:
            heat_days += 1
            continue
        if grace_used < grace_days:
            grace_used += 1
            continue
        break
    return heat_days


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

    ds = cached_dataset(nc_file)
    var = ds[var_name]
    for d in range(1, int((target - min(valid_dates)).days) + 1):
        dd = target - pd.Timedelta(days=d)
        ti = date_index.get(dd)
        if ti is None:
            continue
        slab = spatial_time_slab(var, ti, zm)
        if slab is None:
            continue
        mask = zm.mask[: slab.shape[0], : slab.shape[1]]
        slab[~mask] = np.nan
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


_HOSPITAL_CONTEXT = {}
_NC_DATASET_CACHE = {}
_ZIP_AVG_CACHE = {}


def cached_dataset(path):
    key = str(path)
    ds = _NC_DATASET_CACHE.get(key)
    if ds is None:
        ds = netCDF4.Dataset(key)
        _NC_DATASET_CACHE[key] = ds
    return ds


def zip_avg_cache_key(nc_file, var_name, ti, zm: ZipGridMask, zero_if_date_missing: bool):
    return (
        str(nc_file),
        str(var_name),
        int(ti),
        int(zm.r_start),
        int(zm.r_count),
        int(zm.c_start),
        int(zm.c_count),
        bool(zero_if_date_missing),
    )


def init_hospital_worker(context):
    global _HOSPITAL_CONTEXT
    _HOSPITAL_CONTEXT = context


def compute_patient_values(i, zc, d0, context=None):
    ctx = _HOSPITAL_CONTEXT if context is None else context
    a = ctx["args"]
    masks_hi = ctx["masks_hi"]
    masks_t = ctx["masks_t"]
    hi_contexts = ctx["hi_contexts"]
    legacy_hi = ctx["legacy_hi"]
    hsci_t_by_date = ctx["hsci_t_by_date"]
    hsci_hi_by_date = ctx["hsci_hi_by_date"]
    idx_hw_t = ctx["idx_hw_t"]
    idx_daily_hi = ctx["idx_daily_hi"]
    dates_daily_hi = ctx["dates_daily_hi"]
    lat_grid = ctx["lat_grid"]
    lon_grid = ctx["lon_grid"]

    values = {}
    zkey = "z" + str(zc).strip()
    zm_hi = masks_hi.get(zkey)
    zm_t = masks_t.get(zkey)
    if (zm_hi is None and zm_t is None) or pd.isna(d0):
        return i, values
    d0 = pd.Timestamp(d0).normalize()

    values["HSCI_T_admit"] = hsci_t_by_date.get(d0, np.nan)
    values["HSCI_HI_admit"] = hsci_hi_by_date.get(d0, np.nan)
    values["zcta_EXCD_T_admit"] = np.nan
    values["zcta_EXCD_HI_admit"] = np.nan
    if zm_t is not None:
        values["zcta_EXCD_T_admit"] = read_zip_avg(a["hw_nc_t"], "EXCD", idx_hw_t, zm_t, d0, True)
    if zm_hi is not None:
        values["zcta_EXCD_HI_admit"] = read_zip_avg(legacy_hi["mag_nc"], "HI_EXCDMAG", idx_daily_hi, zm_hi, d0, False)
    values["miss_excd_T_admit"] = float(np.isnan(values["zcta_EXCD_T_admit"]))
    values["miss_excd_HI_admit"] = float(np.isnan(values["zcta_EXCD_HI_admit"]))

    if zm_hi is not None and values["miss_excd_HI_admit"]:
        back_days, space_km, excd_val = find_backward_nearest_heatwave_cell(
            legacy_hi["mag_nc"], "HI_EXCDMAG", idx_daily_hi, dates_daily_hi, zm_hi, d0, lat_grid, lon_grid
        )
        values["nearest_hw_back_days_HI_admit_nan"] = back_days
        values["nearest_hw_space_km_HI_admit_nan"] = space_km
        values["nearest_hw_excd_HI_admit_nan"] = excd_val

    if zm_hi is not None:
        for pct, hi_ctx in hi_contexts.items():
            s = hi_ctx["suffix"]
            values[f"HSCI_HI_30d_prior_{s}"] = sum_hsci_prior(hi_ctx["hsci_by_date"], d0, 30)
            values[f"event_duration_HI_admit_anchor_{s}"] = anchored_heat_duration_with_grace(hi_ctx, zm_hi, d0)
            values[f"days_heatwave_HI_30d_prior_{s}"] = count_zip_heat_days(hi_ctx, zm_hi, d0, 30)
            values[f"days_heatwave_HI_21d_prior_{s}"] = count_zip_heat_days(hi_ctx, zm_hi, d0, 21)
            values[f"days_heatwave_HI_14d_prior_{s}"] = count_zip_heat_days(hi_ctx, zm_hi, d0, 14)

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
        vt = read_zip_avg(a["hw_nc_t"], "EXCD", idx_hw_t, zm_t, dd, True) if zm_t is not None else np.nan
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

        vhi = read_zip_avg(legacy_hi["mag_nc"], "HI_EXCDMAG", idx_daily_hi, zm_hi, dd, False) if zm_hi is not None else np.nan
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

    values["HSCI_T_3d_prior"] = acc_hsci_t_3
    values["HSCI_HI_3d_prior"] = acc_hsci_hi_3
    values["zcta_EXCD_T_3d_prior"] = acc_excd_t_3
    values["zcta_EXCD_HI_3d_prior"] = acc_excd_hi_3
    values["HSCI_T_7d_prior"] = acc_hsci_t
    values["HSCI_HI_7d_prior"] = acc_hsci_hi
    values["zcta_EXCD_T_7d_prior"] = acc_excd_t
    values["zcta_EXCD_HI_7d_prior"] = acc_excd_hi
    values["days_excd_T_3d_prior"] = cnt_excd_t_3
    values["days_excd_HI_3d_prior"] = cnt_excd_hi_3
    values["days_excd_T_7d_prior"] = cnt_excd_t_7
    values["days_excd_HI_7d_prior"] = cnt_excd_hi_7
    values["max_excd_T_3d_prior"] = coerce_nan_max_to_zero(mx_excd_t_3)
    values["max_excd_HI_3d_prior"] = coerce_nan_max_to_zero(mx_excd_hi_3)
    values["max_excd_T_7d_prior"] = coerce_nan_max_to_zero(mx_excd_t_7)
    values["max_excd_HI_7d_prior"] = coerce_nan_max_to_zero(mx_excd_hi_7)
    return i, values


def process_patient_chunk(rows):
    return [compute_patient_values(i, zc, d0) for i, zc, d0 in rows]


def patient_chunks(rows, chunk_size):
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


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

    hi_pcts = parse_pcts(args.hi_pcts)
    hi_contexts = {pct: load_hi_context(pct, args) for pct in hi_pcts}
    legacy_pct = 90 if 90 in hi_contexts else hi_pcts[0]
    legacy_hi = hi_contexts[legacy_pct]

    hw_time_t, hw_dates_t, hw_hsci_t = read_time_and_hsci(args.hw_nc_t)
    lat_grid, lon_grid = read_lat_lon(legacy_hi["mag_nc"])
    lat_grid_t, lon_grid_t = read_lat_lon(args.hw_nc_t)
    dates_daily_hi = legacy_hi["dates_daily"]
    print(f"Daily HI file P{legacy_pct}: {len(dates_daily_hi)} MJJAS days")
    print(f"HW days T     : {len(hw_time_t)} days")
    for pct, ctx in hi_contexts.items():
        print(f"HW days HI P{pct}: {len(ctx['hw_dates'])} days")

    print("Building ZIP masks for HI grid...")
    masks_hi = build_zip_masks(args, needed_zips, lat_grid, lon_grid)
    if lat_grid.shape == lat_grid_t.shape and lon_grid.shape == lon_grid_t.shape and np.allclose(lat_grid, lat_grid_t) and np.allclose(lon_grid, lon_grid_t):
        print("T grid matches HI grid; reusing HI ZIP masks for T.")
        masks_t = masks_hi
    else:
        print("Building ZIP masks for T grid...")
        masks_t = build_zip_masks(args, needed_zips, lat_grid_t, lon_grid_t)
    idx_daily_hi = legacy_hi["idx_daily"]
    idx_hw_t = make_date_index(hw_dates_t)

    hsci_t_by_date = {pd.Timestamp(d).normalize(): float(v) for d, v in zip(hw_dates_t, hw_hsci_t)}
    hsci_hi_by_date = legacy_hi["hsci_by_date"]

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
    for pct, ctx in hi_contexts.items():
        s = ctx["suffix"]
        cols[f"HSCI_HI_30d_prior_{s}"] = np.full(n_p, np.nan)
        cols[f"event_duration_HI_admit_anchor_{s}"] = np.full(n_p, np.nan)
        cols[f"days_heatwave_HI_30d_prior_{s}"] = np.zeros(n_p)
        cols[f"days_heatwave_HI_21d_prior_{s}"] = np.zeros(n_p)
        cols[f"days_heatwave_HI_14d_prior_{s}"] = np.zeros(n_p)

    worker_context = {
        "args": {
            "hw_nc_t": args.hw_nc_t,
        },
        "masks_hi": masks_hi,
        "masks_t": masks_t,
        "hi_contexts": hi_contexts,
        "legacy_hi": legacy_hi,
        "hsci_t_by_date": hsci_t_by_date,
        "hsci_hi_by_date": hsci_hi_by_date,
        "idx_hw_t": idx_hw_t,
        "idx_daily_hi": idx_daily_hi,
        "dates_daily_hi": dates_daily_hi,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }
    rows = [(i, str(p.at[i, "zip5"]).strip(), admit_dt.iloc[i]) for i in range(n_p)]
    worker_count = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size))
    print(f"Patient exposure workers: {worker_count}; chunk size: {chunk_size}")

    def store_result(i, values):
        for name, value in values.items():
            cols[name][i] = value

    processed = 0
    if worker_count == 1:
        for row in rows:
            i, values = compute_patient_values(*row, context=worker_context)
            store_result(i, values)
            processed += 1
            if processed == 1 or processed % 100 == 0 or processed == n_p:
                print(f"  Processed {processed}/{n_p}")
    else:
        chunks = list(patient_chunks(rows, chunk_size))
        with ProcessPoolExecutor(max_workers=worker_count, initializer=init_hospital_worker, initargs=(worker_context,)) as ex:
            futs = [ex.submit(process_patient_chunk, chunk) for chunk in chunks]
            for fut in as_completed(futs):
                for i, values in fut.result():
                    store_result(i, values)
                    processed += 1
                if processed == n_p or processed % max(100, chunk_size) < chunk_size:
                    print(f"  Processed {processed}/{n_p}")

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
    if args.out_pickle:
        import pickle

        with Path(args.out_pickle).open("wb") as f:
            pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDone. Saved {args.out_csv}")


if __name__ == "__main__":
    main()
