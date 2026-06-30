from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import shutil
import zipfile

import netCDF4
import numpy as np
import rasterio
import requests

from heatindex.utils import (
    compute_heat_index_c,
    daily_h5_reusable,
    load_daily_array,
    load_dates_from_mat,
    matlab_prctile_nan_last_axis,
    month,
    retry,
    saturation_vapor_pressure_hpa,
    save_daily_h5,
    save_mat_variable,
    year,
    yyyymmdd,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PRISM HI/T daily exceedance magnitude NetCDF files.")
    p.add_argument("--pct", type=float, default=90)
    p.add_argument("--t2m-var", choices=["tmean", "tmax"], default="tmax")
    p.add_argument("--dates-mat", default="Dir_MJJAS_HI.mat")
    p.add_argument("--sample-raster", default="prism_tdmean_us_30s_19810101.tif")
    p.add_argument("--base-url", default="https://data.prism.oregonstate.edu/time_series/us/an/800m")
    p.add_argument("--cache-root", default="PRISM_cache")
    p.add_argument("--tmp-root", default="_HI_tmp")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--block-rows", type=int, default=64)
    return p.parse_args()


def pick_raster(folder: Path) -> Path:
    for pattern in ("*.bil", "*.tif"):
        found = sorted(folder.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"No .bil/.tif raster found in {folder}")


def download_file(url: str, path: Path, timeout: int = 180) -> None:
    if path.is_file():
        return
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(path)


def unzip_once(zip_path: Path, out_dir: Path) -> None:
    if out_dir.is_dir():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def read_prism_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
    arr[arr == -9999] = np.nan
    return arr


def cleanup_day(*paths: Path) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
            tmp = p.with_suffix(p.suffix + ".tmp")
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass


def build_one_day(d: datetime, args: argparse.Namespace, cache_root: Path, tmp_root: Path) -> None:
    ds = d.strftime("%Y%m%d")
    out_mat_day = tmp_root / f"HI_{ds}.mat"
    if daily_h5_reusable(out_mat_day, args.t2m_var):
        return

    u_t2 = f"{args.base_url}/{args.t2m_var}/daily/{d.year}/prism_{args.t2m_var}_us_30s_{ds}.zip"
    u_td = f"{args.base_url}/tdmean/daily/{d.year}/prism_tdmean_us_30s_{ds}.zip"
    z_t2 = cache_root / f"{args.t2m_var}_{ds}.zip"
    z_td = cache_root / f"tdmean_{ds}.zip"
    d_t2 = cache_root / f"unz_{args.t2m_var}_{ds}"
    d_td = cache_root / f"unz_tdmean_{ds}"

    try:
        download_file(u_t2, z_t2)
        unzip_once(z_t2, d_t2)
        download_file(u_td, z_td)
        unzip_once(z_td, d_td)

        t2_raw = read_prism_raster(pick_raster(d_t2))
        td = read_prism_raster(pick_raster(d_td))
        rh = 100.0 * (saturation_vapor_pressure_hpa(td) / saturation_vapor_pressure_hpa(t2_raw))
        hi = compute_heat_index_c(t2_raw, rh).astype(np.float32)
        save_daily_h5(out_mat_day, hi, t2_raw.astype(np.float32), args.t2m_var)
    except Exception as exc:
        print(f"Failed {ds}: {exc}")
    finally:
        # Match the MATLAB flow: keep only the reusable HI/T2 mat, and remove
        # this day's PRISM archives/extracted rasters after success or failure.
        cleanup_day(z_t2, z_td, d_t2, d_td)


def lat_lon_from_sample(sample_raster: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(sample_raster) as src:
        rows = np.arange(src.height)
        cols = np.arange(src.width)
        xs, _ = rasterio.transform.xy(src.transform, np.zeros_like(cols), cols, offset="center")
        _, ys = rasterio.transform.xy(src.transform, rows, np.zeros_like(rows), offset="center")
    return np.asarray(ys, dtype=np.float32), np.asarray(xs, dtype=np.float32)


def threshold_for_month(
    dates_all: list[datetime],
    tmp_root: Path,
    month_value: int,
    var_name: str,
    nlat: int,
    nlon: int,
    pct: float,
    block_rows: int,
) -> np.ndarray:
    idx_dates = [d for d in dates_all if d.month == month_value]
    out = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for r0 in range(0, nlat, block_rows):
        r1 = min(nlat, r0 + block_rows)
        cube = np.full((r1 - r0, nlon, len(idx_dates)), np.nan, dtype=np.float32)
        for k, d in enumerate(idx_dates):
            mpath = tmp_root / f"HI_{d:%Y%m%d}.mat"
            if not mpath.is_file():
                print(f"    [warn] missing mat {d:%Y%m%d}")
                continue
            cube[:, :, k] = load_daily_array(mpath, var_name, slice(r0, r1))
        out[r0:r1, :] = matlab_prctile_nan_last_axis(cube, pct).astype(np.float32)
        print(f"    rows {r0 + 1}-{r1}/{nlat} done")
    return out


def create_mag_nc(
    out_nc: Path,
    varname: str,
    nlat: int,
    nlon: int,
    n_time: int,
    lat: np.ndarray,
    lon: np.ndarray,
    pct: float,
    y1: int,
    y2: int,
    vshort: str,
) -> netCDF4.Dataset:
    if out_nc.exists():
        out_nc.unlink()
    ds = netCDF4.Dataset(out_nc, "w", format="NETCDF4")
    ds.createDimension("lat", nlat)
    ds.createDimension("lon", nlon)
    ds.createDimension("time", n_time)
    v = ds.createVariable(varname, "f4", ("lat", "lon", "time"), zlib=True, complevel=4, chunksizes=(nlat, nlon, 1))
    ds.createVariable("lat", "f4", ("lat",))[:] = lat
    ds.createVariable("lon", "f4", ("lon",))[:] = lon
    ds.createVariable("time", "f8", ("time",))
    ds.title = f"PRISM {vshort} exceedance MAGNITUDE (degC), daily MJJAS, {y1}-{y2}"
    ds.summary = (
        f"Daily {vshort} exceedance magnitude above *climatological* monthly P{pct:g}.\n"
        f"For each month (May-Sep), the threshold is P{pct:g} computed from ALL years ({y1}-{y2}) at each grid cell.\n"
        f"{varname} = value_day - monthly_threshold; non-exceedances are NaN."
    )
    ds.source = "Derived from PRISM temperature and tdmean (800m); NOAA/Steadman HI formulation for HI."
    ds.Conventions = "CF-1.8"
    ds.history = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds["lat"].standard_name = "latitude"
    ds["lat"].long_name = "latitude"
    ds["lat"].units = "degrees_north"
    ds["lon"].standard_name = "longitude"
    ds["lon"].long_name = "longitude"
    ds["lon"].units = "degrees_east"
    ds["time"].long_name = "date in yyyymmdd format (MJJAS only)"
    ds["time"].units = "yyyymmdd"
    ds["time"].calendar = "gregorian"
    v.units = "degree_Celsius"
    v.long_name = f"Daily {vshort} exceedance magnitude above climatological monthly P{pct:g} (NaN if not exceeding)"
    v.coordinates = "lat lon time"
    v.missing_value = np.float32(np.nan)
    return ds


def main() -> None:
    args = parse_args()
    print("=== PRISM Daily Exceedance MAG: HI + T2m (climatological monthly thresholds) ===")
    dates_all = load_dates_from_mat(args.dates_mat)
    years = sorted({d.year for d in dates_all})
    months_sel = set(range(5, 10))
    dates_mjjas = [d for d in dates_all if d.month in months_sel]
    y1, y2 = years[0], years[-1]

    pct_tag = f"{int(args.pct)}" if float(args.pct).is_integer() else f"{args.pct:g}"
    out_mag_hi = Path(f"HI_EXCDMAG_daily_{y1}_{y2}_{pct_tag}.nc")
    out_mag_t = Path(f"T_EXCDMAG_daily_{y1}_{y2}_{pct_tag}.nc")
    build_hi_mag = not out_mag_hi.is_file()
    build_t_mag = not out_mag_t.is_file()
    print(f"HI mag ({out_mag_hi}): {'BUILD' if build_hi_mag else 'exists, skip'}")
    print(f"T  mag ({out_mag_t}): {'BUILD' if build_t_mag else 'exists, skip'}")

    cache_root = Path(args.cache_root)
    tmp_root = Path(args.tmp_root)
    cache_root.mkdir(exist_ok=True)
    tmp_root.mkdir(exist_ok=True)

    print("Reading grid from sample raster...")
    lat, lon = lat_lon_from_sample(Path(args.sample_raster))
    nlat, nlon = len(lat), len(lon)

    print(f"Stage 1: building daily mats (HI from {args.t2m_var} + tdmean, T2 = {args.t2m_var}) ...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(build_one_day, d, args, cache_root, tmp_root) for d in dates_mjjas]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i == 1 or i % 50 == 0 or i == len(futs):
                print(f"  processed {i}/{len(futs)} MJJAS days")
    print("Stage 1 complete.")

    print(f"Stage 2: monthly thresholds at P{args.pct:g} (HI and T) ...")
    for mm in range(5, 10):
        f_hi = Path(f"HI_THR_{pct_tag}_{mm}.mat")
        f_t = Path(f"T_THR_{pct_tag}_{mm}.mat")
        if not f_hi.is_file():
            print(f"  Month {mm:02d}: building HI threshold")
            thr = threshold_for_month(dates_all, tmp_root, mm, "HI", nlat, nlon, args.pct, args.block_rows)
            save_mat_variable(f_hi, "tmp", thr)
        if not f_t.is_file():
            print(f"  Month {mm:02d}: building T threshold")
            thr = threshold_for_month(dates_all, tmp_root, mm, "T2", nlat, nlon, args.pct, args.block_rows)
            save_mat_variable(f_t, "tmp", thr)

    if build_hi_mag or build_t_mag:
        print("Stage 3: writing daily exceedance slices ...")
        thr_hi = {}
        thr_t = {}
        for mm in range(5, 10):
            if build_hi_mag:
                from heatindex.utils import load_mat_variable

                thr_hi[mm] = np.asarray(load_mat_variable(f"HI_THR_{pct_tag}_{mm}.mat"), dtype=np.float32)
            if build_t_mag:
                from heatindex.utils import load_mat_variable

                thr_t[mm] = np.asarray(load_mat_variable(f"T_THR_{pct_tag}_{mm}.mat"), dtype=np.float32)

        ds_hi = create_mag_nc(out_mag_hi, "HI_EXCDMAG", nlat, nlon, len(dates_mjjas), lat, lon, args.pct, y1, y2, f"Heat Index (from {args.t2m_var})") if build_hi_mag else None
        ds_t = create_mag_nc(out_mag_t, "T_EXCDMAG", nlat, nlon, len(dates_mjjas), lat, lon, args.pct, y1, y2, f"Temperature ({args.t2m_var})") if build_t_mag else None
        nan_slice = np.full((nlat, nlon), np.nan, dtype=np.float32)

        try:
            for ti, d in enumerate(dates_mjjas):
                ds = d.strftime("%Y%m%d")
                mpath = tmp_root / f"HI_{ds}.mat"
                if mpath.is_file():
                    hi = load_daily_array(mpath, "HI")
                    t2 = load_daily_array(mpath, "T2")
                    if ds_hi is not None:
                        mag = (hi - thr_hi[d.month]).astype(np.float32)
                        mag[(mag <= 0) | np.isnan(hi)] = np.nan
                        retry(lambda: ds_hi["time"].__setitem__(ti, yyyymmdd(d)), f"time {ds}")
                        retry(lambda: ds_hi["HI_EXCDMAG"].__setitem__((slice(None), slice(None), ti), mag), f"HI_EXCDMAG {ds}")
                    if ds_t is not None:
                        mag_t = (t2 - thr_t[d.month]).astype(np.float32)
                        mag_t[(mag_t <= 0) | np.isnan(t2)] = np.nan
                        retry(lambda: ds_t["time"].__setitem__(ti, yyyymmdd(d)), f"time {ds}")
                        retry(lambda: ds_t["T_EXCDMAG"].__setitem__((slice(None), slice(None), ti), mag_t), f"T_EXCDMAG {ds}")
                else:
                    print(f"Missing mat {ds}; writing date + NaN slice.")
                    if ds_hi is not None:
                        ds_hi["time"][ti] = yyyymmdd(d)
                        ds_hi["HI_EXCDMAG"][:, :, ti] = nan_slice
                    if ds_t is not None:
                        ds_t["time"][ti] = yyyymmdd(d)
                        ds_t["T_EXCDMAG"][:, :, ti] = nan_slice
                if ti == 0 or (ti + 1) % 50 == 0 or ti + 1 == len(dates_mjjas):
                    print(f"  wrote slice {ti + 1}/{len(dates_mjjas)} ({ds})")
        finally:
            if ds_hi is not None:
                ds_hi.close()
            if ds_t is not None:
                ds_t.close()
    else:
        print("Stage 3: both mag files already exist, nothing to write.")

    print("\nFinished.")
    if build_hi_mag:
        print(f"HI MAG : {out_mag_hi}")
    if build_t_mag:
        print(f"T  MAG : {out_mag_t}")


if __name__ == "__main__":
    main()
