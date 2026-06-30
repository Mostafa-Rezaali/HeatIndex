from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    daily_cache_reusable,
    load_daily_array,
    load_dates_from_mat,
    matlab_prctile_nan_last_axis,
    month,
    retry,
    saturation_vapor_pressure_hpa,
    save_daily_cache,
    year,
    yyyymmdd,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PRISM HI/T daily exceedance magnitude NetCDF files.")
    p.add_argument("--pct", type=float, default=90)
    p.add_argument("--t2m-var", choices=["tmean", "tmax"], default="tmax")
    p.add_argument("--dates-mat", default="Dir_MJJAS_HI.mat")
    p.add_argument("--dates-var", default="dates")
    p.add_argument("--sample-raster", default="prism_tdmean_us_30s_19810101.tif")
    p.add_argument("--base-url", default="https://data.prism.oregonstate.edu/time_series/us/an/800m")
    p.add_argument("--cache-root", default="PRISM_cache")
    p.add_argument("--tmp-root", default="_HI_tmp")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold-workers", type=int, default=1)
    p.add_argument("--slice-workers", type=int, default=1)
    p.add_argument("--block-rows", type=int, default=64)
    p.add_argument("--keep-python-cache", action="store_true")
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
    out_cache_day = tmp_root / f"HI_{ds}.npz"
    if daily_cache_reusable(out_cache_day, args.t2m_var):
        return
    for stale in (out_cache_day, tmp_root / f"HI_{ds}.mat"):
        try:
            if stale.is_file():
                stale.unlink()
        except OSError:
            pass

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
        save_daily_cache(out_cache_day, hi, t2_raw.astype(np.float32), args.t2m_var)
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
    threshold_workers: int,
) -> np.ndarray:
    idx_dates = [d for d in dates_all if d.month == month_value]
    out = np.full((nlat, nlon), np.nan, dtype=np.float32)
    blocks = [(r0, min(nlat, r0 + block_rows)) for r0 in range(0, nlat, block_rows)]
    worker_count = max(1, min(int(threshold_workers), len(blocks)))
    print(f"    threshold row blocks: {len(blocks)}; workers: {worker_count}")
    if worker_count == 1:
        for r0, r1 in blocks:
            br0, br1, block, warnings = threshold_block(tmp_root, idx_dates, var_name, nlon, pct, r0, r1)
            for warning in warnings:
                print(warning)
            out[br0:br1, :] = block
            print(f"    rows {br0 + 1}-{br1}/{nlat} done")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = [
                ex.submit(threshold_block, tmp_root, idx_dates, var_name, nlon, pct, r0, r1)
                for r0, r1 in blocks
            ]
            for fut in as_completed(futures):
                br0, br1, block, warnings = fut.result()
                for warning in warnings:
                    print(warning)
                out[br0:br1, :] = block
                print(f"    rows {br0 + 1}-{br1}/{nlat} done")
    return out


def threshold_block(
    tmp_root: Path,
    idx_dates: list[datetime],
    var_name: str,
    nlon: int,
    pct: float,
    r0: int,
    r1: int,
) -> tuple[int, int, np.ndarray, list[str]]:
    warnings: list[str] = []
    cube = np.full((r1 - r0, nlon, len(idx_dates)), np.nan, dtype=np.float32)
    for k, d in enumerate(idx_dates):
        mpath = tmp_root / f"HI_{d:%Y%m%d}.npz"
        if not mpath.is_file():
            warnings.append(f"    [warn] missing cache {d:%Y%m%d}")
            continue
        try:
            cube[:, :, k] = load_daily_array(mpath, var_name, slice(r0, r1))
        except Exception as exc:
            warnings.append(f"    [warn] unreadable cache {d:%Y%m%d}: {exc}")
            continue
    block = matlab_prctile_nan_last_axis(cube, pct).astype(np.float32)
    return r0, r1, block, warnings


def mag_slice_for_day(
    d: datetime,
    tmp_root: Path,
    build_hi_mag: bool,
    build_t_mag: bool,
    thr_hi_month: np.ndarray | None,
    thr_t_month: np.ndarray | None,
    nlat: int,
    nlon: int,
) -> tuple[int, np.ndarray | None, np.ndarray | None, list[str]]:
    ds = d.strftime("%Y%m%d")
    warnings: list[str] = []
    time_value = yyyymmdd(d)
    hi_mag = None
    t_mag = None
    mpath = tmp_root / f"HI_{ds}.npz"

    if mpath.is_file():
        try:
            hi = load_daily_array(mpath, "HI") if build_hi_mag else None
            t2 = load_daily_array(mpath, "T2") if build_t_mag else None
        except Exception as exc:
            warnings.append(f"Unreadable cache {ds}; writing date + NaN slice. {exc}")
            hi = None
            t2 = None
    else:
        warnings.append(f"Missing cache {ds}; writing date + NaN slice.")
        hi = None
        t2 = None

    if build_hi_mag:
        if hi is not None and thr_hi_month is not None:
            hi_mag = (hi - thr_hi_month).astype(np.float32)
            hi_mag[(hi_mag <= 0) | np.isnan(hi)] = np.nan
        else:
            hi_mag = np.full((nlat, nlon), np.nan, dtype=np.float32)

    if build_t_mag:
        if t2 is not None and thr_t_month is not None:
            t_mag = (t2 - thr_t_month).astype(np.float32)
            t_mag[(t_mag <= 0) | np.isnan(t2)] = np.nan
        else:
            t_mag = np.full((nlat, nlon), np.nan, dtype=np.float32)

    return time_value, hi_mag, t_mag, warnings


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
    dates_all = load_dates_from_mat(args.dates_mat, args.dates_var)
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
        f_hi = tmp_root / f"HI_THR_{pct_tag}_{mm}.npy"
        f_t = tmp_root / f"T_THR_{pct_tag}_{mm}.npy"
        if not f_hi.is_file():
            print(f"  Month {mm:02d}: building HI threshold")
            thr = threshold_for_month(dates_all, tmp_root, mm, "HI", nlat, nlon, args.pct, args.block_rows, args.threshold_workers)
            np.save(f_hi, thr.astype(np.float32))
        if not f_t.is_file():
            print(f"  Month {mm:02d}: building T threshold")
            thr = threshold_for_month(dates_all, tmp_root, mm, "T2", nlat, nlon, args.pct, args.block_rows, args.threshold_workers)
            np.save(f_t, thr.astype(np.float32))

    if build_hi_mag or build_t_mag:
        print("Stage 3: writing daily exceedance slices ...")
        thr_hi = {}
        thr_t = {}
        for mm in range(5, 10):
            if build_hi_mag:
                thr_hi[mm] = np.asarray(np.load(tmp_root / f"HI_THR_{pct_tag}_{mm}.npy"), dtype=np.float32)
            if build_t_mag:
                thr_t[mm] = np.asarray(np.load(tmp_root / f"T_THR_{pct_tag}_{mm}.npy"), dtype=np.float32)

        ds_hi = create_mag_nc(out_mag_hi, "HI_EXCDMAG", nlat, nlon, len(dates_mjjas), lat, lon, args.pct, y1, y2, f"Heat Index (from {args.t2m_var})") if build_hi_mag else None
        ds_t = create_mag_nc(out_mag_t, "T_EXCDMAG", nlat, nlon, len(dates_mjjas), lat, lon, args.pct, y1, y2, f"Temperature ({args.t2m_var})") if build_t_mag else None
        nan_slice = np.full((nlat, nlon), np.nan, dtype=np.float32)
        slice_workers = max(1, int(args.slice_workers))

        try:
            print(f"  slice prep workers: {slice_workers}")
            executor = ThreadPoolExecutor(max_workers=slice_workers) if slice_workers > 1 else None
            for batch_start in range(0, len(dates_mjjas), slice_workers):
                batch = list(enumerate(dates_mjjas[batch_start : batch_start + slice_workers], start=batch_start))
                if executor is None:
                    results = [
                        (
                            ti,
                            mag_slice_for_day(
                                d,
                                tmp_root,
                                build_hi_mag,
                                build_t_mag,
                                thr_hi.get(d.month) if build_hi_mag else None,
                                thr_t.get(d.month) if build_t_mag else None,
                                nlat,
                                nlon,
                            ),
                        )
                        for ti, d in batch
                    ]
                else:
                    futs = {
                        executor.submit(
                            mag_slice_for_day,
                            d,
                            tmp_root,
                            build_hi_mag,
                            build_t_mag,
                            thr_hi.get(d.month) if build_hi_mag else None,
                            thr_t.get(d.month) if build_t_mag else None,
                            nlat,
                            nlon,
                        ): ti
                        for ti, d in batch
                    }
                    results = [(futs[fut], fut.result()) for fut in as_completed(futs)]

                for ti, (time_value, mag, mag_t, warnings) in sorted(results, key=lambda item: item[0]):
                    d = dates_mjjas[ti]
                    ds = d.strftime("%Y%m%d")
                    for warning in warnings:
                        print(warning)
                    if ds_hi is not None:
                        retry(lambda ti=ti, time_value=time_value: ds_hi["time"].__setitem__(ti, time_value), f"time {ds}")
                        retry(
                            lambda ti=ti, mag=mag: ds_hi["HI_EXCDMAG"].__setitem__((slice(None), slice(None), ti), mag if mag is not None else nan_slice),
                            f"HI_EXCDMAG {ds}",
                        )
                    if ds_t is not None:
                        retry(lambda ti=ti, time_value=time_value: ds_t["time"].__setitem__(ti, time_value), f"time {ds}")
                        retry(
                            lambda ti=ti, mag_t=mag_t: ds_t["T_EXCDMAG"].__setitem__((slice(None), slice(None), ti), mag_t if mag_t is not None else nan_slice),
                            f"T_EXCDMAG {ds}",
                        )
                    if ti == 0 or (ti + 1) % 50 == 0 or ti + 1 == len(dates_mjjas):
                        print(f"  wrote slice {ti + 1}/{len(dates_mjjas)} ({ds})")
        finally:
            if "executor" in locals() and executor is not None:
                executor.shutdown()
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
    if not args.keep_python_cache:
        cleanup_day(tmp_root, cache_root)


if __name__ == "__main__":
    main()
