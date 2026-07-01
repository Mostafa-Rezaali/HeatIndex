from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle

import netCDF4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, convex_hull_image, disk

from heatindex.utils import (
    haversine_km,
    interp_grid_vector,
    load_mat_variable,
    masked_to_nan,
    matlab_round_positive,
    retry,
    yyyymmdd_to_datetime,
)


@dataclass
class DayResult:
    is_hw: bool
    ahsci: float
    area_km2: float
    prop_hw: float
    union_idx: np.ndarray
    cluster_hulls: list[np.ndarray]
    cluster_cents: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect HI heat-wave days and write HSCI NetCDF.")
    p.add_argument("--pct", type=int, default=90)
    p.add_argument("--mag-nc", default=None)
    p.add_argument("--area-mat", default="/blue/nessie/mostafarezaali/Research_Project/DayMet_FL/CR_Areas.mat")
    p.add_argument("--area-var", default="Area")
    p.add_argument("--out-nc", default=None)
    p.add_argument("--det-ckpt", default=None)
    p.add_argument("--write-ckpt", default=None)
    p.add_argument("--min-area-km2", type=float, default=1000.0)
    p.add_argument("--pixel-size-km", type=float, default=0.8)
    p.add_argument("--cluster-cutoff-km", type=float, default=1000.0)
    p.add_argument("--min-duration", type=int, default=3)
    p.add_argument("--grace-days", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--write-workers", type=int, default=1)
    p.add_argument("--keep-checkpoints", action="store_true")
    return p.parse_args()


def read_grid_vectors(nc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(nc_path) as ds:
        if "lat" in ds.variables and "lon" in ds.variables:
            y = np.asarray(ds["lat"][:], dtype=np.float64)
            x = np.asarray(ds["lon"][:], dtype=np.float64)
        elif "y" in ds.variables and "x" in ds.variables:
            y = np.asarray(ds["y"][:], dtype=np.float64)
            x = np.asarray(ds["x"][:], dtype=np.float64)
        else:
            raise KeyError("MAG NetCDF must contain lat/lon or y/x coordinate variables.")
    return y, x


def read_mag_time(nc_path: Path) -> tuple[np.ndarray, list[datetime], np.ndarray, tuple[int, int, int]]:
    with netCDF4.Dataset(nc_path) as ds:
        tvec = np.asarray(ds["time"][:], dtype=np.float64)
        shape = ds["HI_EXCDMAG"].shape
    dates_all = yyyymmdd_to_datetime(tvec)
    keep = np.array([5 <= d.month <= 9 for d in dates_all], dtype=bool)
    idx_all = np.nonzero(keep)[0]
    dates = [d for d, k in zip(dates_all, keep) if k]
    return tvec, dates, idx_all, shape


def morph_heat_mask(mask: np.ndarray) -> np.ndarray:
    return binary_closing(binary_opening(mask, disk(1)), disk(2))


def flat_indices_f(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.ravel(mask, order="F")).astype(np.uint32)


def restore_mask_from_indices(shape: tuple[int, int], idx: np.ndarray) -> np.ndarray:
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    if idx.size:
        out[idx.astype(np.int64)] = True
    return out.reshape(shape, order="F")


def process_one_day(
    mag_nc: str,
    ti_nc: int,
    ny: int,
    nx: int,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    pix_area_km2: float,
    min_pixels: int,
    region_area: float,
    cluster_cutoff_km: float,
) -> DayResult:
    with netCDF4.Dataset(mag_nc) as ds:
        excd = masked_to_nan(ds["HI_EXCDMAG"][:, :, ti_nc]).astype(np.float32)

    empty = DayResult(False, np.nan, np.nan, np.nan, np.array([], dtype=np.uint32), [np.array([], dtype=np.uint32)], np.zeros((0, 2)))
    if np.count_nonzero(excd > 0) <= 90:
        return empty

    lbl = morph_heat_mask(excd > 0)
    cc = label(lbl, connectivity=2)
    props = regionprops(cc)
    idx_big = [pr.label for pr in props if pr.area > min_pixels]
    if idx_big:
        keep = np.isin(cc, idx_big)
        excd[~keep] = np.nan
    else:
        excd[:] = np.nan

    if np.count_nonzero(excd > 0) <= min_pixels:
        return empty

    lbl = morph_heat_mask(excd > 0)
    cc = label(lbl, connectivity=2)
    props = regionprops(cc)
    convex_hull_mask = np.zeros((ny, nx), dtype=bool)
    pix_lists: list[np.ndarray] = []
    centroids_ll: list[tuple[float, float]] = []

    if len(props) > 1:
        centroids = np.array([pr.centroid for pr in props], dtype=np.float64)
        rows_intr = matlab_round_positive(centroids[:, 0] + 1.0)
        cols_intr = matlab_round_positive(centroids[:, 1] + 1.0)
        rows_intr = np.clip(rows_intr, 1, ny)
        cols_intr = np.clip(cols_intr, 1, nx)
        lat_c = lat_grid[rows_intr - 1]
        lon_c = lon_grid[cols_intr - 1]
        xy = np.column_stack([lon_c, lat_c])

        if xy.shape[0] >= 2:
            dkm = pdist(xy, lambda a, b: haversine_km(a[1], a[0], b[1], b[0]))
            z = linkage(dkm, method="single")
            labels = fcluster(z, t=cluster_cutoff_km, criterion="distance")
        else:
            labels = np.ones(xy.shape[0], dtype=np.int64)

        cluster_label_img = np.zeros_like(cc, dtype=np.uint16)
        for pr, lab in zip(props, labels):
            cluster_label_img[cc == pr.label] = int(lab)

        for lab in sorted(set(labels.tolist())):
            msk = cluster_label_img == lab
            if not np.any(msk):
                continue
            ch = convex_hull_image(msk)
            convex_hull_mask |= ch
            idx = flat_indices_f(ch)
            pix_lists.append(idx)
            rr, cc0 = np.nonzero(ch)
            r0 = float(np.mean(rr + 1.0))
            c0 = float(np.mean(cc0 + 1.0))
            centroids_ll.append((float(interp_grid_vector(lat_grid, np.array([r0]))[0]), float(interp_grid_vector(lon_grid, np.array([c0]))[0])))
    else:
        ch = convex_hull_image(lbl)
        convex_hull_mask = ch
        idx = flat_indices_f(ch)
        pix_lists = [idx]
        rr, cc0 = np.nonzero(ch)
        r0 = float(np.mean(rr + 1.0))
        c0 = float(np.mean(cc0 + 1.0))
        centroids_ll = [(float(interp_grid_vector(lat_grid, np.array([r0]))[0]), float(interp_grid_vector(lon_grid, np.array([c0]))[0]))]

    area_km2 = float(np.count_nonzero(convex_hull_mask) * pix_area_km2)
    if area_km2 <= 1000.0:
        return empty

    prop_hw = float(np.count_nonzero(lbl) / np.count_nonzero(convex_hull_mask))
    areg = (prop_hw * area_km2) / float(region_area)
    mean_in = float(np.nanmean(excd[lbl]))
    ahsci = mean_in * areg
    return DayResult(
        True,
        ahsci,
        area_km2,
        prop_hw,
        flat_indices_f(convex_hull_mask),
        pix_lists,
        np.asarray(centroids_ll, dtype=np.float64),
    )


def detect_heatwaves_by_year(
    is_hw: np.ndarray,
    years: np.ndarray,
    min_duration: int,
    grace_days: int,
) -> np.ndarray:
    events = np.zeros(is_hw.shape[0], dtype=np.int32)
    event_id = 1
    for yr in np.unique(years):
        idx = np.nonzero(years == yr)[0]
        pos = 0
        while pos < idx.size:
            if not is_hw[idx[pos]]:
                pos += 1
                continue

            run_positions = []
            hot_count = 0
            gap_count = 0
            j = pos

            while j < idx.size:
                day_idx = idx[j]
                if is_hw[day_idx]:
                    run_positions.append(day_idx)
                    hot_count += 1
                    gap_count = 0
                    j += 1
                    continue

                # Grace is based on post-rule HSCI-H days: this gap is allowed
                # only between two days that passed the spatial/min-area rules.
                if gap_count < grace_days and j + 1 < idx.size and is_hw[idx[j + 1]]:
                    gap_count += 1
                    j += 1
                    continue

                break

            if hot_count >= min_duration:
                events[np.asarray(run_positions, dtype=np.int64)] = event_id
                event_id += 1
            pos = max(j, pos + 1)
    return events


def detection_checkpoint_matches(det: dict, idx_all: np.ndarray, min_duration: int, grace_days: int) -> bool:
    return (
        int(det.get("min_duration", -1)) == int(min_duration)
        and int(det.get("grace_days", -1)) == int(grace_days)
        and np.array_equal(det.get("idx_all"), idx_all)
    )


def sets_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    if a.size == 0 or b.size == 0:
        return False
    if a.size <= b.size:
        return np.intersect1d(a, b, assume_unique=False).size > 0
    return np.intersect1d(b, a, assume_unique=False).size > 0


def resolve_root(remap: dict[int, int], rid: int) -> int:
    while rid in remap:
        rid = remap[rid]
    return rid


def link_events_and_dynamics(day_idx, dates, day_clusters, day_cents, lat_grid, lon_grid):
    n_hw = len(day_idx)
    event_ids_per_day: list[np.ndarray] = []
    next_id = 1
    remap: dict[int, int] = {}

    for k in range(n_hw):
        this_clusters = day_clusters[k]
        this_ids = np.zeros(len(this_clusters), dtype=np.uint32)
        for c, pix_a in enumerate(this_clusters):
            pri: list[int] = []
            for j in range(max(0, k - 3), k):
                for pc, prev_pix in enumerate(day_clusters[j]):
                    if sets_overlap(pix_a, prev_pix):
                        pri.append(int(event_ids_per_day[j][pc]))
            pri = sorted({p for p in pri if p != 0})
            if not pri:
                this_ids[c] = next_id
                next_id += 1
            else:
                roots = [resolve_root(remap, p) for p in pri]
                keep_id = min(roots)
                this_ids[c] = keep_id
                for rid in roots:
                    if rid != keep_id:
                        remap[rid] = keep_id
        for c, idc in enumerate(this_ids):
            if idc != 0:
                this_ids[c] = resolve_root(remap, int(idc))
        event_ids_per_day.append(this_ids)

    n_clusters = np.zeros(n_hw, dtype=np.uint16)
    n_merges = np.zeros(n_hw, dtype=np.uint16)
    n_splits = np.zeros(n_hw, dtype=np.uint16)
    min_pair_km = np.full(n_hw, np.nan, dtype=np.float32)
    mean_pair_km = np.full(n_hw, np.nan, dtype=np.float32)
    d_mean_pair_km = np.full(n_hw, np.nan, dtype=np.float32)

    rows = []
    for k in range(n_hw):
        clusters = day_clusters[k]
        n_clusters[k] = len(clusters)
        cents = day_cents[k]
        if cents.shape[0] >= 2:
            d = pdist(cents, lambda a, b: haversine_km(a[0], a[1], b[0], b[1]))
            min_pair_km[k] = np.float32(np.min(d))
            mean_pair_km[k] = np.float32(np.mean(d))
        if k >= 1 and np.isfinite(mean_pair_km[k - 1]) and np.isfinite(mean_pair_km[k]):
            d_mean_pair_km[k] = np.float32(mean_pair_km[k] - mean_pair_km[k - 1])

        if k >= 1 and day_clusters[k - 1] and clusters:
            overlap = np.zeros((len(day_clusters[k - 1]), len(clusters)), dtype=bool)
            for p, idx_p in enumerate(day_clusters[k - 1]):
                for q, idx_q in enumerate(clusters):
                    overlap[p, q] = sets_overlap(idx_p, idx_q)
            n_splits[k] = np.uint16(np.sum(np.sum(overlap, axis=1) >= 2))
            n_merges[k] = np.uint16(np.sum(np.sum(overlap, axis=0) >= 2))

        for c, eid in enumerate(event_ids_per_day[k], start=1):
            rows.append({"day_idx": int(day_idx[k]), "cluster_idx": c, "event_id": int(eid), "date": dates[day_idx[k]].strftime("%Y-%m-%d")})

    clusters_df = pd.DataFrame(rows, columns=["day_idx", "cluster_idx", "event_id", "date"])
    if not clusters_df.empty:
        events_df = clusters_df.groupby("event_id", as_index=False).size().rename(columns={"size": "duration_days"})
    else:
        events_df = pd.DataFrame(columns=["event_id", "duration_days"])

    return event_ids_per_day, clusters_df, events_df, n_clusters, n_merges, n_splits, min_pair_km, mean_pair_km, d_mean_pair_km


def create_output_nc(
    path: Path,
    ny: int,
    nx: int,
    n_hw: int,
    x: np.ndarray,
    y: np.ndarray,
    pct: int,
    arrays: dict,
    min_duration: int,
    grace_days: int,
):
    if path.exists():
        path.unlink()
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.percentile = int(pct)
    ds.min_duration_days = int(min_duration)
    ds.grace_days = int(grace_days)
    ds.heatwave_rule = (
        "At least min_duration_days valid HSCI-H days after spatial/min-area "
        "heatwave rules, allowing up to grace_days consecutive post-rule "
        "non-HSCI days to bridge an event; grace days are not written as HSCI "
        "days."
    )
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    ds.createDimension("time", n_hw)
    ds.createVariable("EXCD", "f4", ("y", "x", "time"), zlib=True, complevel=4, chunksizes=(ny, nx, 1))
    ds.createVariable("time", "f8", ("time",))
    ds.createVariable("x", "f8", ("x",))
    ds.createVariable("y", "f8", ("y",))
    for name, dtype in [
        ("HSCI", "f4"),
        ("area_km2", "f4"),
        ("propHW", "f4"),
        ("n_clusters", "u2"),
        ("n_merges", "u2"),
        ("n_splits", "u2"),
        ("min_pair_km", "f4"),
        ("mean_pair_km", "f4"),
        ("d_mean_pair_km", "f4"),
    ]:
        ds.createVariable(name, dtype, ("time",))
    ds["EXCD"].long_name = f"Daily exceedance magnitude (HI - climatological monthly P{pct}), pruned and hull-restricted"
    ds["EXCD"].units = "degree_Celsius"
    ds["time"].long_name = "date as YYYYMMDD"
    ds["time"].units = "YYYYMMDD"
    ds["x"].long_name = "Longitude"
    ds["x"].units = "degree_east"
    ds["y"].long_name = "Latitude"
    ds["y"].units = "degree_north"
    ds["HSCI"].long_name = "daily HSCI (m_i * a_i)"
    ds["HSCI"].units = "degree_Celsius"
    ds["area_km2"].long_name = "area of union convex hull"
    ds["area_km2"].units = "km^2"
    ds["propHW"].long_name = "proportion of hull flagged hot"
    ds["propHW"].units = "1"
    ds["n_clusters"].long_name = "Number of heat-wave clusters detected this day"
    ds["n_merges"].long_name = "Count of merges (>=2 prev clusters overlapping one current)"
    ds["n_splits"].long_name = "Count of splits (one prev cluster overlapping >=2 current)"
    ds["min_pair_km"].long_name = "Minimum centroid distance among clusters (great-circle)"
    ds["min_pair_km"].units = "km"
    ds["mean_pair_km"].long_name = "Mean centroid distance among clusters (great-circle)"
    ds["mean_pair_km"].units = "km"
    ds["d_mean_pair_km"].long_name = "Change in mean centroid distance vs previous HW day"
    ds["d_mean_pair_km"].units = "km"
    ds["x"][:] = x
    ds["y"][:] = y
    for k, v in arrays.items():
        ds[k][:] = v
    return ds


def build_excd_output_slice(
    k: int,
    mag_nc: str,
    ti_nc: int,
    ny: int,
    nx: int,
    idx_u: np.ndarray,
) -> tuple[int, np.ndarray]:
    with netCDF4.Dataset(mag_nc) as src:
        excd = masked_to_nan(src["HI_EXCDMAG"][:, :, ti_nc]).astype(np.float32)
    if len(idx_u):
        keep = restore_mask_from_indices((ny, nx), idx_u)
        excd[~keep] = np.nan
    else:
        excd[:] = np.nan
    return k, excd


def main() -> None:
    args = parse_args()
    pct = args.pct
    mag_nc = Path(args.mag_nc or f"HI_EXCDMAG_daily_1981_2025_{pct}.nc")
    out_nc = Path(args.out_nc or f"HI_EXCD_MJJAS_HWdays_{pct}.nc")
    det_ckpt = Path(args.det_ckpt or f"HI_EXCD_det_ckpt_{pct}.pkl")
    write_ckpt = Path(args.write_ckpt or f"HI_EXCD_write_ckpt_{pct}.pkl")
    pix_area_km2 = args.pixel_size_km**2
    min_pixels = int(np.ceil(args.min_area_km2 / pix_area_km2))

    y, x = read_grid_vectors(mag_nc)
    _, dates, idx_all, shape = read_mag_time(mag_nc)
    ny, nx = shape[0], shape[1]
    n_t = len(idx_all)
    print(f"Total MJJAS daily slices in MAG file: {n_t}")

    det = None
    if det_ckpt.exists():
        print(f"Found detection checkpoint {det_ckpt}; checking compatibility.")
        with det_ckpt.open("rb") as f:
            candidate = pickle.load(f)
        if detection_checkpoint_matches(candidate, idx_all, args.min_duration, args.grace_days):
            print("Detection checkpoint matches min-duration/grace-days; skipping detection.")
            det = candidate
        else:
            print(
                "Detection checkpoint does not match min-duration/grace-days "
                "or time index; rebuilding detection."
            )

    if det is None:
        region_area = float(np.sum(load_mat_variable(args.area_mat, args.area_var)))
        is_hw = np.zeros(n_t, dtype=bool)
        ahsci_all = np.full(n_t, np.nan, dtype=np.float32)
        area_all = np.full(n_t, np.nan, dtype=np.float32)
        prop_all = np.full(n_t, np.nan, dtype=np.float32)
        union_cells: list[np.ndarray] = [np.array([], dtype=np.uint32) for _ in range(n_t)]
        cluster_hulls: list[list[np.ndarray]] = [[np.array([], dtype=np.uint32)] for _ in range(n_t)]
        cluster_cents: list[np.ndarray] = [np.zeros((0, 2)) for _ in range(n_t)]

        jobs = []
        if args.workers <= 1:
            for i, ti_nc in enumerate(idx_all):
                res = process_one_day(str(mag_nc), int(ti_nc), ny, nx, y, x, pix_area_km2, min_pixels, region_area, args.cluster_cutoff_km)
                jobs.append((i, res))
                if i == 0 or (i + 1) % 250 == 0 or i + 1 == n_t:
                    print(f"  {i + 1:5d}/{n_t:5d} {dates[i]:%Y-%m-%d} HW={int(res.is_hw)} HSCI={res.ahsci}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(process_one_day, str(mag_nc), int(ti_nc), ny, nx, y, x, pix_area_km2, min_pixels, region_area, args.cluster_cutoff_km): i
                    for i, ti_nc in enumerate(idx_all)
                }
                for done, fut in enumerate(as_completed(futs), 1):
                    i = futs[fut]
                    jobs.append((i, fut.result()))
                    if done == 1 or done % 250 == 0 or done == n_t:
                        print(f"  processed {done}/{n_t}")

        for i, res in jobs:
            is_hw[i] = res.is_hw
            ahsci_all[i] = res.ahsci
            area_all[i] = res.area_km2
            prop_all[i] = res.prop_hw
            union_cells[i] = res.union_idx
            cluster_hulls[i] = res.cluster_hulls
            cluster_cents[i] = res.cluster_cents

        years = np.array([d.year for d in dates], dtype=np.int32)
        hw_events = detect_heatwaves_by_year(is_hw, years, args.min_duration, args.grace_days)
        keep_hw = hw_events != 0
        hw_idx = np.nonzero(keep_hw)[0]
        hw_dates = [dates[i] for i in hw_idx]
        print(f"Detected {len(hw_idx)} HW days out of {n_t} total MJJAS days.")

        day_clusters = [cluster_hulls[i] for i in hw_idx]
        day_cents = [cluster_cents[i] for i in hw_idx]
        (
            event_ids_per_day,
            clusters_df,
            events_df,
            n_clusters,
            n_merges,
            n_splits,
            min_pair_km,
            mean_pair_km,
            d_mean_pair_km,
        ) = link_events_and_dynamics(hw_idx, dates, day_clusters, day_cents, y, x)

        hw_time = np.array([int(d.strftime("%Y%m%d")) for d in hw_dates], dtype=np.float64)
        det = {
            "hw_idx": hw_idx,
            "idx_all": idx_all,
            "hw_dates": hw_dates,
            "HW_Time": hw_time,
            "AHSCI": ahsci_all[keep_hw].astype(np.float32),
            "areaKm2": area_all[keep_hw].astype(np.float32),
            "propHW": prop_all[keep_hw].astype(np.float32),
            "n_clusters": n_clusters,
            "n_merges": n_merges,
            "n_splits": n_splits,
            "min_pair_km": min_pair_km,
            "mean_pair_km": mean_pair_km,
            "d_mean_pair_km": d_mean_pair_km,
            "unionIdx_hw": [union_cells[i] for i in hw_idx],
            "xv": x,
            "yv": y,
            "nHW_keep": len(hw_idx),
            "keepHW": keep_hw,
            "isHW": is_hw,
            "dates": dates,
            "months": np.array([d.month for d in dates], dtype=np.int16),
            "min_duration": int(args.min_duration),
            "grace_days": int(args.grace_days),
        }
        with det_ckpt.open("wb") as f:
            pickle.dump(det, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved detection checkpoint {det_ckpt}")

        clusters_df.to_csv(f"HW_events_clusters_{pct}.csv", index=False)
        events_df.to_csv(f"HW_events_summary_{pct}.csv", index=False)

    hw_idx = det["hw_idx"]
    idx_all = det["idx_all"]
    n_hw = int(det["nHW_keep"])
    hw_time = det["HW_Time"]

    ti_start = 0
    fresh = True
    if out_nc.exists() and write_ckpt.exists():
        with write_ckpt.open("rb") as f:
            w = pickle.load(f)
        write_ckpt_matches = (
            w.get("nHW_keep") == n_hw
            and np.array_equal(w.get("HW_Time"), hw_time)
            and int(w.get("min_duration", -1)) == int(args.min_duration)
            and int(w.get("grace_days", -1)) == int(args.grace_days)
        )
        if write_ckpt_matches:
            fresh = False
            ti_start = int(w.get("last_written", -1)) + 1
            print(f"Resuming EXCD write at slice {ti_start + 1}/{n_hw} (file + checkpoint match).")
        else:
            print("Write checkpoint mismatch; recreating output from scratch.")

    if fresh:
        arrays = {
            "time": hw_time,
            "HSCI": det["AHSCI"],
            "area_km2": det["areaKm2"],
            "propHW": det["propHW"],
            "n_clusters": det["n_clusters"],
            "n_merges": det["n_merges"],
            "n_splits": det["n_splits"],
            "min_pair_km": det["min_pair_km"],
            "mean_pair_km": det["mean_pair_km"],
            "d_mean_pair_km": det["d_mean_pair_km"],
        }
        out = create_output_nc(out_nc, ny, nx, n_hw, x, y, pct, arrays, args.min_duration, args.grace_days)
        out.close()
        with write_ckpt.open("wb") as f:
            pickle.dump(
                {
                    "last_written": -1,
                    "nHW_keep": n_hw,
                    "HW_Time": hw_time,
                    "min_duration": int(args.min_duration),
                    "grace_days": int(args.grace_days),
                },
                f,
            )

    write_workers = max(1, int(args.write_workers))
    print(f"Writing EXCD slices {ti_start + 1}..{n_hw} of {n_hw} to {out_nc} ...")
    print(f"  EXCD slice prep workers: {write_workers}")
    executor = None if write_workers == 1 else ProcessPoolExecutor(max_workers=write_workers)
    try:
        with netCDF4.Dataset(out_nc, "a") as dst:
            for batch_start in range(ti_start, n_hw, write_workers):
                batch_stop = min(n_hw, batch_start + write_workers)
                if executor is None:
                    results = [
                        build_excd_output_slice(
                            k,
                            str(mag_nc),
                            int(idx_all[int(hw_idx[k])]),
                            ny,
                            nx,
                            det["unionIdx_hw"][k],
                        )
                        for k in range(batch_start, batch_stop)
                    ]
                else:
                    futs = [
                        executor.submit(
                            build_excd_output_slice,
                            k,
                            str(mag_nc),
                            int(idx_all[int(hw_idx[k])]),
                            ny,
                            nx,
                            det["unionIdx_hw"][k],
                        )
                        for k in range(batch_start, batch_stop)
                    ]
                    results = [fut.result() for fut in as_completed(futs)]

                for k, excd in sorted(results, key=lambda item: item[0]):
                    retry(lambda k=k, excd=excd: dst["EXCD"].__setitem__((slice(None), slice(None), k), excd), f"EXCD slice {k + 1}")
                    if (k + 1) % 25 == 0 or k == ti_start or k + 1 == n_hw:
                        with write_ckpt.open("wb") as f:
                            pickle.dump(
                                {
                                    "last_written": k,
                                    "nHW_keep": n_hw,
                                    "HW_Time": hw_time,
                                    "min_duration": int(args.min_duration),
                                    "grace_days": int(args.grace_days),
                                },
                                f,
                            )
                        print(f"Wrote {k + 1:5d}/{n_hw:5d} {det['hw_dates'][k]:%Y-%m-%d}")
    finally:
        if executor is not None:
            executor.shutdown()
    print("Done writing EXCD stack.")
    if not args.keep_checkpoints:
        for path in (det_ckpt, write_ckpt):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
