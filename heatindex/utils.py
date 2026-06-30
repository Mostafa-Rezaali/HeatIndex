from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import math
import re
import time
from typing import Callable, Iterable

import numpy as np


MATLAB_NAT = None


def matlab_datenum_to_datetime(value: float) -> datetime:
    ordinal = int(value)
    frac = float(value) % 1
    return datetime.fromordinal(ordinal) + timedelta(days=frac) - timedelta(days=366)


def load_mat_variable(path: str | Path, name: str | None = None):
    from scipy.io import loadmat

    path = Path(path)
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in data if not k.startswith("__")]
        if name is not None:
            resolved = resolve_mat_variable_name(keys, name, path)
            return data[resolved]
        if not keys:
            raise KeyError(f"No MATLAB variables found in {path}")
        return data[keys[0]]
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h5:
            keys = [k for k in h5.keys() if not k.startswith("#")]
            if name is None:
                if not keys:
                    raise KeyError(f"No HDF5 MATLAB variables found in {path}")
                name = keys[0]
            resolved = resolve_mat_variable_name(keys, name, path)
            arr = np.array(h5[resolved])
        return np.squeeze(arr)


def mat_variable_names(path: str | Path) -> list[str]:
    from scipy.io import loadmat

    path = Path(path)
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        return [k for k in data if not k.startswith("__")]
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h5:
            return [k for k in h5.keys() if not k.startswith("#")]


def resolve_mat_variable_name(keys: list[str], requested: str, path: Path) -> str:
    if requested in keys:
        return requested
    requested_lc = requested.lower()
    matches = [k for k in keys if k.lower() == requested_lc]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(keys) if keys else "<none>"
    raise KeyError(
        f"Variable '{requested}' was not found in {path}. Available variables: {available}"
    )


def save_mat_variable(path: str | Path, name: str, value) -> None:
    from scipy.io import savemat

    savemat(path, {name: value}, do_compression=False)


def load_dates_from_mat(path: str | Path, var_name: str = "dates") -> list[datetime]:
    try:
        raw = np.ravel(load_mat_variable(path, var_name))
    except KeyError:
        raw = load_dates_fallback_from_dir_struct(path)
    out: list[datetime] = []
    for x in raw:
        if isinstance(x, datetime):
            out.append(x)
        elif np.issubdtype(np.asarray(x).dtype, np.number):
            out.append(matlab_datenum_to_datetime(float(x)))
        else:
            s = str(x)
            out.append(datetime.fromisoformat(s))
    return out


def load_dates_fallback_from_dir_struct(path: str | Path) -> np.ndarray:
    """Derive dates from MATLAB dir-style structs when a direct dates variable is absent."""
    vars_available = mat_variable_names(path)
    for candidate in ("Dir_filtered", "Dir"):
        if candidate not in vars_available:
            continue
        obj = load_mat_variable(path, candidate)
        strings = collect_strings_from_mat_object(obj, preferred_fields=("name", "folder"))
        parsed = parse_dates_from_strings(strings)
        if parsed:
            return np.array(parsed, dtype=object)
    available = ", ".join(vars_available) if vars_available else "<none>"
    raise KeyError(
        f"No '{path}' dates variable found and no dates could be derived from "
        f"Dir_filtered/Dir. Available variables: {available}"
    )


def collect_strings_from_mat_object(obj, preferred_fields: tuple[str, ...] = ()) -> list[str]:
    out: list[str] = []

    def add_value(value) -> None:
        if isinstance(value, bytes):
            out.append(value.decode(errors="ignore"))
        elif isinstance(value, str):
            out.append(value)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S"}:
                if value.ndim == 0:
                    out.append(str(value.item()))
                else:
                    flat = value.ravel()
                    if flat.size and all(str(x).size == 1 for x in flat[: min(flat.size, 32)]):
                        out.append("".join(str(x) for x in flat))
                    else:
                        out.extend(str(x) for x in flat)
            elif value.dtype.names:
                collect(value)
            elif value.dtype == object:
                for item in value.ravel():
                    collect(item)
        elif hasattr(value, "_fieldnames"):
            collect(value)

    def collect(value) -> None:
        if isinstance(value, np.ndarray) and value.dtype.names:
            for field in preferred_fields + tuple(n for n in value.dtype.names if n not in preferred_fields):
                if field in value.dtype.names:
                    add_value(value[field])
            return
        if hasattr(value, "_fieldnames"):
            for field in preferred_fields + tuple(n for n in value._fieldnames if n not in preferred_fields):
                if hasattr(value, field):
                    add_value(getattr(value, field))
            return
        add_value(value)

    collect(obj)
    return out


def parse_dates_from_strings(strings: Iterable[str]) -> list[datetime]:
    found: list[datetime] = []
    seen: set[datetime] = set()
    for s in strings:
        for token in re.findall(r"(?<!\d)(19\d{6}|20\d{6})(?!\d)", str(s)):
            try:
                dt = datetime.strptime(token, "%Y%m%d")
            except ValueError:
                continue
            if dt not in seen:
                seen.add(dt)
                found.append(dt)
    return found


def yyyymmdd(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def yyyymmdd_to_datetime(values: Iterable[int | float]) -> list[datetime]:
    return [datetime.strptime(str(int(v)), "%Y%m%d") for v in values]


def month(dt: datetime) -> int:
    return dt.month


def year(dt: datetime) -> int:
    return dt.year


def saturation_vapor_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def compute_heat_index_c(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NOAA/Steadman heat index in C, matching the MATLAB formula."""
    temp_f = (9.0 / 5.0) * temp_c + 32.0
    heat_index = np.full(temp_f.shape, np.nan, dtype=np.float64)

    mask_simple = temp_f < 80.0
    heat_index[mask_simple] = 0.5 * (
        temp_f[mask_simple]
        + 61.0
        + ((temp_f[mask_simple] - 68.0) * 1.2)
        + (rh[mask_simple] * 0.094)
    )

    adj1 = np.zeros(temp_f.shape, dtype=np.float64)
    adj2 = np.zeros(temp_f.shape, dtype=np.float64)

    mask_adj1 = (rh < 13.0) & (temp_f >= 80.0) & (temp_f <= 112.0)
    adj1[mask_adj1] = -(
        ((13.0 - rh[mask_adj1]) / 4.0)
        * np.sqrt((17.0 - np.abs(temp_f[mask_adj1] - 95.0)) / 17.0)
    )

    mask_adj2 = (rh > 85.0) & (temp_f >= 80.0) & (temp_f <= 87.0)
    adj2[mask_adj2] = ((rh[mask_adj2] - 85.0) / 10.0) * (
        (87.0 - temp_f[mask_adj2]) / 5.0
    )

    mask_hi = temp_f >= 80.0
    tf = temp_f[mask_hi]
    r = rh[mask_hi]
    hi = (
        -42.379
        + (2.04901523 * tf)
        + (10.14333127 * r)
        - (0.22475541 * tf * r)
        - (0.00683783 * tf**2)
        - (0.05481717 * r**2)
        + (0.00122874 * tf**2 * r)
        + (0.00085282 * tf * r**2)
        - (0.00000199 * tf**2 * r**2)
    )
    heat_index[mask_hi] = hi + adj1[mask_hi] + adj2[mask_hi]
    return (5.0 / 9.0) * (heat_index - 32.0)


def matlab_prctile_nan_last_axis(values: np.ndarray, pct: float) -> np.ndarray:
    """MATLAB prctile(..., pct, dim) equivalent for the last axis with NaNs omitted.

    MATLAB uses plotting positions 100*((1:n)-0.5)/n and linearly interpolates
    between sorted observations, with end values clamped.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return arr
    n = arr.shape[-1]
    flat = arr.reshape(-1, n)
    counts = np.sum(np.isfinite(flat), axis=1)
    sorted_vals = np.sort(flat, axis=1)

    out = np.full(flat.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    if not np.any(valid):
        return out.reshape(arr.shape[:-1])

    rows = np.nonzero(valid)[0]
    c = counts[valid].astype(np.float64)
    r = (pct / 100.0) * c + 0.5
    r = np.clip(r, 1.0, c)
    lo = np.floor(r).astype(np.int64)
    hi = np.ceil(r).astype(np.int64)
    frac = r - lo

    v_lo = sorted_vals[rows, lo - 1]
    v_hi = sorted_vals[rows, hi - 1]
    out[rows] = v_lo + frac * (v_hi - v_lo)
    return out.reshape(arr.shape[:-1])


def save_daily_h5(path: str | Path, hi: np.ndarray, t2: np.ndarray, tvar: str) -> None:
    from scipy.io import savemat

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    savemat(
        tmp,
        {
            "HI": np.asarray(hi, dtype=np.float32),
            "T2": np.asarray(t2, dtype=np.float32),
            "tvar": np.array(str(tvar)),
        },
        do_compression=False,
    )
    tmp.replace(path)


def daily_h5_reusable(path: str | Path, tvar: str) -> bool:
    from scipy.io import loadmat

    path = Path(path)
    if not path.is_file():
        return False
    try:
        data = loadmat(path, variable_names=["HI", "T2", "tvar"], squeeze_me=True)
    except (NotImplementedError, OSError, ValueError):
        return False
    if "HI" not in data or "T2" not in data or "tvar" not in data:
        return False
    return str(np.asarray(data["tvar"]).item()) == str(tvar)


def load_daily_array(path: str | Path, var_name: str, row_slice: slice | None = None) -> np.ndarray:
    from scipy.io import loadmat

    try:
        arr = loadmat(path, variable_names=[var_name], squeeze_me=True)[var_name]
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{path} appears to be a MATLAB v7.3/HDF5 file, but h5py is not installed. "
            "Delete this cache file so it can be rebuilt as a standard MAT file, or install h5py."
        ) from exc
    if row_slice is not None:
        arr = arr[row_slice, :]
    return np.asarray(arr, dtype=np.float32)


def retry(action: Callable[[], None], label: str, max_try: int = 5) -> None:
    for attempt in range(1, max_try + 1):
        try:
            action()
            return
        except Exception:
            if attempt == max_try:
                raise
            print(f"  retry {attempt}/{max_try} for {label}")
            time.sleep(2 * attempt)


def masked_to_nan(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if np.ma.isMaskedArray(arr):
        out = arr.filled(np.nan).astype(np.float64)
    return out


def matlab_round_positive(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def interp_grid_vector(vec: np.ndarray, intrinsic_1based: np.ndarray) -> np.ndarray:
    idx = np.arange(1, len(vec) + 1, dtype=np.float64)
    return np.interp(intrinsic_1based, idx, vec)


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return radius * c


@dataclass
class ZipGridMask:
    r_start: int
    r_count: int
    c_start: int
    c_count: int
    mask: np.ndarray
    n_cells: int
    zip_lat: float
    zip_lon: float
