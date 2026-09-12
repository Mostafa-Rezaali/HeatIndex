import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if importlib.util.find_spec("netCDF4") is None:
    netcdf4_stub = types.ModuleType("netCDF4")
    netcdf4_stub.Dataset = None
    sys.modules["netCDF4"] = netcdf4_stub

for _name in ("rasterio", "requests"):
    if importlib.util.find_spec(_name) is None:
        _stub = types.ModuleType(_name)
        if _name == "rasterio":
            _stub.open = None
            _stub.transform = types.SimpleNamespace(xy=None)
        sys.modules[_name] = _stub

from heatindex.utils import ZipGridMask, daily_cache_reusable, load_daily_array
from scripts import append_hsci_to_hospital_admittance as hospital
from scripts import build_prism_exceedance_mag as builder


ADMIT = pd.Timestamp("2023-07-15")


def scripted_read_zip_avg(values_by_date):
    def fake(nc_file, var_name, date_index, zm, target_date, zero_if_date_missing):
        return values_by_date.get(pd.Timestamp(target_date).normalize(), np.nan)

    return fake


def day(offset):
    return ADMIT + pd.Timedelta(days=offset)


class CountZipHeatDaysTests(unittest.TestCase):
    def setUp(self):
        self.mask = ZipGridMask(0, 1, 0, 1, np.ones((1, 1), dtype=bool), 1, 0.0, 0.0)
        # Local exceedance is positive on offsets -1..-4; -5 is a non-exceedance
        # day; everything else is missing.
        self.local = {day(o): 1.5 for o in (-1, -2, -3, -4)}
        self.local[day(-5)] = np.nan
        # Domain HW days cover only offsets -1 and -2 (plus an HW day with no
        # local exceedance at -10, which must not count).
        self.ctx = {
            "mag_nc": "fake.nc",
            "idx_daily": {},
            "hsci_by_date": {day(-1): 3.0, day(-2): 3.0, day(-10): 3.0},
        }

    def test_heatwave_days_require_domain_and_local(self):
        with patch.object(hospital, "read_zip_avg", scripted_read_zip_avg(self.local)):
            self.assertEqual(hospital.count_zip_heat_days(self.ctx, self.mask, ADMIT, 30), 2)

    def test_excd_days_are_local_only(self):
        with patch.object(hospital, "read_zip_avg", scripted_read_zip_avg(self.local)):
            self.assertEqual(hospital.count_zip_excd_days(self.ctx, self.mask, ADMIT, 30), 4)

    def test_window_limits_apply(self):
        with patch.object(hospital, "read_zip_avg", scripted_read_zip_avg(self.local)):
            self.assertEqual(hospital.count_zip_excd_days(self.ctx, self.mask, ADMIT, 3), 3)
            self.assertEqual(hospital.count_zip_heat_days(self.ctx, self.mask, ADMIT, 1), 1)


class AnchoredDurationTests(unittest.TestCase):
    def setUp(self):
        self.mask = ZipGridMask(0, 1, 0, 1, np.ones((1, 1), dtype=bool), 1, 0.0, 0.0)
        self.ctx = {"mag_nc": "fake.nc", "idx_daily": {}, "hsci_by_date": {}}

    def run_case(self, values_by_date):
        with patch.object(hospital, "read_zip_avg", scripted_read_zip_avg(values_by_date)):
            return hospital.anchored_heat_duration_with_grace(self.ctx, self.mask, ADMIT)

    def test_admit_day_value_never_counts(self):
        vals = {day(0): 99.0, day(-1): 1.0, day(-2): 1.0, day(-3): 1.0}
        self.assertEqual(self.run_case(vals), 3)
        # Removing the admit-day value must not change the result.
        del vals[day(0)]
        self.assertEqual(self.run_case(vals), 3)

    def test_single_gap_is_bridged_but_not_counted(self):
        vals = {day(-1): 1.0, day(-3): 1.0, day(-4): 1.0}
        self.assertEqual(self.run_case(vals), 3)

    def test_two_consecutive_gaps_terminate(self):
        vals = {day(-1): 1.0, day(-4): 1.0, day(-5): 1.0}
        self.assertEqual(self.run_case(vals), 1)

    def test_streak_starting_only_before_admit_minus_one_uses_grace(self):
        # No heat on admit-1: one grace day is spent immediately, so a streak
        # at admit-2 onward is still reached.
        vals = {day(-2): 1.0, day(-3): 1.0}
        self.assertEqual(self.run_case(vals), 2)


class ConvertLegacyMatTests(unittest.TestCase):
    def setUp(self):
        self.hi = np.arange(12, dtype=np.float32).reshape(3, 4) / 2.0
        self.t2 = np.arange(12, dtype=np.float32).reshape(3, 4) + 20.0

    def write_mat(self, path, tvar="tmax"):
        import h5py

        with h5py.File(path, "w") as h5:
            # MATLAB v7.3 stores arrays column-major, so [3 x 4] appears
            # transposed to h5py; char arrays are stored as uint16 codes.
            h5.create_dataset("HI", data=self.hi.T)
            h5.create_dataset("T2", data=self.t2.T)
            h5.create_dataset("tvar", data=np.array([ord(c) for c in tvar], dtype=np.uint16))

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mat = Path(tmp) / "HI_20230715.mat"
            npz = Path(tmp) / "HI_20230715.npz"
            self.write_mat(mat)
            self.assertTrue(builder.convert_legacy_mat(mat, npz, "tmax"))
            self.assertTrue(mat.is_file())  # never deleted
            self.assertTrue(daily_cache_reusable(npz, "tmax"))
            hi = load_daily_array(npz, "HI")
            t2 = load_daily_array(npz, "T2")
            self.assertEqual(hi.shape, (3, 4))
            self.assertEqual(hi.dtype, np.float32)
            np.testing.assert_array_equal(hi, self.hi)
            np.testing.assert_array_equal(t2, self.t2)

    def test_tvar_mismatch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mat = Path(tmp) / "HI_20230715.mat"
            npz = Path(tmp) / "HI_20230715.npz"
            self.write_mat(mat, tvar="tmean")
            self.assertFalse(builder.convert_legacy_mat(mat, npz, "tmax"))
            self.assertFalse(npz.exists())

    def test_missing_variable_writes_nothing(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            mat = Path(tmp) / "HI_20230715.mat"
            npz = Path(tmp) / "HI_20230715.npz"
            with h5py.File(mat, "w") as h5:
                h5.create_dataset("HI", data=self.hi.T)
            self.assertFalse(builder.convert_legacy_mat(mat, npz, "tmax"))
            self.assertFalse(npz.exists())


if __name__ == "__main__":
    unittest.main()
