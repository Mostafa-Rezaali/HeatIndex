import unittest
from unittest.mock import patch
import importlib.util
import sys
import types

import numpy as np
import pandas as pd

if importlib.util.find_spec("netCDF4") is None:
    netcdf4_stub = types.ModuleType("netCDF4")
    netcdf4_stub.Dataset = None
    sys.modules["netCDF4"] = netcdf4_stub

from heatindex.utils import ZipGridMask
from scripts import append_hsci_to_hospital_admittance as hospital


class PercentileOutputTests(unittest.TestCase):
    def test_default_paths_are_percentile_specific(self):
        for pct in (80, 90, 95):
            with self.subTest(pct=pct):
                paths = hospital.resolve_percentile_paths(pct)
                self.assertEqual(paths["hw_nc_t"], f"EXCD_MJJAS_HWdays_{pct}.nc")
                self.assertEqual(paths["hw_nc_hi"], f"HI_EXCD_MJJAS_HWdays_{pct}.nc")
                self.assertEqual(paths["mag_nc_hi"], f"HI_EXCDMAG_daily_1981_2025_{pct}.nc")
                self.assertEqual(paths["out_csv"], f"Hospital_Admittance_with_HSCI_{pct}.csv")

    def test_invalid_percentile_is_rejected(self):
        for pct in (0, 100):
            with self.subTest(pct=pct), self.assertRaises(ValueError):
                hospital.resolve_percentile_paths(pct)

    def test_patient_metrics_use_only_the_selected_percentile_context(self):
        admit = pd.Timestamp("2023-07-10")
        mask = ZipGridMask(0, 1, 0, 1, np.ones((1, 1), dtype=bool), 1, 0.0, 0.0)
        prior_dates = [admit - pd.Timedelta(days=d) for d in range(1, 8)]
        context = {
            "args": {"hw_nc_t": "T_P95.nc"},
            "masks_hi": {"z32608": mask},
            "masks_t": {"z32608": mask},
            "hi_context": {
                "pct": 95,
                "mag_nc": "HI_P95.nc",
                "idx_daily": {d: i for i, d in enumerate([admit, *prior_dates])},
                "hsci_by_date": {d: float(i + 10) for i, d in enumerate(prior_dates)},
            },
            "hsci_t_by_date": {admit: 1.0, **{d: float(i + 1) for i, d in enumerate(prior_dates)}},
            "hsci_hi_by_date": {admit: 2.0, **{d: float(i + 10) for i, d in enumerate(prior_dates)}},
            "idx_hw_t": {d: i for i, d in enumerate([admit, *prior_dates])},
            "idx_daily_hi": {d: i for i, d in enumerate([admit, *prior_dates])},
            "dates_daily_hi": pd.DatetimeIndex([admit, *prior_dates]),
            "lat_grid": np.zeros((1, 1)),
            "lon_grid": np.zeros((1, 1)),
        }
        files_read = []

        def fake_read_zip_avg(nc_file, _var_name, _date_index, _mask, target_date, _zero_if_missing):
            files_read.append(nc_file)
            day_offset = int((admit - pd.Timestamp(target_date)).days)
            return float(100 - day_offset if str(nc_file).startswith("HI") else 10 - day_offset)

        with patch.object(hospital, "read_zip_avg", side_effect=fake_read_zip_avg):
            _, values = hospital.compute_patient_values(0, "32608", admit, context=context)

        self.assertEqual(values["HSCI_T_1d_prior"], 1.0)
        self.assertEqual(values["HSCI_HI_2d_prior"], 21.0)
        self.assertEqual(values["days_excd_T_6d_prior"], 6)
        self.assertEqual(values["days_excd_HI_6d_prior"], 6)
        self.assertEqual(values["max_excd_T_6d_prior"], 9.0)
        self.assertEqual(values["max_excd_HI_6d_prior"], 99.0)
        self.assertIn("HSCI_HI_30d_prior", values)
        self.assertFalse(any(key.endswith("_p95") for key in values))
        self.assertEqual(set(files_read), {"T_P95.nc", "HI_P95.nc"})


if __name__ == "__main__":
    unittest.main()
