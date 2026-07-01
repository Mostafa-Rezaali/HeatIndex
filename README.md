# HeatIndex Python Translation

This repository contains a Python translation of the provided MATLAB HeatIndex
workflow:

1. Download PRISM daily data, compute heat index, monthly climatological
   thresholds, and daily exceedance-magnitude NetCDF files.
2. Detect HI heat-wave days and write the HSCI/EXCD NetCDF product.
3. Append HSCI and ZIP-level exposure metrics to the hospital-admission CSV.

The numerical formulas and decision rules were translated directly from the
MATLAB scripts. The Python HSCI-H detector uses a default minimum duration of
3 valid HSCI-H days after the spatial/min-area heatwave rules and allows a
1-day post-rule non-HSCI grace gap to bridge an event; pass `--min-duration`
or `--grace-days` to override those values.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Scripts

Build daily HI/T exceedance magnitude products:

```powershell
python scripts/build_prism_exceedance_mag.py --pct 90 --t2m-var tmax
```

The MATLAB source expects `Dir_MJJAS_HI.mat` to contain a variable named
`dates`. If the file uses a different variable name, pass `--dates-var NAME`
or set `DATES_VAR=NAME` in the Slurm submission.

To inspect the variables in a MAT file:

```bash
python scripts/inspect_mat_variables.py /blue/nessie/mostafarezaali/400M_PRISM/Dir_MJJAS_HI.mat
```

This stage uses temporary Python cache files under `_HI_tmp` while running,
deletes each day's downloaded PRISM zip files and extracted rasters after that
day is processed, and removes the Python cache after successful NetCDF output
creation unless `--keep-python-cache` is set.

Detect HI heat-wave days and HSCI:

```powershell
python scripts/detect_hi_heatwave_days.py --pct 90
```

The default HSCI-H heatwave persistence rule is `--min-duration 3
--grace-days 1`. The grace day means a no-HSCI day after spatial/min-area
heatwave filtering. It can bridge an event, but it is not written as an HSCI
day.

Plot annual accumulated HSCI, i.e. AHSCI summed by year from the final HSCI
NetCDF:

```powershell
python scripts/plot_monthly_hsci.py --hsci-nc HI_EXCD_MJJAS_HWdays_90.nc --group-by year
```

Append exposure metrics to hospital-admission records:

```powershell
python scripts/append_hsci_to_hospital_admittance.py
```

The hospital output includes percentile-specific HI additions for each value in
`--hi-pcts` (default `90,95`):

- `HSCI_HI_30d_prior_p90`, `HSCI_HI_30d_prior_p95`
- `event_duration_HI_admit_anchor_p90`, `event_duration_HI_admit_anchor_p95`
- `days_heatwave_HI_30d_prior_p90`, `days_heatwave_HI_30d_prior_p95`
- `days_heatwave_HI_21d_prior_p90`, `days_heatwave_HI_21d_prior_p95`
- `days_heatwave_HI_14d_prior_p90`, `days_heatwave_HI_14d_prior_p95`

The anchored event duration counts ZIP-level HI exceedance days in the run
ending on the admit day, scanning backward up to 30 days and allowing one
non-exceedance grace day without counting the grace day as a heat day.

All scripts default to the same filenames used in the MATLAB code and can be
customized with `--help`.

## HiPerGator Slurm

On HiPerGator, keep the Git checkout separate from the main PRISM data
directory. All inputs, intermediate files, logs, and outputs are read from or
written to the data directory by default:

```bash
DATA_DIR=/blue/nessie/mostafarezaali/400M_PRISM
CODE_DIR=/blue/nessie/mostafarezaali/400M_PRISM/HeatIndex_code
```

Create or refresh the code checkout:

```bash
cd "$DATA_DIR"
git clone https://github.com/Mostafa-Rezaali/HeatIndex.git HeatIndex_code
cd "$CODE_DIR"
git pull --ff-only origin master
```

GitHub HTTPS requires a personal access token or cached credentials. The Slurm
scripts do not run `git pull` by default, so batch jobs use the code already in
`CODE_DIR`. Set `AUTO_GIT_PULL=1` only after Git credentials are configured.

The repository includes a HiPerGator full-pipeline submission script with
`--mem=500G`. It runs all stages sequentially in one Slurm job allocation:

```bash
cd "$CODE_DIR"
bash submit_heatindex_pipeline.sh
```

Defaults can be overridden at submission time, for example:

```bash
cd "$DATA_DIR"
PCTS=90,95 T2M_VAR=tmax WORKERS=64 THRESHOLD_WORKERS=16 SLICE_WORKERS=64 DETECT_WORKERS=64 DETECT_WRITE_WORKERS=64 APPEND_WORKERS=64 sbatch "$CODE_DIR/submit_heatindex_pipeline.slurm"
```
