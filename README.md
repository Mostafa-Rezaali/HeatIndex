# HeatIndex Python Translation

This repository contains a Python translation of the provided MATLAB HeatIndex
workflow:

1. Download PRISM daily data, compute heat index, monthly climatological
   thresholds, and daily exceedance-magnitude NetCDF files.
2. Detect HI heat-wave days and write the HSCI/EXCD NetCDF product.
3. Append HSCI and ZIP-level exposure metrics to the hospital-admission CSV.

The numerical formulas and decision rules were translated directly from the
MATLAB scripts. The main known dependency gap is that the original MATLAB helper
`detectHeatwavesByYear` was not included in the provided code. The Python
version implements the usual by-year consecutive-run rule with a default
minimum duration of 3 days; pass `--min-duration` if the original helper used a
different value.

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

This stage keeps the reusable per-day HI/T2 files in `_HI_tmp`, matching the
MATLAB workflow, and deletes each day's downloaded PRISM zip files and extracted
rasters after that day is processed.

Detect HI heat-wave days and HSCI:

```powershell
python scripts/detect_hi_heatwave_days.py --pct 90
```

Append exposure metrics to hospital-admission records:

```powershell
python scripts/append_hsci_to_hospital_admittance.py
```

All scripts default to the same filenames used in the MATLAB code and can be
customized with `--help`.

## HiPerGator Slurm

On HiPerGator, keep the Git checkout separate from the main PRISM data
directory. All inputs, intermediate files, logs, and outputs are read from or
written to the data directory by default:

```bash
DATA_DIR=/blue/nessie/mostafarezaali/400M_PRISM
CODE_DIR=/blue/nessie/mostafarezaali/HeatIndex_code
```

Create or refresh the code checkout:

```bash
cd /blue/nessie/mostafarezaali
git clone https://github.com/Mostafa-Rezaali/HeatIndex.git HeatIndex_code
cd "$CODE_DIR"
git pull --ff-only origin master
```

The repository includes HiPerGator submission scripts with `--mem=500G`. To
submit the full pipeline with dependencies:

```bash
cd "$CODE_DIR"
bash submit_heatindex_pipeline.sh
```

Defaults can be overridden at submission time, for example:

```bash
cd "$DATA_DIR"
PCT=90 T2M_VAR=tmax WORKERS=16 sbatch "$CODE_DIR/submit_build_prism_exceedance_mag.slurm"
```
