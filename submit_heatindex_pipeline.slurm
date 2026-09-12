#!/bin/bash
#SBATCH --job-name=hi_pipeline
#SBATCH --output=hi_pipeline_%j.log
#SBATCH --error=hi_pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/blue/nessie/mostafarezaali/400M_PRISM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

CODE_DIR=${CODE_DIR:-/blue/nessie/mostafarezaali/400M_PRISM/HeatIndex_code}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}

PCTS=${PCTS:-80,90,95}
T2M_VAR=${T2M_VAR:-tmax}
WORKERS=${WORKERS:-64}
THRESHOLD_WORKERS=${THRESHOLD_WORKERS:-16}
SLICE_WORKERS=${SLICE_WORKERS:-64}
BLOCK_ROWS=${BLOCK_ROWS:-64}
DATES_MAT=${DATES_MAT:-Dir_MJJAS_HI.mat}
DATES_VAR=${DATES_VAR:-dates}

DETECT_WORKERS=${DETECT_WORKERS:-64}
DETECT_WRITE_WORKERS=${DETECT_WRITE_WORKERS:-64}
MIN_DURATION=${MIN_DURATION:-3}
GRACE_DAYS=${GRACE_DAYS:-1}
AREA_MAT=${AREA_MAT:-/blue/nessie/mostafarezaali/Research_Project/DayMet_FL/CR_Areas.mat}

ZIP_BUFFER_CELLS=${ZIP_BUFFER_CELLS:-1}
APPEND_WORKERS=${APPEND_WORKERS:-64}
APPEND_CHUNK_SIZE=${APPEND_CHUNK_SIZE:-100}
AUTO_GIT_PULL=${AUTO_GIT_PULL:-0}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$CODE_DIR"
if [ "$AUTO_GIT_PULL" = "1" ]; then
  module load git || true
  git pull --ff-only origin master
fi

export PYTHONPATH="$CODE_DIR"
"$PY" -m py_compile \
  "$CODE_DIR/heatindex/utils.py" \
  "$CODE_DIR/scripts/build_prism_exceedance_mag.py" \
  "$CODE_DIR/scripts/detect_hi_heatwave_days.py" \
  "$CODE_DIR/scripts/append_hsci_to_hospital_admittance.py"

cd "$DATA_DIR"

echo "Stage 1/3: build PRISM HI/T exceedance magnitude"
IFS=',' read -r -a PCT_LIST <<< "$PCTS"
for PCT in "${PCT_LIST[@]}"; do
  PCT="${PCT//[[:space:]]/}"
  echo "  Building percentile P${PCT}"
  "$PY" -u "$CODE_DIR/scripts/build_prism_exceedance_mag.py" \
    --pct "$PCT" \
    --t2m-var "$T2M_VAR" \
    --dates-mat "$DATES_MAT" \
    --dates-var "$DATES_VAR" \
    --workers "$WORKERS" \
    --threshold-workers "$THRESHOLD_WORKERS" \
    --slice-workers "$SLICE_WORKERS" \
    --block-rows "$BLOCK_ROWS"
done

echo "Stage 2/3: detect HSCI and HSCI-H heatwave days"
for PCT in "${PCT_LIST[@]}"; do
  PCT="${PCT//[[:space:]]/}"
  echo "  Detecting HSCI percentile P${PCT}"
  "$PY" -u "$CODE_DIR/scripts/detect_hi_heatwave_days.py" \
    --measure t \
    --pct "$PCT" \
    --area-mat "$AREA_MAT" \
    --min-duration "$MIN_DURATION" \
    --grace-days "$GRACE_DAYS" \
    --workers "$DETECT_WORKERS" \
    --write-workers "$DETECT_WRITE_WORKERS"
  echo "  Detecting HSCI-H percentile P${PCT}"
  "$PY" -u "$CODE_DIR/scripts/detect_hi_heatwave_days.py" \
    --measure hi \
    --pct "$PCT" \
    --area-mat "$AREA_MAT" \
    --min-duration "$MIN_DURATION" \
    --grace-days "$GRACE_DAYS" \
    --workers "$DETECT_WORKERS" \
    --write-workers "$DETECT_WRITE_WORKERS"
done

echo "Stage 3/3: append percentile-specific HSCI exposure metrics to hospital admittance"
for PCT in "${PCT_LIST[@]}"; do
  PCT="${PCT//[[:space:]]/}"
  echo "  Appending hospital exposures for P${PCT}"
  "$PY" -u "$CODE_DIR/scripts/append_hsci_to_hospital_admittance.py" \
    --pct "$PCT" \
    --zip-buffer-cells "$ZIP_BUFFER_CELLS" \
    --workers "$APPEND_WORKERS" \
    --chunk-size "$APPEND_CHUNK_SIZE"
done

rm -rf "$DATA_DIR/_HI_tmp" "$DATA_DIR/PRISM_cache"

echo "HEATINDEX FULL PIPELINE COMPLETE"
