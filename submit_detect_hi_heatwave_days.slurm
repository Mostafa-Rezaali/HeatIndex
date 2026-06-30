#!/bin/bash
#SBATCH --job-name=hi_detect_hw
#SBATCH --output=hi_detect_hw_%j.log
#SBATCH --error=hi_detect_hw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=${WORK_DIR:-/blue/nessie/mostafarezaali/HeatIndex}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}
PCT=${PCT:-90}
DETECT_WORKERS=${DETECT_WORKERS:-1}
AREA_MAT=${AREA_MAT:-/blue/nessie/mostafarezaali/Research_Project/DayMet_FL/CR_Areas.mat}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
module load git || true
git pull --ff-only origin master

"$PY" -m py_compile heatindex/utils.py scripts/detect_hi_heatwave_days.py

"$PY" -u scripts/detect_hi_heatwave_days.py \
  --pct "$PCT" \
  --area-mat "$AREA_MAT" \
  --workers "$DETECT_WORKERS"

echo "HEATINDEX HI HEATWAVE DAY DETECTION COMPLETE"
