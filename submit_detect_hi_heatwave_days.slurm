#!/bin/bash
#SBATCH --job-name=hi_detect_hw
#SBATCH --output=hi_detect_hw_%j.log
#SBATCH --error=hi_detect_hw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/blue/nessie/mostafarezaali/400M_PRISM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

CODE_DIR=${CODE_DIR:-/blue/nessie/mostafarezaali/400M_PRISM/HeatIndex_code}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}
PCT=${PCT:-90}
DETECT_WORKERS=${DETECT_WORKERS:-64}
DETECT_WRITE_WORKERS=${DETECT_WRITE_WORKERS:-64}
AREA_MAT=${AREA_MAT:-/blue/nessie/mostafarezaali/Research_Project/DayMet_FL/CR_Areas.mat}
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
"$PY" -m py_compile "$CODE_DIR/heatindex/utils.py" "$CODE_DIR/scripts/detect_hi_heatwave_days.py"

cd "$DATA_DIR"
"$PY" -u "$CODE_DIR/scripts/detect_hi_heatwave_days.py" \
  --pct "$PCT" \
  --area-mat "$AREA_MAT" \
  --workers "$DETECT_WORKERS" \
  --write-workers "$DETECT_WRITE_WORKERS"

echo "HEATINDEX HI HEATWAVE DAY DETECTION COMPLETE"
