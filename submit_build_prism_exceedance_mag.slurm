#!/bin/bash
#SBATCH --job-name=hi_build_mag
#SBATCH --output=hi_build_mag_%j.log
#SBATCH --error=hi_build_mag_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/blue/nessie/mostafarezaali/400M_PRISM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

CODE_DIR=${CODE_DIR:-/blue/nessie/mostafarezaali/HeatIndex_code}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}
PCT=${PCT:-90}
T2M_VAR=${T2M_VAR:-tmax}
WORKERS=${WORKERS:-16}
BLOCK_ROWS=${BLOCK_ROWS:-64}
AUTO_GIT_PULL=${AUTO_GIT_PULL:-0}
DATES_MAT=${DATES_MAT:-Dir_MJJAS_HI.mat}
DATES_VAR=${DATES_VAR:-dates}

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
"$PY" -m py_compile "$CODE_DIR/heatindex/utils.py" "$CODE_DIR/scripts/build_prism_exceedance_mag.py"

cd "$DATA_DIR"
"$PY" -u "$CODE_DIR/scripts/build_prism_exceedance_mag.py" \
  --pct "$PCT" \
  --t2m-var "$T2M_VAR" \
  --dates-mat "$DATES_MAT" \
  --dates-var "$DATES_VAR" \
  --workers "$WORKERS" \
  --block-rows "$BLOCK_ROWS"

echo "HEATINDEX PRISM EXCEEDANCE MAG BUILD COMPLETE"
