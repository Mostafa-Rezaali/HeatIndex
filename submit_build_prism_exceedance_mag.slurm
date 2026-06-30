#!/bin/bash
#SBATCH --job-name=hi_build_mag
#SBATCH --output=hi_build_mag_%j.log
#SBATCH --error=hi_build_mag_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=${WORK_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}
PCT=${PCT:-90}
T2M_VAR=${T2M_VAR:-tmax}
WORKERS=${WORKERS:-16}
BLOCK_ROWS=${BLOCK_ROWS:-64}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
module load git || true
git pull --ff-only origin master

"$PY" -m py_compile heatindex/utils.py scripts/build_prism_exceedance_mag.py

"$PY" -u scripts/build_prism_exceedance_mag.py \
  --pct "$PCT" \
  --t2m-var "$T2M_VAR" \
  --workers "$WORKERS" \
  --block-rows "$BLOCK_ROWS"

echo "HEATINDEX PRISM EXCEEDANCE MAG BUILD COMPLETE"
