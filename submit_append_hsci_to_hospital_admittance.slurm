#!/bin/bash
#SBATCH --job-name=hi_hospital
#SBATCH --output=hi_hospital_%j.log
#SBATCH --error=hi_hospital_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/blue/nessie/mostafarezaali/400M_PRISM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

CODE_DIR=${CODE_DIR:-/blue/nessie/mostafarezaali/400M_PRISM/HeatIndex_code}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
PY=${PY:-/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python}
ZIP_BUFFER_CELLS=${ZIP_BUFFER_CELLS:-1}
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
"$PY" -m py_compile "$CODE_DIR/heatindex/utils.py" "$CODE_DIR/scripts/append_hsci_to_hospital_admittance.py"

cd "$DATA_DIR"
"$PY" -u "$CODE_DIR/scripts/append_hsci_to_hospital_admittance.py" \
  --zip-buffer-cells "$ZIP_BUFFER_CELLS"

echo "HEATINDEX HOSPITAL ADMITTANCE HSCI APPEND COMPLETE"
