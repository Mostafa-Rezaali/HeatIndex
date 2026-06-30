#!/bin/bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
AUTO_GIT_PULL=${AUTO_GIT_PULL:-0}

cd "$CODE_DIR"
if [ "$AUTO_GIT_PULL" = "1" ]; then
  module load git || true
  git pull --ff-only origin master
fi

cd "$DATA_DIR"

jid=$(sbatch --parsable "$CODE_DIR/submit_heatindex_pipeline.slurm")
echo "Submitted HeatIndex full pipeline as one Slurm job: $jid"
