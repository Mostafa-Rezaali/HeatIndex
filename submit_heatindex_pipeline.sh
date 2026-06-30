#!/bin/bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
DATA_DIR=${DATA_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}

module load git || true
cd "$CODE_DIR"
git pull --ff-only origin master

cd "$DATA_DIR"

jid1=$(sbatch --parsable "$CODE_DIR/submit_build_prism_exceedance_mag.slurm")
echo "Submitted PRISM/HI magnitude build: $jid1"

jid2=$(sbatch --parsable --dependency=afterok:"$jid1" "$CODE_DIR/submit_detect_hi_heatwave_days.slurm")
echo "Submitted HI heatwave-day detection after $jid1: $jid2"

jid3=$(sbatch --parsable --dependency=afterok:"$jid2" "$CODE_DIR/submit_append_hsci_to_hospital_admittance.slurm")
echo "Submitted hospital-admittance extraction after $jid2: $jid3"

echo "HeatIndex pipeline submitted: $jid1 -> $jid2 -> $jid3"
