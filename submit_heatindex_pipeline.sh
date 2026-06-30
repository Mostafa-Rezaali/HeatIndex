#!/bin/bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-/blue/nessie/mostafarezaali/400M_PRISM}
cd "$WORK_DIR"

jid1=$(sbatch --parsable submit_build_prism_exceedance_mag.slurm)
echo "Submitted PRISM/HI magnitude build: $jid1"

jid2=$(sbatch --parsable --dependency=afterok:"$jid1" submit_detect_hi_heatwave_days.slurm)
echo "Submitted HI heatwave-day detection after $jid1: $jid2"

jid3=$(sbatch --parsable --dependency=afterok:"$jid2" submit_append_hsci_to_hospital_admittance.slurm)
echo "Submitted hospital-admittance extraction after $jid2: $jid3"

echo "HeatIndex pipeline submitted: $jid1 -> $jid2 -> $jid3"
