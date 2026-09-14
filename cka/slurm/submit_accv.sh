#!/usr/bin/env bash
# Submit the ACCV compute plan on the HPC head node: T1 array first, its evals
# after it, T2 queued behind T1, T2 evals after T2, T3 after T2 evals.
# Usage (on hpc-head2, from ~/medsam-vpt): bash cka/slurm/submit_accv.sh
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
# Researcher access (14 Sep 2026): account dmb, QOS research lifts the two-GPU
# student limit. Override with SBATCH_ACCOUNT / SBATCH_QOS if needed.
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dmb}"
export SBATCH_QOS="${SBATCH_QOS:-research}"

T1=$(sbatch --parsable cka/slurm/accv_t1.sbatch)
echo "T1 array:      $T1"
T1E=$(sbatch --parsable --dependency=afterany:$T1 cka/slurm/accv_t1_eval.sbatch)
echo "T1 eval:       $T1E (afterany:$T1)"
T2=$(sbatch --parsable --dependency=afterany:$T1 cka/slurm/accv_t2.sbatch)
echo "T2 array:      $T2 (afterany:$T1)"
T2E=$(sbatch --parsable --dependency=afterany:$T2 cka/slurm/accv_t2_eval.sbatch)
echo "T2 eval:       $T2E (afterany:$T2)"
T3=$(sbatch --parsable --dependency=afterany:$T2E cka/slurm/accv_t3.sbatch)
echo "T3 dumps:      $T3 (afterany:$T2E)"
echo
squeue -u "$USER" --array
