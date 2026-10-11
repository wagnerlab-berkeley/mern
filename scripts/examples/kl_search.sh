#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=kl_search_%A_%a.out
#SBATCH --error=kl_search_%A_%a.err
#SBATCH --array=0-119

# Activate your MeRN environment and edit the paths/species below.
# For X replicates per combination, set N_REPS=X and the array to 0 through 8*X-1.
# Submit: sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/examples/kl_search.sh
set -euo pipefail

DATA_DIR="/path/to/data"
ADATA_FILE="prepared.h5ad"
OUTPUT_DIR="/path/to/kl_search"
SPECIES="mouse"
N_REPS=15

# Mouse-intestine KL grid from the paper, at its selected latent dimensions (25/15).
# This is the KL-only sweep; the 840-model search also varied latent dimensions.
max_kl_weights=(0.0001 0.001 0.005 0.01)
graph_kl_weights=(0.01 0.1)

rep=$((SLURM_ARRAY_TASK_ID % N_REPS))
param_idx=$((SLURM_ARRAY_TASK_ID / N_REPS))
max_kl_weight=${max_kl_weights[$((param_idx % 4))]}
graph_kl_weight=${graph_kl_weights[$((param_idx / 4))]}

mern train reps \
  --rep "$rep" \
  --output_dir "$OUTPUT_DIR/max_kl_${max_kl_weight}_graph_kl_${graph_kl_weight}" \
  --data_dir "$DATA_DIR" \
  --adata_file "$ADATA_FILE" \
  --species "$SPECIES" \
  --max_kl_weight "$max_kl_weight" \
  --graph_kl_weight "$graph_kl_weight"
