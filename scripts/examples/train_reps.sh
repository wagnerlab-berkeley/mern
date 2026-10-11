#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --output=train_reps_%A_%a.out
#SBATCH --error=train_reps_%A_%a.err
#SBATCH --array=0-9

# Activate your MeRN environment and edit the paths/species below.
# For X replicates, change the array to 0 through X-1 (e.g. 0-14 for 15).
# Submit: sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/examples/train_reps.sh
set -euo pipefail

DATA_DIR="/path/to/data"
ADATA_FILE="prepared.h5ad"
OUTPUT_DIR="/path/to/model_reps"
SPECIES="mouse"

# CLI defaults match the paper's model hyperparameters.
# Add --batch_key COLUMN or --counts_layer LAYER below when needed.
mern train reps \
  --rep "$SLURM_ARRAY_TASK_ID" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --adata_file "$ADATA_FILE" \
  --species "$SPECIES"
