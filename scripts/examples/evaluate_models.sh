#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=evaluate_models_%j.out
#SBATCH --error=evaluate_models_%j.err

# Activate your MeRN environment and edit the paths/species below.
# Submit: sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION scripts/examples/evaluate_models.sh
set -euo pipefail

ADATA_PATH="/path/to/data/prepared.h5ad"
MODEL_DIR="/path/to/kl_search"
SPECIES="mouse"

# Run after training completes. MODEL_DIR contains condition directories,
# each containing mern_rep_* models (as produced by kl_search.sh).
# Defaults: 8 decodes/model, 100 cell neighbors, 25 reaction neighbors.
# Saves MODEL_DIR/evaluation_results.pkl.
mern evaluate-models \
  --dir "$MODEL_DIR" \
  --adata_path "$ADATA_PATH" \
  --species "$SPECIES"
