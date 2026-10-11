#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=average_reaction_activity_%j.out
#SBATCH --error=average_reaction_activity_%j.err

# Activate your MeRN environment and edit the paths/species below.
# Submit: sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION scripts/examples/average_reaction_activity.sh
set -euo pipefail

ADATA_PATH="/path/to/data/prepared.h5ad"
MODEL_DIR="/path/to/model_reps"
SPECIES="mouse"

# Run after training completes. MODEL_DIR contains mern_rep_* models.
# For a KL search, point MODEL_DIR to the chosen KL-condition directory.
# Averages 8 independent decodes/model, then averages across model replicates.
# Saves MODEL_DIR/average_enzyme_activity.csv.gz.
mern average-enzyme-activity \
  --dir "$MODEL_DIR" \
  --adata_path "$ADATA_PATH" \
  --species "$SPECIES"
