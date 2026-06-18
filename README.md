# Metabolic Representation Net (MeRN)

`mern` is a Python package for graph-guided metabolic representation learning from single-cell transcriptomic data. It builds on `scvi-tools` and `anndata`, providing APIs for learning interpretable metabolic latent structure, working with metabolic reaction graphs, and analyzing metabolic programs from single-cell RNA-seq data.

## MeRN overview

Single-cell RNA-seq data provides rich information about gene expression across thousands of cells, but interpreting cellular metabolism from transcriptomic data alone is challenging. Many metabolic processes are coordinated across genes and reactions, making them difficult to capture with standard dimensionality reduction methods.

Metabolic Representation Net (MeRN) is a graph-guided variational autoencoder designed to infer metabolic state from single-cell transcriptomic data.

Standard VAEs such as scVI learn low-dimensional latent representations of cells and decode them to gene-level parameters using flexible neural networks. While these models capture complex transcriptional structure, the latent dimensions are not explicitly linked to metabolic processes.

MeRN introduces a metabolic prior by leveraging a metabolic reaction graph, where:

- nodes represent metabolic reactions
- edges connect reactions that share metabolites

A graph VAE learns embeddings of this reaction network, and the cell-level latent space is decomposed into two components:

- Metabolic variation: variation explained by metabolic reactions
- Background variation: non-metabolic transcriptional variation

This structure links latent dimensions to reaction activity and enzyme-coding genes, enabling interpretable metabolic representations.

## Documentation

Documentation sources live under `docs/source`. The main Sphinx entry point is
`docs/source/index.rst`, and the tutorial notebook currently lives in
`docs/source/tutorials/MeRN_tutorial_final.ipynb`. Built documentation is
written to `docs/build/html`.

Install the package in editable mode:

```bash
pip install -e .
```

Build the tutorial site from the `docs/` directory with:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in a browser.

## Installation

We recommend running `mern` on a recent Linux or macOS system with Python >= 3.10.

Install the package in editable mode with:

```bash
pip install -e .
```

This installs the model, support utilities, plotting tools, KEGG helpers, test
tools, and documentation build tools.

If installation fails while building `pygraphviz`, install Graphviz first
(`brew install graphviz` on macOS, or `conda install -c conda-forge graphviz pygraphviz`
inside a conda environment), then rerun the editable install.

## Training representations from the command line

Installing the package exposes a `mern` command. The main training entry point is:

```bash
mern train reps \
  --output_dir /path/to/output \
  --data_dir /path/to/data \
  --adata_file mouse_intestine_pp.h5ad.gz
```

This command trains one MERN replicate, saves the model under
`<output_dir>/mern_rep_<rep>/`, and writes latent representations, graph
embeddings, train/validation/test indices, clustering, and UMAP coordinates to:

```text
<output_dir>/mern_rep_<rep>/embeddings.pkl
```

The defaults match the 25-dimensional KL-search runs used by
`generate_mouse_intestine_kl_search.sh` and `generate_wt_main_kl_search.sh`:

```text
rep = 0
n_metabolic_dim = 25
n_background_dim = 15
species = mouse
strict_met_back_separation = true
separate_hvgs = true
rxn_genes_bias = true
max_kl_weight = 0.001
graph_kl_weight = 0.1
n_steps_kl_warmup = 0
```

So the mouse intestine run can be launched with only paths and the input file:

```bash
mern train reps \
  --output_dir data/intestine/model_reps/mouse_intestine_kl_search_25/max_kl_0.001_graph_kl_0.1 \
  --data_dir data/intestine \
  --adata_file mouse_intestine_pp.h5ad.gz
```

The WT folate main run uses the same training defaults with a different input:

```bash
mern train reps \
  --output_dir data/folate/folate_07_11_2025/model_reps/wt_main_kl_search_25/max_kl_0.001_graph_kl_0.1 \
  --data_dir data/folate/folate_07_11_2025 \
  --adata_file main_wt_anndata_pp.h5ad.gz
```

All defaults can be overridden. For example:

```bash
mern train reps \
  --output_dir /path/to/output \
  --data_dir /path/to/data \
  --adata_file mouse_intestine_pp.h5ad.gz \
  --rep 3 \
  --max_kl_weight 0.01 \
  --graph_kl_weight 0.01 \
  --n_background_dim 25
```

Boolean defaults can also be turned off:

```bash
mern train reps \
  --output_dir /path/to/output \
  --data_dir /path/to/data \
  --adata_file mouse_intestine_pp.h5ad.gz \
  --no_strict_met_back_separation \
  --no_separate_hvgs \
  --no_rxn_genes_bias
```

For local CPU smoke checks, override the accelerator:

```bash
mern train reps \
  --output_dir /tmp/mern_check \
  --data_dir tests/data \
  --adata_file mouse_intestine_100.h5ad \
  --max_epochs 1 \
  --accelerator cpu \
  --devices 1
```

### Slurm array wrapper

For larger sweeps, wrap `mern train reps` in a Slurm array script. This mirrors
the existing KL-search launchers: compute `rep`, `max_kl_weight`, and
`graph_kl_weight` from `SLURM_ARRAY_TASK_ID`, then pass those values to the CLI.

Mouse intestine 25-dimensional KL search:

```bash
#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=bash_output/train_mouse_intestine_kl_search_%A_%a.out
#SBATCH --error=bash_output/train_mouse_intestine_kl_search_%A_%a.err
#SBATCH --array=0-119

set -euo pipefail

n_metabolic_dim="${1:-25}"
n_background_dim="${2:-15}"

rep=$(( SLURM_ARRAY_TASK_ID % 15 ))
param_idx=$(( SLURM_ARRAY_TASK_ID / 15 ))
max_kl_idx=$(( param_idx % 4 ))
graph_kl_idx=$(( param_idx / 4 ))

max_kl_weights=(0.0001 0.001 0.005 0.01)
graph_kl_weights=(0.01 0.1)

max_kl_weight="${max_kl_weights[$max_kl_idx]}"
graph_kl_weight="${graph_kl_weights[$graph_kl_idx]}"

DATA_DIR="../../data/intestine"
output_stem="mouse_intestine_kl_search_${n_metabolic_dim}"
if [ "$n_background_dim" != "15" ]; then
    output_stem="${output_stem}_back_${n_background_dim}"
fi
OUTPUT_DIR="$DATA_DIR/model_reps/${output_stem}/max_kl_${max_kl_weight}_graph_kl_${graph_kl_weight}"

mkdir -p "$OUTPUT_DIR"

mern train reps \
  --rep "$rep" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --adata_file mouse_intestine_pp.h5ad.gz \
  --n_metabolic_dim "$n_metabolic_dim" \
  --n_background_dim "$n_background_dim" \
  --max_kl_weight "$max_kl_weight" \
  --graph_kl_weight "$graph_kl_weight"
```

For the WT main KL search, use the same wrapper pattern with
`#SBATCH --array=0-79`, `rep=$(( SLURM_ARRAY_TASK_ID % 10 ))`,
`param_idx=$(( SLURM_ARRAY_TASK_ID / 10 ))`, and:

```bash
max_kl_weights=(0.00001 0.0001 0.001 0.01)
graph_kl_weights=(0.01 0.1)
DATA_DIR="../../data/folate/folate_07_11_2025"
OUTPUT_DIR="$DATA_DIR/model_reps/wt_main_kl_search_${n_metabolic_dim}/max_kl_${max_kl_weight}_graph_kl_${graph_kl_weight}"

mern train reps \
  --rep "$rep" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --adata_file main_wt_anndata_pp.h5ad.gz \
  --n_metabolic_dim "$n_metabolic_dim" \
  --n_background_dim "$n_background_dim" \
  --max_kl_weight "$max_kl_weight" \
  --graph_kl_weight "$graph_kl_weight"
```

Because the command defaults already include `species=mouse`,
`strict_met_back_separation`, `separate_hvgs`, `rxn_genes_bias`, and
`n_steps_kl_warmup=0`, those flags do not need to be repeated in the Slurm
wrapper unless you want to make the script fully explicit.

## Key capabilities

- Learn graph-guided latent representations of cell state from single-cell RNA-seq data.
- Separate metabolic variation from background, non-metabolic transcriptional variation.
- Use reaction graphs and reaction-to-gene mappings to tie latent structure to metabolic biology.
- Work with KEGG-derived metabolic datasets and packaged metabolic network resources.
- Run downstream analysis and visualization utilities for pathway activity, latent factors, and topology-aware plots.

## Package layout

The current repository contains:

- `src/model/`: model code, including the main MeRN model, neural modules,
  base components, dataloaders, optimizer utilities, and configuration/constants
- `src/support/`: analysis, plotting, dataset helpers, and other support code
- `src/support/data/kegg/`: packaged KEGG-derived graph resources and related
  data files
- `docs/source/`: documentation sources for Sphinx
- `docs/source/tutorials/`: tutorial notebooks included in the docs
- `docs/build/html/`: generated HTML documentation after a docs build

## Contributing

Contributions are welcome. The next practical steps for this repository are:

- validating package installation
- resolving remaining import and dependency issues
- running and expanding tests
- improving the user-facing API and documentation

## Citation

Citation information will be added here later.
