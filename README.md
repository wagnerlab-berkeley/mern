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
- Non-metabolic variation: transcriptomic variation explained by genes not encoding enzymes

This structure links latent dimensions to reaction activity and enzyme-coding genes, enabling interpretable metabolic representations.

## Documentation

Documentation sources live under `docs/source`. The main Sphinx entry point is
`docs/source/index.rst`, and the tutorial notebook currently lives in
`docs/source/tutorials/MeRN_tutorial.ipynb`. Built documentation is
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

## Quick Start

See the tutorial in the docs folder for a standard preprocessing, training, and analysis workflow. For larger datasets and cluster usage, the following cli can be used.

## Training representations from the command line

Installing the package exposes a `mern` command. The main training entry point is:

```bash
mern train reps \
  --output_dir /path/to/output \
  --data_dir /path/to/data \
  --adata_file mouse_intestine_pp.h5ad.gz
```

By default, training reads raw counts from `adata.layers["counts"]`. Use
`--counts_layer <layer_name>` (or `--counts-layer`) when the counts are stored
under a different layer name.

This command trains one MERN replicate, saves the model under
`<output_dir>/mern_rep_<rep>/`, and writes latent representations, graph
embeddings, train/validation/test indices, clustering, and UMAP coordinates to:

```text
<output_dir>/mern_rep_<rep>/embeddings.pkl
```
This enables training of many replicates. Example bash scripts are included below.

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

## Averaging reaction activity across replicates

After training several replicate models, average stochastic reaction activity
decodes with:

```bash
mern average-enzyme-activity \
  --dir /path/to/model_reps/reps \
  --adata_path /path/to/data/cancer_endothelium_pp.h5ad \
  --species human
```

By default this reads model directories under `--dir`, decodes each model 8
times with independent seeds, and writes:

```text
<dir>/average_enzyme_activity.csv.gz
```

This command prepares the AnnData with the packaged KEGG support utilities, loads each replicate model on
CPU by default, calls `mern.support.average_enzyme_activity`, and saves a
compressed CSV. Override the output path or sampling settings with:

```bash
mern average-enzyme-activity \
  --dir /path/to/model_reps/reps \
  --adata_path /path/to/data/cancer_endothelium_pp.h5ad \
  --output_path /path/to/average_enzyme_activity.csv.gz \
  --species human \
  --n_samples 16 \
  --seed 8
```

Use `--accelerator cuda` only if you specifically want model loading/decoding on
GPU. For Slurm, this can be run as a single follow-up job after replicate
training finishes:

```bash
srun mern average-enzyme-activity \
  --dir "$DATA_DIR/model_reps/reps" \
  --adata_path "$DATA_DIR/cancer_endothelium_pp.h5ad" \
  --species human
```

## Evaluating replicate consistency metrics

To evaluate replicate consistency within each condition directory:

```bash
mern evaluate-models \
  --dir /path/to/model_reps \
  --adata_path /path/to/data.h5ad \
  --species mouse
```

Each condition under `--dir` is expected to
contain replicate model directories. Stochastic reaction activity is averaged over
8 decode samples by default, and results are written to:

```text
<dir>/evaluation_results.pkl
```

To compare two replicate directories directly:

```bash
mern evaluate-cross-models \
  --dir1 /path/to/reps_a \
  --dir2 /path/to/reps_b \
  --adata_path /path/to/data.h5ad
```

This writes `vs_<dir2>.pkl` under
`--dir1`. Both commands keep the old defaults of `--neighbors 100`,
`--graph_neighbors 25`, `--seed 8`, and CPU model loading.

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
- Separate metabolic variation from non-metabolic transcriptional variation.
- Use reaction graphs and reaction-to-gene mappings to tie latent structure to metabolic biology.
- Work with KEGG-derived metabolic datasets and packaged metabolic network resources.
- Run downstream analysis and visualization utilities for pathway activity, latent factors, and topology-aware plots.

## Package layout

The current repository contains:

- `src/mern/`: model code, including the main MeRN model, neural modules,
  base components, dataloaders, optimizer utilities, and configuration/constants
- `src/mern/support/`: analysis, plotting, dataset helpers, and other support code
- `src/mern/support/data/kegg/`: packaged KEGG-derived graph resources and related
  data files
- `docs/source/`: documentation sources for Sphinx
- `docs/source/tutorials/`: tutorial notebooks included in the docs
- `docs/build/html/`: generated HTML documentation after a docs build

## Contributing

Contributions and bug reports are welcome through GitHub issues and pull requests.

## Acknowledgements

MeRN is built on [scvi-tools](https://github.com/scverse/scvi-tools) and uses its
model, training, and data-management infrastructure. Parts of MeRN's model and
data-splitting implementation were adapted from scvi-tools; its BSD-3-Clause
notice is included in `LICENSE`.

Gayoso A, Lopez R, Xing G, et al. A Python library for probabilistic analysis of
single-cell omics data. *Nature Biotechnology* 40, 163–166 (2022).
https://doi.org/10.1038/s41587-021-01206-w

Early development of MERN drew on the software architecture of
[scGLUE](https://github.com/gao-lab/GLUE). The current MERN implementation
has been substantially rewritten and implements a distinct model and workflow.
We gratefully acknowledge the scGLUE developers for their foundational work.

Cao, Z.-J. and Gao, G. Multi-omics single-cell data integration and regulatory
inference with graph-linked embedding. *Nature Biotechnology* 40, 1458–1466
(2022). https://doi.org/10.1038/s41587-022-01284-4

## Citation

### MeRN

Lewinsohn DP, Dias N, Chau A, Koike Y, Smith ZD, Ioannidis NM, Wagner A. Extracting interpretable single-cell metabolic states with graph-guided representation learning. bioRxiv. 2026. doi:[10.64898/2026.09.17.751504](https://doi.org/10.64898/2026.09.17.751504)

The `v1.0.0` and `v1.0.1` tags are the manuscript-referenced releases before and
after the documented self-loop and reaction-gene mapping fixes.

### Application of MeRN to mouse embryogenesis and source of tutorial data

Dias N, Lewinsohn DP, Colgan WN, Wang M, Kijima Y, Villagrana J, Hou TCJ, Gowri G, Chau A, Aktas T, Sumigray K, Weissman JS, Koblan LW, Wagner A, Smith ZD. Folate deficiency disrupts key metabolic transitions within the developing neural ectoderm. bioRxiv. 2026. doi:[10.64898/2026.09.18.752622](https://doi.org/10.64898/2026.09.18.752622)
