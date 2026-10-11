# Metabolic Representation Net (MeRN)

[Install](#installation) | [Tutorial](https://wagnerlab-berkeley.github.io/mern/) | [Paper](https://doi.org/10.64898/2026.09.17.751504) | [Citation](#citation)

MeRN infers single-cell metabolic states from transcriptomic data by jointly
modeling a graph representation of the metabolic network and enzyme-coding gene
expression with a graph-guided variational autoencoder.

MeRN provides two complementary descriptions of metabolic state:

1. **A metabolic latent space**, useful for clustering cells, pseudotime analysis,
   and visualization based on metabolic variability. The model also learns a
   non-metabolic latent space, enabling comparison of metabolic and non-metabolic
   variation.
2. **An inferred reaction activity matrix**, which captures the relative activity
   of every reaction across cells and enables comparison of reaction- or
   pathway-level metabolic activity across cell types, states, and experimental
   conditions.

**Metabolic data-driven pathways (DDPs).** Using a graph-constrained
agglomerative clustering procedure on inferred reaction activity, MeRN defines
connected modules supported by transcriptomic evidence, allowing you to learn
pathway definitions for the cellular context you are interested in.
**Weakest link analysis** identifies DDPs whose metabolic coordination breaks
between baseline and perturbed conditions.

The package also supports **metabolic directions** for comparing metabolic and
non-metabolic perturbation effects, **scoring of KEGG pathways and DDPs**,
differential activity analysis, and **visualization of KEGG or custom pathways**
with relevant information such as inferred reaction activity, changes in
activity between conditions, and DDP membership.

## Requirements

- **Software:** Python >= 3.10 on Linux or macOS.
- **Hardware:** CPU execution is supported for the tutorial and small training
  runs. A CUDA-capable NVIDIA GPU is recommended for larger training runs.
  RAM and GPU memory needs depend on dataset size and model settings.

## Installation

Install from GitHub in a dedicated virtual or conda environment. We recommend
Python 3.10, which is used in the supplied tutorial environment.

```bash
python -m pip install "git+https://github.com/wagnerlab-berkeley/mern.git"
```

Check that the package imports successfully:

```bash
python -c "import mern; print(mern.__version__)"
```

If installation fails while building `pygraphviz`, install Graphviz first
(`brew install graphviz` on macOS, or `conda install -c conda-forge graphviz pygraphviz`
inside a conda environment), then rerun the installation.
See the [PyGraphviz installation guide](https://pygraphviz.github.io/documentation/stable/install.html)
for platform-specific troubleshooting.

## Quick Start

Follow the [MeRN tutorial](https://wagnerlab-berkeley.github.io/mern/)
for a walkthrough of core functionality on mouse embryogenesis data: loading
or training a model, extracting embeddings, decoding reaction activity, scoring
pathways, discovering DDPs, visualizing metabolic programs, and comparing
metabolic coordination with weakest link analysis.

The tutorial provides prepared data and pretrained models, so you can explore
these functions on CPU without preprocessing data or training new models.
To get the notebook, data, and models from the repository and run it locally,
follow the [tutorial setup instructions](docs/source/tutorials/README.md).
These tutorial files are not installed by pip.

For larger datasets, you may prefer the command-line workflow below to train
replicate models and average reaction activity.

## Preparing your own data

The tutorial includes preprocessed data and example code for selecting highly
variable genes in your own dataset. Before training on your own data:

1. Store the data in an AnnData object and preserve raw counts in a layer.
   Use species-appropriate gene symbols in `adata.var_names`, matching the
   capitalization of the packaged mappings (for example, `Gapdh` for mouse
   and `GAPDH` for human). Convert Ensembl IDs to gene symbols before annotation.
   The CLI reads `adata.layers["counts"]` by default; use
   `--counts_layer <layer_name>` for a different layer, such as the tutorial's
   `counts_RNA`.
2. Annotate metabolic and non-metabolic genes using
   `mern.support.KeggKGMLMetabolicDataset` and its `add_module_info` method.
   Select highly variable genes separately for the two groups and store the
   resulting boolean flags in `adata.var["highly_variable_metabolic"]` and
   `adata.var["highly_variable_background"]`.
3. Save the prepared AnnData as an `.h5ad` file for training and reaction-activity
   averaging. The CLI uses the gene flags to select features and constructs the
   reaction graph from the packaged metabolic resources.

Use the same species for preprocessing, training, and averaging. The CLI
defaults to mouse; pass `--species human` to both commands for human data.

## Training and averaging reaction activity

The default model hyperparameters for `mern train reps` match those used to
generate the models in the paper.

### Train replicate models

We recommend training **at least 5 independent models** and averaging their
inferred reaction activity. In the paper, reaction-activity consistency plateaued
around 5 models (Figure S1J); the analyses used **15 models for mouse intestine**
and **10 models for each other dataset or condition**, averaging reaction
activity across all replicates.

For example, train 10 models with different random seeds:

```bash
for rep in {0..9}; do
  mern train reps \
    --rep "$rep" \
    --output_dir /path/to/model_reps \
    --data_dir /path/to/data \
    --adata_file data.h5ad
done
```

The training CLI defaults to CUDA. For CPU training, add
`--accelerator cpu --devices 1`.

Each invocation trains one model. `--rep` sets the random seed for initialization
and data splitting and saves the model and embeddings under
`<output_dir>/mern_rep_<rep>/`. On Slurm, the same command can be run in an array
job with `--rep "$SLURM_ARRAY_TASK_ID"` and an array range of `0-9`.

### Average reaction activity

After training, average reaction activity across the replicate models:

```bash
mern average-enzyme-activity \
  --dir /path/to/model_reps \
  --adata_path /path/to/data/data.h5ad
```

This command averages **8 independent Monte Carlo decoding samples per model**
by default, then averages across models, following the paper's procedure
(Figure S1I-J). It saves the cell-by-reaction activity matrix to
`<model_reps>/average_enzyme_activity.csv.gz`. The command name uses
`enzyme-activity` for the inferred reaction activity described in the paper.

For additional options, use `mern train reps --help` or
`mern average-enzyme-activity --help`.

Example Slurm scripts are available for [training replicates](scripts/examples/train_reps.sh),
the [paper's KL-weight search](scripts/examples/kl_search.sh),
[evaluating models](scripts/examples/evaluate_models.sh), and
[averaging reaction activity](scripts/examples/average_reaction_activity.sh).
Activate your MeRN environment and edit the paths, species, and Slurm settings
before submitting.

## KEGG data and reproducibility

MeRN includes fixed, KEGG-derived human and mouse reaction graphs,
reaction-gene mappings, and the associated information needed for training and
plotting. These resources were generated from a [KEGG Release 113.0-era
snapshot](https://www.kegg.jp/kegg/docs/relnote.html) dated January 15, 2025.
MeRN uses this fixed snapshot for reproducibility and does not provide or
continually distribute snapshots of newer KEGG releases.

To rebuild the dataset using current KEGG data, use:

```python
import mern.support as mern_support

dataset = mern_support.KeggKGMLMetabolicDataset(
    species="mouse", rebuild_from_kegg=True
)
```

This option queries KEGG and relies on the academic access provided by the KEGG
REST API. Users are responsible for complying with the [KEGG API
restrictions](https://www.kegg.jp/kegg/rest/) and [KEGG
terms](https://www.kegg.jp/kegg/legal.html). Because KEGG data and API results
can change over time, users who rebuild the dataset should retain a copy of the
downloaded files from `~/.cache/mern/kegg` (or `$XDG_CACHE_HOME/mern/kegg`) for
reproducibility. Pathway plots may also download the pathway-specific KGML used
for their layout.

## Citation

### MeRN

Lewinsohn DP, Dias N, Chau A, Koike Y, Smith ZD, Ioannidis NM, Wagner A. Extracting interpretable single-cell metabolic states with graph-guided representation learning. bioRxiv. 2026. doi:[10.64898/2026.09.17.751504](https://doi.org/10.64898/2026.09.17.751504)

The `v1.0.0` and `v1.0.1` tags are the manuscript-referenced releases before and
after the documented self-loop and reaction-gene mapping fixes.

### Application of MeRN to mouse embryogenesis and source of tutorial data

Dias N, Lewinsohn DP, Colgan WN, Wang M, Kijima Y, Villagrana J, Hou TCJ, Gowri G, Chau A, Aktas T, Sumigray K, Weissman JS, Koblan LW, Wagner A, Smith ZD. Folate deficiency disrupts key metabolic transitions within the developing neural ectoderm. bioRxiv. 2026. doi:[10.64898/2026.09.18.752622](https://doi.org/10.64898/2026.09.18.752622)

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
