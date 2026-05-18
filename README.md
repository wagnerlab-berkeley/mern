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
