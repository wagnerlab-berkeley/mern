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

Documentation is not published yet. For now, the main package code lives under `src/mern`, and the repository includes the core model, support utilities, and packaged KEGG resources.

## Installation

We recommend running `mern` on a recent Linux or macOS system with Python >= 3.10.

Install the package in editable mode with:

```bash
pip install -e .
```

For development with plotting, KEGG utilities, and test dependencies:

```bash
pip install -e ".[plot,kegg,dev]"
```

## Key capabilities

- Learn graph-guided latent representations of cell state from single-cell RNA-seq data.
- Separate metabolic variation from background, non-metabolic transcriptional variation.
- Use reaction graphs and reaction-to-gene mappings to tie latent structure to metabolic biology.
- Work with KEGG-derived metabolic datasets and packaged metabolic network resources.
- Run downstream analysis and visualization utilities for pathway activity, latent factors, and topology-aware plots.

## Package layout

The current repository contains:

- `src/mern/model.py`: main MERN model class
- `src/mern/module.py`: core neural module
- `src/mern/base_components.py`: graph and decoder components
- `src/mern/datasets.py`: metabolic dataset and KEGG utilities
- `src/mern/analysis.py`: downstream analysis helpers
- `src/mern/plots.py`: plotting helpers

## Contributing

Contributions are welcome. The next practical steps for this repository are:

- validating package installation
- resolving remaining import and dependency issues
- running and expanding tests
- improving the user-facing API and documentation

## Citation

Citation information will be added here later.
