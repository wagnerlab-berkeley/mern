# MeRN tutorial

A walkthrough of MeRN on mouse embryogenesis data: loading a pretrained model, extracting the
metabolic and background embeddings, decoding reaction activity, building KEGG pathway scores and
data-driven pathways (DDPs), and comparing metabolic coordination between wild-type and *Folr1*
knockout embryos.

## Running it

```bash
cd docs/source/tutorials
conda env create -f environment.yml
conda activate mern-tutorial
jupyter lab MeRN_tutorial.ipynb
```

Create the environment from this directory: `environment.yml` installs the `mern` package from the
surrounding clone by relative path.

Everything the notebook reads is in this directory.

### Installing without conda

```bash
brew install graphviz            # or: apt-get install graphviz graphviz-dev
pip install -e ../../.. kaleido jupyterlab ipywidgets
```

## What this folder contains

| path | contents |
| --- | --- |
| `data/main_wt_anndata_pp_subsample.h5ad.gz` | wild-type cells, model feature space, log1p `X` plus raw `counts_RNA` |
| `data/folr1_anndata_subsample.h5ad.gz` | *Folr1* knockout cells, same feature space |
| `data/folr1_average_enzyme_activity.csv.gz` | knockout reaction activity, its own replicate ensemble |
| `data/wt_ref_color_mapping.csv` | cell-type colors used in the paper's figures |
| `model/mern_rep_0/model.pt.gz` … `mern_rep_4/` | five pretrained replicate checkpoints, gzipped |
| `model/mern_rep_0/metabolic_clustering_0.4.csv` | the paper's metabolic clusters, for comparison with the ones recomputed here |

## About the subsampled data

This is a **cell subsample** of the published dataset. The full study covers
94,204 cells (62,433 wild-type, 31,771 *Folr1*); here is a stratified sample of those,
drawn proportionally across (metabolic state x germ layer).

Only cells were sampled. The gene space, the `highly_variable_metabolic` /
`highly_variable_background` flags, the reaction graph, and the models are exactly those of the
published run, so the pretrained checkpoints load and decode as they do on the full data. The
wild-type reaction activity is decoded in the notebook from the 5 included replicates. The *Folr1*
reaction activity table is the published 8-decode, 10-replicate ensemble, subset to these cells.