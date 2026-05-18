from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

from mern import MERN

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def mouse_intestine_100_path() -> Path:
    return TEST_DATA_DIR / "mouse_intestine_100.h5ad"


@pytest.fixture(scope="session")
def mouse_intestine_100(mouse_intestine_100_path):
    return ad.read_h5ad(mouse_intestine_100_path)


@pytest.fixture(scope="session")
def mouse_kegg_dataset():
    from mern.support import KeggKGMLMetabolicDataset

    return KeggKGMLMetabolicDataset(
        species="mouse",
        capitalize=False,
        add_oxphos=True,
    )


@pytest.fixture(scope="session")
def general_package_adata_graph_rxn_genes(mouse_intestine_100_path, mouse_kegg_dataset):
    adata = ad.read_h5ad(mouse_intestine_100_path)
    mouse_kegg_dataset.add_module_info(adata)
    features = list(
        adata.var_names[
            adata.var["highly_variable_metabolic"] | adata.var["highly_variable_background"]
        ]
    )
    features.sort()
    adata = adata[:, features].copy()

    graph = mouse_kegg_dataset.metabolic_topology(adata, self_loops=False)
    mouse_kegg_dataset.add_rxn_module_info(adata, graph)
    rxn_to_genes = mouse_kegg_dataset.get_rxn_genes_all()

    return adata, graph, rxn_to_genes


@pytest.fixture
def general_package_model(general_package_adata_graph_rxn_genes):
    torch.manual_seed(0)
    np.random.seed(0)

    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    adata = adata.copy()
    graph = graph.copy()
    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    return MERN(
        adata,
        graph,
        rxn_to_genes,
        n_hidden=32,
        n_layers=1,
        n_metabolic_dim=25,
        n_background_dim=15,
        dropout_rate=0.0,
        encode_covariates=False,
        strict_met_back_separation=True,
        rxn_genes_bias=True,
    )
