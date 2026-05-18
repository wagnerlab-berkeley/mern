import numpy as np
import torch

from mern import MERN


def test_mern_setup_init_and_forward_smoke(general_package_adata_graph_rxn_genes):
    torch.manual_seed(0)
    np.random.seed(0)

    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    adata = adata.copy()
    graph = graph.copy()

    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    model = MERN(
        adata,
        graph,
        rxn_to_genes,
        n_hidden=32,
        n_layers=1,
        n_metabolic_dim=5,
        n_background_dim=3,
        dropout_rate=0.0,
        encode_covariates=False,
        strict_met_back_separation=True,
        rxn_genes_bias=True,
    )

    assert model.module.n_input == adata.n_vars
    assert model.module.n_metabolic_dim == 5
    assert model.module.n_background_dim == 3
    assert len(model.vertex_names_ordered) == graph.number_of_nodes()

    data_loader = model._make_data_loader(
        model.adata, indices=np.arange(model.adata.n_obs), batch_size=16
    )
    batch = next(iter(data_loader))
    model.module.eval()
    with torch.no_grad():
        _, _, loss_output = model.module(batch)

    assert torch.isfinite(loss_output.loss)
    assert "g_nll" in loss_output.extra_metrics
    assert "kl_v" in loss_output.extra_metrics
    assert loss_output.reconstruction_loss["reconstruction_loss"].shape[0] == 16
