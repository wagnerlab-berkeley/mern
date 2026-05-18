import numpy as np
import networkx as nx
import pytest
import torch
from scvi import REGISTRY_KEYS
from torch_geometric.utils.convert import from_networkx

from mern._constants import GRAPH_REGISTRY_KEYS, MODULE_KEYS
from mern._graph_dataloader import GraphDataLoader


def _dense_counts(adata):
    counts = adata.layers["counts"]
    return counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)


def _one_model_batch(model, batch_size=8):
    data_loader = model._make_data_loader(
        model.adata, indices=np.arange(model.adata.n_obs), batch_size=batch_size
    )
    return next(iter(data_loader))


def _model_forward_outputs(model, batch):
    model.module.eval()
    with torch.no_grad():
        inference_outputs = model.module.inference(**model.module._get_inference_input(batch))
        generative_outputs = model.module.generative(
            **model.module._get_generative_input(batch, inference_outputs)
        )
    return inference_outputs, generative_outputs


def test_support_inputs_select_expected_features_and_counts(
    mouse_intestine_100, general_package_adata_graph_rxn_genes
):
    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    expected_features = list(
        mouse_intestine_100.var_names[
            mouse_intestine_100.var["highly_variable_metabolic"]
            | mouse_intestine_100.var["highly_variable_background"]
        ]
    )
    expected_features.sort()

    assert list(adata.var_names) == expected_features
    assert adata.var["highly_variable_metabolic"].any()
    assert adata.var["highly_variable_background"].any()

    gene_to_active_rxns = {}
    for reaction in graph.nodes:
        for gene in rxn_to_genes.get(reaction, []):
            gene_to_active_rxns.setdefault(gene, set()).add(reaction)

    metabolic_genes = adata.var_names[adata.var["highly_variable_metabolic"]]
    background_genes = adata.var_names[adata.var["highly_variable_background"]]
    assert all(gene in gene_to_active_rxns for gene in metabolic_genes)
    assert all(gene not in gene_to_active_rxns for gene in background_genes)

    counts = _dense_counts(adata)
    assert np.all(counts >= 0)
    assert np.allclose(counts, np.rint(counts))

    is_metabolic = (adata.var["Metabolic Gene"] == "Metabolic").to_numpy()
    assert np.all(counts[:, is_metabolic].sum(axis=1) > 0)
    assert np.all(counts[:, ~is_metabolic].sum(axis=1) > 0)


def test_model_strict_preserves_feature_order(general_package_model):
    assert list(general_package_model.gene_names_ordered) == list(
        general_package_model.adata.var_names
    )
    assert list(general_package_model.module.genes) == list(general_package_model.adata.var_names)
    assert list(general_package_model.module.decoder.rxn_gene_layer.genes) == list(
        general_package_model.adata.var_names
    )

    decoding = general_package_model.get_decoding(indices=[0, 1], batch_size=2, return_numpy=False)
    assert list(decoding["met_mu"].columns) == list(general_package_model.adata.var_names)
    assert list(decoding["bg_mu"].columns) == list(general_package_model.adata.var_names)
    assert list(decoding["full_mu"].columns) == list(general_package_model.adata.var_names)


@pytest.mark.xfail(
    reason="The current fixture graph contains five natural self-loops despite self_loops=False.",
    strict=True,
)
def test_support_graph_has_no_self_loops(general_package_adata_graph_rxn_genes):
    _, graph, _ = general_package_adata_graph_rxn_genes
    assert not list(nx.selfloop_edges(graph))


def test_support_graph_preserves_attributes_and_pyg_order(
    general_package_adata_graph_rxn_genes,
):
    _, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    assert set(graph.nodes()).issubset(rxn_to_genes)

    edge_records = list(graph.edges(data=True))
    assert edge_records
    for _, _, edge_data in edge_records:
        assert isinstance(edge_data["weight"], (int, float))
        assert edge_data["weight"] >= 0
        assert edge_data["sign"] in {-1, 1}

    pyg_graph = from_networkx(graph, group_edge_attrs=["weight", "sign"])
    node_to_index = {node: index for index, node in enumerate(graph.nodes())}
    for edge_index, (source, target, edge_data) in enumerate(edge_records):
        assert pyg_graph.edge_index[0, edge_index].item() == node_to_index[source]
        assert pyg_graph.edge_index[1, edge_index].item() == node_to_index[target]
        assert pyg_graph.edge_attr[edge_index, 0].item() == edge_data["weight"]
        assert pyg_graph.edge_attr[edge_index, 1].item() == edge_data["sign"]


def test_graph_dataloader_negative_samples_exclude_positive_edges_and_self_loops(
    general_package_model,
):
    positive_edge_count = general_package_model.graph.edge_index.shape[1]
    graph_batch = next(
        iter(
            GraphDataLoader(
                general_package_model.graph, length=1, neg_sampling_ratio=1, undirected=False
            )
        )
    )

    positive_edges = {
        tuple(edge)
        for edge in general_package_model.graph.edge_index.t()
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .tolist()
    }
    negative_edges = (
        graph_batch[GRAPH_REGISTRY_KEYS.EIDX_KEY][:, positive_edge_count:]
        .t()
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .tolist()
    )

    assert negative_edges
    assert all(tuple(edge) not in positive_edges for edge in negative_edges)
    assert all(source != target for source, target in negative_edges)
    assert torch.all(graph_batch[GRAPH_REGISTRY_KEYS.EWT_KEY][positive_edge_count:] == 0)
    assert torch.all(graph_batch[GRAPH_REGISTRY_KEYS.ESGN_KEY][positive_edge_count:] == 1)


def test_rxn_gene_layer_strict_mask_matches_annotations_and_zeroes_disallowed_weights(
    general_package_model,
):
    layer = general_package_model.module.decoder.rxn_gene_layer
    weights = general_package_model.get_rxn_genes_weights(return_numpy=False)

    assert weights.shape == (
        general_package_model.adata.n_vars,
        len(general_package_model.vertex_names_ordered),
    )
    assert list(weights.columns) == general_package_model.vertex_names_ordered

    mask = layer.weight_mask.detach().cpu().numpy().astype(bool)
    genes = np.asarray(layer.genes)
    for gene_index, reaction_index in np.argwhere(mask):
        reaction = general_package_model.vertex_names_ordered[reaction_index]
        assert genes[gene_index] in general_package_model.rxn_to_genes[reaction]

    is_metabolic = general_package_model.module.gene_is_metabolic
    assert not mask[~is_metabolic, :].any()

    batch = _one_model_batch(general_package_model)
    _model_forward_outputs(general_package_model, batch)
    strict_weights = layer.linear.weight.detach()
    disallowed_mask = ~layer.weight_mask.bool()
    assert torch.all(strict_weights >= 0)
    assert torch.all(
        strict_weights[:, : len(general_package_model.vertex_names_ordered)][disallowed_mask] == 0
    )


def test_model_strict_forward_shapes_and_decoder_separation(general_package_model):
    batch = _one_model_batch(general_package_model)
    inference_outputs, generative_outputs = _model_forward_outputs(general_package_model, batch)
    batch_size = batch["cells"][REGISTRY_KEYS.X_KEY].shape[0]
    n_genes = general_package_model.adata.n_vars
    n_reactions = len(general_package_model.vertex_names_ordered)
    is_metabolic = general_package_model.module.gene_is_metabolic

    assert inference_outputs[MODULE_KEYS.QM_KEY].loc.shape == (batch_size, 25)
    assert inference_outputs[MODULE_KEYS.QB_KEY].loc.shape == (batch_size, 15)
    assert inference_outputs[MODULE_KEYS.QV_KEY].loc.shape == (n_reactions, 25)

    enzyme_activity = generative_outputs[MODULE_KEYS.ENZYME_ACTIVITY_KEY]
    assert enzyme_activity.shape == (batch_size, n_reactions)
    assert torch.all(enzyme_activity >= 0)

    met_mu = generative_outputs[MODULE_KEYS.PX_MET_KEY].get_normalized("mu")
    bg_mu = generative_outputs[MODULE_KEYS.PX_BACK_KEY].get_normalized("mu")
    full_mu = generative_outputs[MODULE_KEYS.PX_KEY].get_normalized("mu")
    assert met_mu.shape == (batch_size, n_genes)
    assert bg_mu.shape == (batch_size, n_genes)
    assert torch.allclose(full_mu, met_mu + bg_mu)

    met_scale = generative_outputs[MODULE_KEYS.PX_MET_KEY].get_normalized("scale")
    bg_scale = generative_outputs[MODULE_KEYS.PX_BACK_KEY].get_normalized("scale")
    assert torch.all(met_scale[:, ~is_metabolic] == 0)
    assert torch.all(bg_scale[:, is_metabolic] == 0)
    assert torch.allclose(met_scale[:, is_metabolic].sum(dim=1), torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(bg_scale[:, ~is_metabolic].sum(dim=1), torch.ones(batch_size), atol=1e-5)

    decoding = general_package_model.get_decoding(
        indices=np.arange(batch_size), batch_size=batch_size
    )
    assert list(decoding["enzyme_activity"].columns) == general_package_model.vertex_names_ordered


def test_loss_terms_are_finite_and_strict_auxiliary_losses_are_zero(
    general_package_model,
):
    batch = _one_model_batch(general_package_model)
    inference_outputs, generative_outputs = _model_forward_outputs(general_package_model, batch)
    loss_output = general_package_model.module.loss(batch, inference_outputs, generative_outputs)

    assert torch.isfinite(loss_output.loss)
    assert all(torch.isfinite(value).all() for value in loss_output.reconstruction_loss.values())
    assert all(torch.isfinite(value).all() for value in loss_output.kl_local.values())
    assert torch.isfinite(loss_output.extra_metrics["g_nll"])
    assert torch.isfinite(loss_output.extra_metrics["kl_v"])
    assert loss_output.extra_metrics["rxns_to_genes_loss"] == 0
    assert loss_output.extra_metrics["background_to_metabolic_loss"] == 0


def test_loss_weights_scale_cell_and_graph_terms(general_package_model):
    batch = _one_model_batch(general_package_model)
    inference_outputs, generative_outputs = _model_forward_outputs(general_package_model, batch)

    no_cell_kl = general_package_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        kl_weight=0.0,
        data_elbo_weight=1.0,
        graph_elbo_weight=0.0,
    )
    weighted_cell_kl = general_package_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        kl_weight=0.25,
        data_elbo_weight=1.0,
        graph_elbo_weight=0.0,
    )
    expected_cell_delta = (
        0.25
        * (
            weighted_cell_kl.kl_local[MODULE_KEYS.KL_M_KEY]
            + weighted_cell_kl.kl_local[MODULE_KEYS.KL_B_KEY]
        ).mean()
    )
    assert torch.allclose(weighted_cell_kl.loss - no_cell_kl.loss, expected_cell_delta)

    no_graph_kl = general_package_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        data_elbo_weight=0.0,
        graph_elbo_weight=1.0,
        graph_kl_weight=0.0,
    )
    weighted_graph_kl = general_package_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        data_elbo_weight=0.0,
        graph_elbo_weight=1.0,
        graph_kl_weight=0.5,
    )
    expected_graph_delta = 0.5 * weighted_graph_kl.extra_metrics["kl_v"]
    assert torch.allclose(weighted_graph_kl.loss - no_graph_kl.loss, expected_graph_delta)

    graph_elbo_weighted = general_package_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        data_elbo_weight=0.0,
        graph_elbo_weight=3.0,
        graph_kl_weight=0.5,
    )
    assert torch.allclose(graph_elbo_weighted.loss, 3.0 * weighted_graph_kl.loss)
