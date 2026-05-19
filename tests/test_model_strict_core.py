import numpy as np
import networkx as nx
import pytest
import torch
from scvi import REGISTRY_KEYS
from torch_geometric.utils.convert import from_networkx

from mern import MERN
from mern._constants import GRAPH_REGISTRY_KEYS, MODULE_KEYS
from mern._custom_optimizer import create_rmsprop_optimizer
from mern._mern_data_splitting import MERNDataSplitter
from mern._graph_dataloader import GraphDataLoader


STRICT_MODEL_KWARGS = {
    "n_hidden": 256,
    "n_layers": 2,
    "n_metabolic_dim": 25,
    "n_background_dim": 15,
    "encode_covariates": False,
    "positive_met_dims": False,
    "fixed_rxn_genes": False,
    "strict_met_back_separation": True,
    "rxn_genes_bias": True,
    "fixed_graph_cell_kl": False,
}

STRICT_LOSS_KWARGS = {
    "kl_weight": 0.001,
    "graph_kl_weight": 0.1,
    "data_elbo_weight": 1.0,
    "graph_elbo_weight": 0.2,
    "rxn_genes_weight": 0.5,
    "background_to_metabolic_weight": 30000,
}

SCRIPT_PLAN_KWARGS = {
    "max_kl_weight": 0.001,
    "graph_kl_weight": 0.1,
    "data_elbo_weight": 1.0,
    "graph_elbo_weight": 0.2,
    "rxn_genes_weight": 0.5,
    "background_to_metabolic_weight": 30000,
    "lr": 2e-3,
    "n_steps_kl_warmup": 0,
    "n_epochs_kl_warmup": None,
}


@pytest.fixture
def full_strict_model(general_package_adata_graph_rxn_genes):
    torch.manual_seed(1)
    np.random.seed(1)

    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    adata = adata.copy()
    graph = graph.copy()
    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    return MERN(adata, graph, rxn_to_genes, **STRICT_MODEL_KWARGS)


@pytest.fixture
def batch_corrected_model(general_package_adata_graph_rxn_genes):
    torch.manual_seed(2)
    np.random.seed(2)

    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    adata = adata.copy()
    graph = graph.copy()
    MERN.setup_anndata(adata, layer="counts", batch_key="Phase")
    return MERN(
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


def test_mouse_intestine_kl_search_script_defaults_build_expected_contract(
    general_package_adata_graph_rxn_genes,
):
    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes

    assert adata.shape == (100, 5800)
    assert int(adata.var["highly_variable_metabolic"].sum()) == 800
    assert int(adata.var["highly_variable_background"].sum()) == 5000
    assert not (
        adata.var["highly_variable_metabolic"] & adata.var["highly_variable_background"]
    ).any()
    assert int((adata.var["Metabolic Gene"] == "Metabolic").sum()) == 800

    assert graph.number_of_nodes() == 1213
    assert graph.number_of_edges() == 5587
    assert len(list(nx.selfloop_edges(graph))) == 5
    assert set(graph.nodes()).issubset(rxn_to_genes)

    torch.manual_seed(1)
    np.random.seed(1)
    adata = adata.copy()
    graph = graph.copy()
    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    model = MERN(adata, graph, rxn_to_genes, **STRICT_MODEL_KWARGS)

    assert list(model.gene_names_ordered) == list(adata.var_names)
    assert list(model.vertex_names_ordered) == list(graph.nodes())
    assert model.graph.edge_index.shape == (2, graph.number_of_edges())
    assert model.graph.edge_attr.shape == (graph.number_of_edges(), 2)
    assert model.module.n_hidden == STRICT_MODEL_KWARGS["n_hidden"]
    assert model.module.n_layers == STRICT_MODEL_KWARGS["n_layers"]
    assert model.module.n_metabolic_dim == STRICT_MODEL_KWARGS["n_metabolic_dim"]
    assert model.module.n_background_dim == STRICT_MODEL_KWARGS["n_background_dim"]


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


def test_strict_model_kwargs_reach_core_components(full_strict_model):
    module = full_strict_model.module
    decoder = module.decoder
    rxn_gene_layer = decoder.rxn_gene_layer
    n_reactions = len(full_strict_model.vertex_names_ordered)

    assert module.n_hidden == 256
    assert module.n_layers == 2
    assert module.n_metabolic_dim == 25
    assert module.n_background_dim == 15
    assert module.encode_covariates is False
    assert module.positive_met_dims is False
    assert module.fixed_graph_cell_kl is False

    assert module.m_encoder.mean_encoder.out_features == 25
    assert module.b_encoder.mean_encoder.out_features == 15
    assert module.v_encoder.vrepr.shape == (n_reactions, 25)
    assert decoder.back_linear.in_features == 15
    assert decoder.fixed_rxn_genes is False
    assert decoder.strict_met_back_separation is True
    assert rxn_gene_layer.strict_met_back_separation is True
    assert rxn_gene_layer.linear.bias is not None
    assert rxn_gene_layer.linear.in_features == n_reactions
    assert rxn_gene_layer.linear.out_features == full_strict_model.adata.n_vars


def test_batch_key_reaches_model_and_decoder_covariates(batch_corrected_model):
    n_reactions = len(batch_corrected_model.vertex_names_ordered)

    assert batch_corrected_model.summary_stats.n_batch == 3
    assert batch_corrected_model.module.n_batch == 3
    assert batch_corrected_model.module.encode_covariates is False
    assert batch_corrected_model.module.decoder.n_cov == 3
    assert batch_corrected_model.module.decoder.rxn_gene_layer.n_cov == 3
    assert batch_corrected_model.module.decoder.rxn_gene_layer.linear.in_features == (
        n_reactions + 3
    )
    assert batch_corrected_model.module.decoder.back_linear.in_features == 3 + 3

    weights = batch_corrected_model.get_rxn_genes_weights(return_numpy=False)
    assert list(weights.columns[-3:]) == [
        "decoder_covariate_one_hot_0",
        "decoder_covariate_one_hot_1",
        "decoder_covariate_one_hot_2",
    ]

    batch = _one_model_batch(batch_corrected_model, batch_size=12)
    observed_batch_codes = batch["cells"][REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
    assert set(observed_batch_codes.detach().cpu().numpy().astype(int)).issubset({0, 1, 2})


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


def test_deterministic_splitter_and_dataloader_contract(full_strict_model):
    splitter = MERNDataSplitter(
        full_strict_model.adata_manager,
        full_strict_model.graph,
        train_size=0.8,
        validation_size=0.2,
        shuffle_set_split=False,
        batch_size=32,
    )
    splitter.setup()

    assert splitter.val_idx.tolist() == list(range(20))
    assert splitter.train_idx.tolist() == list(range(20, 100))
    assert splitter.test_idx.tolist() == []

    batch = next(iter(splitter.train_dataloader()))
    cells = batch["cells"]
    graph_batch = batch["graph"]
    positive_edge_count = full_strict_model.graph.edge_index.shape[1]

    assert cells[REGISTRY_KEYS.X_KEY].shape == (32, full_strict_model.adata.n_vars)
    assert graph_batch[GRAPH_REGISTRY_KEYS.EIDX_KEY].shape[0] == 2
    assert graph_batch[GRAPH_REGISTRY_KEYS.EWT_KEY].shape[0] == positive_edge_count * 2
    assert graph_batch[GRAPH_REGISTRY_KEYS.ESGN_KEY].shape[0] == positive_edge_count * 2

    positive_edges = {
        tuple(edge)
        for edge in full_strict_model.graph.edge_index.t()
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


def test_module_input_mapping_matches_mouse_intestine_batch_contract(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    positive_edge_count = full_strict_model.graph.edge_index.shape[1]

    inference_input = full_strict_model.module._get_inference_input(batch)
    inference_outputs = full_strict_model.module.inference(**inference_input)
    generative_input = full_strict_model.module._get_generative_input(batch, inference_outputs)

    assert inference_input[MODULE_KEYS.X_KEY].shape == (32, full_strict_model.adata.n_vars)
    assert inference_input[MODULE_KEYS.BATCH_INDEX_KEY].shape == (32, 1)
    assert inference_input[MODULE_KEYS.EIDX_KEY].shape == (2, positive_edge_count * 2)
    assert inference_input[MODULE_KEYS.EWT_KEY].shape == (positive_edge_count * 2,)
    assert inference_input[MODULE_KEYS.ESGN_KEY].shape == (positive_edge_count * 2,)

    assert torch.all(inference_input[MODULE_KEYS.EWT_KEY][:positive_edge_count] == 1)
    assert torch.all(inference_input[MODULE_KEYS.EWT_KEY][positive_edge_count:] == 0)
    assert torch.all(inference_input[MODULE_KEYS.ESGN_KEY] == 1)

    assert generative_input[MODULE_KEYS.M_KEY] is inference_outputs[MODULE_KEYS.M_KEY]
    assert generative_input[MODULE_KEYS.B_KEY] is inference_outputs[MODULE_KEYS.B_KEY]
    assert generative_input[MODULE_KEYS.V_KEY] is inference_outputs[MODULE_KEYS.V_KEY]
    assert generative_input[MODULE_KEYS.EIDX_KEY] is batch["graph"][GRAPH_REGISTRY_KEYS.EIDX_KEY]
    assert generative_input[MODULE_KEYS.EWT_KEY] is batch["graph"][GRAPH_REGISTRY_KEYS.EWT_KEY]
    assert generative_input[MODULE_KEYS.ESGN_KEY] is batch["graph"][GRAPH_REGISTRY_KEYS.ESGN_KEY]


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


def test_strict_forward_semantics(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    inference_outputs, generative_outputs = _model_forward_outputs(full_strict_model, batch)
    batch_size = batch["cells"][REGISTRY_KEYS.X_KEY].shape[0]
    n_genes = full_strict_model.adata.n_vars
    n_reactions = len(full_strict_model.vertex_names_ordered)
    is_metabolic = full_strict_model.module.gene_is_metabolic

    assert inference_outputs[MODULE_KEYS.QM_KEY].loc.shape == (batch_size, 25)
    assert inference_outputs[MODULE_KEYS.QB_KEY].loc.shape == (batch_size, 15)
    assert inference_outputs[MODULE_KEYS.QV_KEY].loc.shape == (n_reactions, 25)
    assert torch.isfinite(inference_outputs[MODULE_KEYS.METABOLIC_LIBRARY_KEY]).all()
    assert torch.isfinite(inference_outputs[MODULE_KEYS.BACKGROUND_LIBRARY_KEY]).all()

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


def test_batch_transform_changes_module_generative_outputs(batch_corrected_model):
    batch = _one_model_batch(batch_corrected_model, batch_size=16)
    batch_corrected_model.module.eval()
    with torch.no_grad():
        inference_outputs = batch_corrected_model.module.inference(
            **batch_corrected_model.module._get_inference_input(batch)
        )
        generative_input = batch_corrected_model.module._get_generative_input(
            batch, inference_outputs
        )
        batch_0_outputs = batch_corrected_model.module.generative(
            **generative_input,
            transform_batch=torch.tensor(0),
        )
        batch_1_outputs = batch_corrected_model.module.generative(
            **generative_input,
            transform_batch=torch.tensor(1),
        )

    batch_0_mu = batch_0_outputs[MODULE_KEYS.PX_KEY].get_normalized("mu")
    batch_1_mu = batch_1_outputs[MODULE_KEYS.PX_KEY].get_normalized("mu")
    batch_0_met_scale = batch_0_outputs[MODULE_KEYS.PX_MET_KEY].get_normalized("scale")
    batch_1_met_scale = batch_1_outputs[MODULE_KEYS.PX_MET_KEY].get_normalized("scale")
    is_metabolic = batch_corrected_model.module.gene_is_metabolic

    assert not torch.allclose(batch_0_mu, batch_1_mu)
    assert not torch.allclose(batch_0_met_scale, batch_1_met_scale)
    assert torch.all(batch_0_met_scale[:, ~is_metabolic] == 0)
    assert torch.all(batch_1_met_scale[:, ~is_metabolic] == 0)


def test_batch_transform_reaches_downstream_decoding_apis(batch_corrected_model):
    indices = [0, 1, 2, 3]
    categories = batch_corrected_model.adata.obs["Phase"].cat.categories
    first_batch = categories[0]
    second_batch = categories[1]

    torch.manual_seed(10)
    first_decoding = batch_corrected_model.get_decoding(
        indices=indices,
        transform_batch=[first_batch],
        batch_size=4,
        return_numpy=True,
    )
    torch.manual_seed(10)
    second_decoding = batch_corrected_model.get_decoding(
        indices=indices,
        transform_batch=[second_batch],
        batch_size=4,
        return_numpy=True,
    )

    assert not np.allclose(first_decoding["full_mu"], second_decoding["full_mu"])
    assert np.allclose(
        first_decoding["full_mu"],
        first_decoding["met_mu"] + first_decoding["bg_mu"],
    )

    torch.manual_seed(20)
    first_expression = batch_corrected_model.get_normalized_expression(
        indices=indices,
        transform_batch=[first_batch],
        library_size="latent",
        batch_size=4,
        return_numpy=True,
    )
    torch.manual_seed(20)
    second_expression = batch_corrected_model.get_normalized_expression(
        indices=indices,
        transform_batch=[second_batch],
        library_size="latent",
        batch_size=4,
        return_numpy=True,
    )

    assert first_expression.shape == (len(indices), batch_corrected_model.adata.n_vars)
    assert second_expression.shape == (len(indices), batch_corrected_model.adata.n_vars)
    assert not np.allclose(first_expression, second_expression)


def test_get_latent_representation_returns_cell_and_graph_latents(full_strict_model):
    indices = np.arange(6)
    n_reactions = len(full_strict_model.vertex_names_ordered)

    with pytest.raises(RuntimeError, match="untrained model"):
        full_strict_model.get_latent_representation(
            indices=indices,
            batch_size=len(indices),
        )

    full_strict_model.is_trained_ = True
    metabolic_latent, background_latent, graph_latent = (
        full_strict_model.get_latent_representation(
            indices=indices,
            batch_size=len(indices),
        )
    )

    assert metabolic_latent.shape == (len(indices), 25)
    assert background_latent.shape == (len(indices), 15)
    assert graph_latent.shape == (n_reactions, 25)
    assert np.isfinite(metabolic_latent).all()
    assert np.isfinite(background_latent).all()
    assert np.isfinite(graph_latent).all()

    (
        metabolic_mean,
        metabolic_var,
        background_mean,
        background_var,
        graph_mean,
        graph_var,
    ) = full_strict_model.get_latent_representation(
        indices=indices,
        batch_size=len(indices),
        return_dist=True,
    )

    assert metabolic_mean.shape == (len(indices), 25)
    assert metabolic_var.shape == (len(indices), 25)
    assert background_mean.shape == (len(indices), 15)
    assert background_var.shape == (len(indices), 15)
    assert graph_mean.shape == (n_reactions, 25)
    assert graph_var.shape == (n_reactions, 25)
    for value in [
        metabolic_mean,
        metabolic_var,
        background_mean,
        background_var,
        graph_mean,
        graph_var,
    ]:
        assert np.isfinite(value).all()
    assert np.all(metabolic_var >= 0)
    assert np.all(background_var >= 0)
    assert np.all(graph_var >= 0)


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


def test_graph_loss_uses_positive_and_negative_edges(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    inference_outputs, generative_outputs = _model_forward_outputs(full_strict_model, batch)
    loss_output = full_strict_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        **STRICT_LOSS_KWARGS,
    )

    ewt = batch["graph"][GRAPH_REGISTRY_KEYS.EWT_KEY]
    pg = generative_outputs[MODULE_KEYS.PG_KEY]
    g_nll_by_edge = -pg.log_prob(ewt)
    positive_mask = ewt != 0
    negative_mask = ~positive_mask

    assert positive_mask.sum().item() == full_strict_model.graph.edge_index.shape[1]
    assert negative_mask.sum().item() == full_strict_model.graph.edge_index.shape[1]
    assert g_nll_by_edge.shape == ewt.shape
    assert torch.isfinite(g_nll_by_edge[positive_mask]).all()
    assert torch.isfinite(g_nll_by_edge[negative_mask]).all()
    assert torch.isfinite(loss_output.extra_metrics["g_nll"])


def test_loss_uses_expected_weighted_terms(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    inference_outputs, generative_outputs = _model_forward_outputs(full_strict_model, batch)
    loss_output = full_strict_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        **STRICT_LOSS_KWARGS,
    )

    x_elbo = (
        loss_output.reconstruction_loss["reconstruction_loss"]
        + STRICT_LOSS_KWARGS["kl_weight"]
        * (
            loss_output.kl_local[MODULE_KEYS.KL_M_KEY]
            + loss_output.kl_local[MODULE_KEYS.KL_B_KEY]
        )
        + loss_output.kl_local[MODULE_KEYS.KL_LM_KEY]
        + loss_output.kl_local[MODULE_KEYS.KL_LB_KEY]
    ).mean()
    graph_elbo = (
        loss_output.extra_metrics["g_nll"]
        + STRICT_LOSS_KWARGS["graph_kl_weight"] * loss_output.extra_metrics["kl_v"]
    )
    expected_loss = (
        STRICT_LOSS_KWARGS["data_elbo_weight"] * x_elbo
        + STRICT_LOSS_KWARGS["graph_elbo_weight"] * graph_elbo
        + STRICT_LOSS_KWARGS["rxn_genes_weight"] * loss_output.extra_metrics["rxns_to_genes_loss"]
        + STRICT_LOSS_KWARGS["background_to_metabolic_weight"]
        * loss_output.extra_metrics["background_to_metabolic_loss"]
    )

    assert torch.isfinite(loss_output.loss)
    assert torch.allclose(loss_output.loss, expected_loss)
    assert loss_output.extra_metrics["rxns_to_genes_loss"] == 0
    assert loss_output.extra_metrics["background_to_metabolic_loss"] == 0


def test_fixed_graph_cell_kl_switches_graph_kl_to_cell_kl_weight(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    inference_outputs, generative_outputs = _model_forward_outputs(full_strict_model, batch)

    common_loss_kwargs = {
        "kl_weight": 0.25,
        "graph_kl_weight": 0.0,
        "data_elbo_weight": 0.0,
        "graph_elbo_weight": 1.0,
        "rxn_genes_weight": 0.0,
        "background_to_metabolic_weight": 0.0,
    }

    full_strict_model.module.fixed_graph_cell_kl = False
    graph_weighted_loss = full_strict_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        **common_loss_kwargs,
    )
    full_strict_model.module.fixed_graph_cell_kl = True
    cell_weighted_loss = full_strict_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        **common_loss_kwargs,
    )
    full_strict_model.module.fixed_graph_cell_kl = False

    expected_delta = common_loss_kwargs["kl_weight"] * cell_weighted_loss.extra_metrics["kl_v"]
    assert torch.allclose(cell_weighted_loss.loss - graph_weighted_loss.loss, expected_delta)


def test_non_strict_auxiliary_losses_are_active(general_package_adata_graph_rxn_genes):
    torch.manual_seed(3)
    np.random.seed(3)

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
        strict_met_back_separation=False,
        rxn_genes_bias=True,
    )

    batch = _one_model_batch(model, batch_size=8)
    inference_outputs, generative_outputs = _model_forward_outputs(model, batch)
    loss_output = model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        rxn_genes_weight=0.5,
        background_to_metabolic_weight=30000,
    )

    assert torch.isfinite(loss_output.loss)
    assert loss_output.extra_metrics["rxns_to_genes_loss"] > 0
    assert loss_output.extra_metrics["background_to_metabolic_loss"] > 0


def test_strict_masks_survive_optimizer_step(full_strict_model):
    batch = _one_model_batch(full_strict_model, batch_size=32)
    optimizer = create_rmsprop_optimizer(full_strict_model.module.parameters(), lr=2e-3)

    full_strict_model.module.train()
    inference_outputs = full_strict_model.module.inference(
        **full_strict_model.module._get_inference_input(batch)
    )
    generative_outputs = full_strict_model.module.generative(
        **full_strict_model.module._get_generative_input(batch, inference_outputs)
    )
    loss_output = full_strict_model.module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        **STRICT_LOSS_KWARGS,
    )
    optimizer.zero_grad()
    loss_output.loss.backward()
    optimizer.step()

    _model_forward_outputs(full_strict_model, batch)
    layer = full_strict_model.module.decoder.rxn_gene_layer
    strict_weights = layer.linear.weight.detach()
    disallowed_mask = ~layer.weight_mask.bool()

    assert torch.isfinite(loss_output.loss)
    assert torch.all(strict_weights >= 0)
    assert torch.all(
        strict_weights[:, : len(full_strict_model.vertex_names_ordered)][disallowed_mask] == 0
    )


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


def test_mouse_intestine_script_default_train_and_post_train_outputs(
    general_package_adata_graph_rxn_genes,
):
    torch.manual_seed(1)
    np.random.seed(1)

    adata, graph, rxn_to_genes = general_package_adata_graph_rxn_genes
    adata = adata.copy()
    graph = graph.copy()
    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    model = MERN(adata, graph, rxn_to_genes, **STRICT_MODEL_KWARGS)

    initial_state_abs_sum = sum(
        tensor.detach().abs().sum().item()
        for tensor in model.module.state_dict().values()
        if torch.is_tensor(tensor)
    )
    model.train(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        early_stopping=False,
        train_size=0.8,
        validation_size=0.2,
        shuffle_set_split=False,
        batch_size=32,
        plan_kwargs=SCRIPT_PLAN_KWARGS,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trained_state_abs_sum = sum(
        tensor.detach().abs().sum().item()
        for tensor in model.module.state_dict().values()
        if torch.is_tensor(tensor)
    )

    assert trained_state_abs_sum != initial_state_abs_sum
    assert model.history is None

    indices = np.arange(4)
    decoding = model.get_decoding(indices=indices, batch_size=4, return_numpy=True)
    assert set(decoding) == {
        "enzyme_activity",
        "met_mu",
        "met_scale",
        "bg_mu",
        "bg_scale",
        "full_mu",
    }
    assert decoding["enzyme_activity"].shape == (len(indices), len(model.vertex_names_ordered))
    for key in ["met_mu", "met_scale", "bg_mu", "bg_scale", "full_mu"]:
        assert decoding[key].shape == (len(indices), model.adata.n_vars)
        assert np.isfinite(decoding[key]).all()
    assert np.isfinite(decoding["enzyme_activity"]).all()
    assert np.allclose(decoding["full_mu"], decoding["met_mu"] + decoding["bg_mu"])

    normalized = model.get_normalized_expression(
        indices=indices,
        library_size="latent",
        batch_size=4,
        return_numpy=True,
    )
    assert normalized.shape == (len(indices), model.adata.n_vars)
    assert np.isfinite(normalized).all()
