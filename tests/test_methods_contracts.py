import ast
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial
from torch.distributions import Normal, kl_divergence
from torch_geometric.data import Data

from mern import MERN
from mern._base_components import DecoderMERN, GraphDecoder
from mern._constants import GRAPH_REGISTRY_KEYS, MODULE_KEYS
from mern._custom_optimizer import create_rmsprop_optimizer
from mern._graph_dataloader import GraphDataLoader
from mern._module import MERNModule
from mern.support import (
    KeggKGMLMetabolicDataset,
    calculate_cophenetic_corr_matrix,
    calculate_ddps,
    calculate_ddp_structural_breaks,
    compare_cophenetic_corr,
)

ROOT = Path(__file__).resolve().parents[1]


class _Compound:
    def __init__(self, name):
        self.name = name


class _Reaction:
    def __init__(self, name, substrates=(), products=()):
        self.name = name
        self.substrates = [_Compound(value) for value in substrates]
        self.products = [_Compound(value) for value in products]


def _fake_kegg_dataset(keep_isolates=False):
    dataset = KeggKGMLMetabolicDataset.__new__(KeggKGMLMetabolicDataset)
    dataset.keep_isolates = keep_isolates
    dataset.rxns = [
        _Reaction("rn:R1", substrates=["cpd:A"], products=["cpd:B"]),
        _Reaction("rn:R2", substrates=["cpd:B"], products=["cpd:C"]),
        _Reaction("rn:R3", substrates=["cpd:X"], products=["cpd:Y"]),
        _Reaction("rn:R1", substrates=["cpd:B"], products=["cpd:D"]),
        _Reaction("rn:R4", substrates=["cpd:Q"], products=["cpd:R"]),
    ]
    dataset.rxn_genes = {
        "rn:R1": ["GeneA"],
        "rn:R2": ["GeneB"],
        "rn:R3": ["GeneC"],
        "rn:R4": ["GeneD"],
    }
    return dataset


def _small_adata(var_names=("GeneA", "GeneB", "GeneD", "Other")):
    return ad.AnnData(np.ones((2, len(var_names))), var=pd.DataFrame(index=list(var_names)))


def _small_decoder(n_cat_list=None):
    genes = ["g_met_1", "g_met_2", "g_back_1", "g_back_2"]
    rxns = {0: "rn:R1", 1: "rn:R2"}
    rxn_to_genes = {"rn:R1": ["g_met_1"], "rn:R2": ["g_met_2"]}
    metabolic_genes = pd.Series(
        ["Metabolic", "Metabolic", "Non-metabolic", "Non-metabolic"], index=genes
    )
    return DecoderMERN(
        genes=genes,
        rxns=rxns,
        background_n_input=2,
        rxns_to_genes=rxn_to_genes,
        metabolic_genes=metabolic_genes,
        n_cat_list=n_cat_list,
        strict_met_back_separation=True,
        rxn_genes_bias=True,
    )


def _one_model_batch(model, batch_size=8):
    data_loader = model._make_data_loader(
        model.adata, indices=np.arange(model.adata.n_obs), batch_size=batch_size
    )
    return next(iter(data_loader))


def _model_outputs(model, batch):
    model.module.eval()
    with torch.no_grad():
        inference_outputs = model.module.inference(**model.module._get_inference_input(batch))
        generative_outputs = model.module.generative(
            **model.module._get_generative_input(batch, inference_outputs)
        )
    return inference_outputs, generative_outputs


def test_synthetic_kgml_graph_edges_self_loops_and_attributes():
    dataset = _fake_kegg_dataset()
    rna = _small_adata()

    graph = dataset.metabolic_topology(rna, self_loops=False)
    assert ("rn:R1", "rn:R2") in graph.edges
    assert ("rn:R2", "rn:R1") in graph.edges
    assert ("rn:R1", "rn:R3") not in graph.edges
    assert ("rn:R3", "rn:R1") not in graph.edges
    assert not list(nx.selfloop_edges(graph))

    graph_with_loops = dataset.metabolic_topology(rna, self_loops=True)
    assert sorted(nx.selfloop_edges(graph_with_loops)) == [(node, node) for node in graph.nodes]
    assert graph_with_loops.number_of_edges() == graph.number_of_edges() + graph.number_of_nodes()

    for _, _, edge_data in graph_with_loops.edges(data=True):
        assert edge_data["weight"] == 1.0
        assert edge_data["sign"] == 1


def test_graph_is_undirected_by_directed_symmetry(general_package_adata_graph_rxn_genes):
    _, graph, _ = general_package_adata_graph_rxn_genes
    non_self_edges = [(source, target) for source, target in graph.edges if source != target]
    undirected_pairs = {frozenset((source, target)) for source, target in non_self_edges}

    assert all((target, source) in graph.edges for source, target in non_self_edges)
    assert graph.number_of_edges() == len(non_self_edges)
    assert graph.number_of_edges() == 2 * len(undirected_pairs)
    assert graph.graph.get("directed_edge_count", graph.number_of_edges()) == 5582
    assert graph.graph.get("undirected_pair_count", len(undirected_pairs)) == 2791


def test_isolate_removal_and_retention_contract():
    rna = _small_adata()

    no_isolates = _fake_kegg_dataset(keep_isolates=False).metabolic_topology(rna)
    assert {"rn:R3", "rn:R4"}.isdisjoint(no_isolates.nodes)

    keep_isolates = _fake_kegg_dataset(keep_isolates=True).metabolic_topology(rna)
    assert "rn:R4" in keep_isolates.nodes
    assert "rn:R3" not in keep_isolates.nodes


def test_oxphos_reactions_are_added_with_expected_compounds():
    expected = {
        "rn:R11945": {
            "substrates": {"cpd:C00399", "cpd:C00004"},
            "products": {"cpd:C00390", "cpd:C00003"},
        },
        "rn:R13223": {
            "substrates": {"cpd:C00399", "cpd:C00042"},
            "products": {"cpd:C00390", "cpd:C00122"},
        },
        "rn:R13224": {
            "substrates": {"cpd:C15603", "cpd:C00125"},
            "products": {"cpd:C15602", "cpd:C00126"},
        },
        "rn:R02161": {
            "substrates": {"cpd:C00390", "cpd:C00125"},
            "products": {"cpd:C00399", "cpd:C00126"},
        },
        "rn:R00081": {"substrates": {"cpd:C00126"}, "products": {"cpd:C00125"}},
    }

    without_oxphos = KeggKGMLMetabolicDataset(species="mouse", capitalize=False, add_oxphos=False)
    with_oxphos = KeggKGMLMetabolicDataset(species="mouse", capitalize=False, add_oxphos=True)

    without_names = {reaction.name for reaction in without_oxphos.rxns}
    with_names = {reaction.name for reaction in with_oxphos.rxns}
    assert set(expected).isdisjoint(without_names)
    assert set(expected).issubset(with_names)
    assert set(expected) == with_names - without_names

    by_name = {reaction.name: reaction for reaction in with_oxphos.rxns}
    for reaction_name, compounds in expected.items():
        reaction = by_name[reaction_name]
        assert {compound.name for compound in reaction.substrates} == compounds["substrates"]
        assert {compound.name for compound in reaction.products} == compounds["products"]


def test_get_rxn_genes_filters_to_anndata_var_names_and_module_labels():
    dataset = _fake_kegg_dataset()
    dataset.rxns = dataset.rxns[:3]
    dataset.rxn_genes = {
        "rn:R1": ["GeneA", "OutsideA"],
        "rn:R2": ["GeneB", "OutsideB"],
        "rn:R3": ["GeneC"],
    }
    rna = _small_adata(("GeneA", "GeneB", "GeneC", "Other"))

    rxn_genes = dataset.get_rxn_genes(rna, subset_genes=True)
    assert rxn_genes["rn:R1"] == ["GeneA"]
    assert rxn_genes["rn:R2"] == ["GeneB"]
    assert rxn_genes["rn:R3"] == ["GeneC"]
    assert dataset.get_rxn_genes(rna, subset_genes=False)["rn:R1"] == ["GeneA", "OutsideA"]

    dataset.add_module_info(rna)
    assert rna.var.loc["GeneA", "Metabolic Gene"] == "Metabolic"
    assert rna.var.loc["GeneB", "Metabolic Gene"] == "Metabolic"
    assert rna.var.loc["Other", "Metabolic Gene"] == "Non-metabolic"


def test_decoder_reaction_activity_is_relu_dot_product():
    decoder = _small_decoder()
    m = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    v = torch.tensor([[3.0, -4.0], [2.0, 1.0]])
    b = torch.zeros(2, 2)
    metabolic_library = torch.zeros(2, 1)
    background_library = torch.zeros(2, 1)

    _, _, enzyme_activity = decoder("gene", m, b, v, metabolic_library, background_library)
    assert torch.allclose(enzyme_activity, (m @ v.t()).clamp(min=0))
    assert (m @ v.t()).min() < 0


def test_decoder_strict_separation_and_library_size_scaling():
    decoder = _small_decoder()
    with torch.no_grad():
        decoder.rxn_gene_layer.linear.weight.zero_()
        decoder.rxn_gene_layer.linear.bias.zero_()
        decoder.rxn_gene_layer.linear.weight[:2, :2] = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        decoder.back_linear.weight.zero_()
        decoder.back_linear.bias[:] = torch.tensor([10.0, 10.0, 0.0, 1.0])

    m = torch.tensor([[1.0, 2.0]])
    v = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    b = torch.zeros(1, 2)
    metabolic_count_sum = torch.tensor([[6.0]])
    background_count_sum = torch.tensor([[8.0]])
    (met_scale, _, met_rate), (back_scale, _, back_rate), _ = decoder(
        "gene",
        m,
        b,
        v,
        metabolic_count_sum.log(),
        background_count_sum.log(),
    )

    is_metabolic = torch.tensor(decoder.gene_is_metabolic)
    assert torch.all(met_scale[:, ~is_metabolic] == 0)
    assert torch.all(back_scale[:, is_metabolic] == 0)
    assert torch.allclose(met_scale[:, is_metabolic].sum(dim=1), torch.ones(1))
    assert torch.allclose(back_scale[:, ~is_metabolic].sum(dim=1), torch.ones(1))
    assert torch.allclose(met_rate, metabolic_count_sum * met_scale)
    assert torch.allclose(back_rate, background_count_sum * back_scale)
    assert torch.allclose(
        met_rate + back_rate, metabolic_count_sum * met_scale + background_count_sum * back_scale
    )


def test_generative_negative_binomial_parameterization(general_package_model):
    batch = _one_model_batch(general_package_model)
    _, generative_outputs = _model_outputs(general_package_model, batch)

    px = generative_outputs[MODULE_KEYS.PX_KEY]
    px_met = generative_outputs[MODULE_KEYS.PX_MET_KEY]
    px_back = generative_outputs[MODULE_KEYS.PX_BACK_KEY]
    assert isinstance(px, NegativeBinomial)
    assert isinstance(px_met, NegativeBinomial)
    assert isinstance(px_back, NegativeBinomial)
    assert torch.allclose(px.mu, px_met.mu + px_back.mu)
    assert torch.allclose(px.theta, general_package_model.module.px_r.exp().expand_as(px.theta))
    assert px.theta.shape == px.mu.shape


def test_graph_decoder_is_parameter_free_signed_dot_product():
    decoder = GraphDecoder()
    v = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5]])
    eidx = torch.tensor([[0, 1, 2], [1, 2, 0]])
    esgn = torch.tensor([1.0, -1.0, 1.0])

    logits = decoder(v, eidx, esgn)
    expected = esgn * torch.tensor([1.0, -6.5, -1.0])
    assert torch.allclose(logits, expected)
    assert list(decoder.parameters()) == []


def test_graph_dataloader_refreshes_negative_samples_and_sets_attrs():
    graph = Data(
        edge_index=torch.tensor([[0, 7], [7, 0]]),
        edge_attr=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        num_nodes=8,
    )
    loader = GraphDataLoader(graph, length=1, neg_sampling_ratio=4, undirected=False)
    positive_count = graph.edge_index.shape[1]

    torch.manual_seed(11)
    first = next(iter(loader))
    torch.manual_seed(23)
    second = next(iter(loader))

    assert not torch.equal(
        first[GRAPH_REGISTRY_KEYS.EIDX_KEY][:, positive_count:],
        second[GRAPH_REGISTRY_KEYS.EIDX_KEY][:, positive_count:],
    )
    assert torch.all(first[GRAPH_REGISTRY_KEYS.EWT_KEY][:positive_count] == 1)
    assert torch.all(first[GRAPH_REGISTRY_KEYS.EWT_KEY][positive_count:] == 0)
    assert torch.all(first[GRAPH_REGISTRY_KEYS.ESGN_KEY] == 1)


def test_variational_posteriors_are_diagonal_normals(general_package_model):
    batch = _one_model_batch(general_package_model)
    inference_outputs, _ = _model_outputs(general_package_model, batch)

    for key in [MODULE_KEYS.QM_KEY, MODULE_KEYS.QB_KEY, MODULE_KEYS.QV_KEY]:
        dist = inference_outputs[key]
        assert isinstance(dist, Normal)
        assert dist.loc.shape == dist.scale.shape
        assert torch.all(dist.scale > 0)


def test_priors_are_standard_normal_unless_positive_met_dims(general_package_model):
    batch = _one_model_batch(general_package_model)
    inference_outputs, generative_outputs = _model_outputs(general_package_model, batch)

    for key, latent_key in [
        (MODULE_KEYS.PM_KEY, MODULE_KEYS.M_KEY),
        (MODULE_KEYS.PB_KEY, MODULE_KEYS.B_KEY),
        (MODULE_KEYS.PV_KEY, MODULE_KEYS.V_KEY),
    ]:
        prior = generative_outputs[key]
        assert torch.allclose(prior.loc, torch.zeros_like(inference_outputs[latent_key]))
        assert torch.allclose(prior.scale, torch.ones_like(inference_outputs[latent_key]))

    general_package_model.module.positive_met_dims = True
    positive_outputs = general_package_model.module.generative(
        **general_package_model.module._get_generative_input(batch, inference_outputs)
    )
    general_package_model.module.positive_met_dims = False

    assert torch.allclose(
        positive_outputs[MODULE_KEYS.PM_KEY].loc,
        torch.ones_like(inference_outputs[MODULE_KEYS.M_KEY]) * 5,
    )
    assert torch.allclose(
        positive_outputs[MODULE_KEYS.PV_KEY].loc,
        torch.ones_like(inference_outputs[MODULE_KEYS.V_KEY]) * 5,
    )
    assert torch.allclose(
        positive_outputs[MODULE_KEYS.PB_KEY].loc,
        torch.zeros_like(inference_outputs[MODULE_KEYS.B_KEY]),
    )


def test_loss_normalization_is_methods_contract(general_package_model):
    batch = _one_model_batch(general_package_model)
    inference_outputs, generative_outputs = _model_outputs(general_package_model, batch)
    loss_output = general_package_model.module.loss(batch, inference_outputs, generative_outputs)
    x = batch["cells"][REGISTRY_KEYS.X_KEY]
    v = inference_outputs[MODULE_KEYS.V_KEY]
    m = inference_outputs[MODULE_KEYS.M_KEY]
    b = inference_outputs[MODULE_KEYS.B_KEY]

    expected_x_nll = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(dim=-1) / x.shape[1]
    expected_kl_m = (
        kl_divergence(
            inference_outputs[MODULE_KEYS.QM_KEY], generative_outputs[MODULE_KEYS.PM_KEY]
        ).sum(dim=-1)
        / m.shape[1]
    )
    expected_kl_b = (
        kl_divergence(
            inference_outputs[MODULE_KEYS.QB_KEY], generative_outputs[MODULE_KEYS.PB_KEY]
        ).sum(dim=-1)
        / b.shape[1]
    )
    expected_kl_v = (
        kl_divergence(
            inference_outputs[MODULE_KEYS.QV_KEY], generative_outputs[MODULE_KEYS.PV_KEY]
        ).sum(dim=-1)
        / v.shape[1]
    ).mean()
    expected_graph_nll = (
        -generative_outputs[MODULE_KEYS.PG_KEY]
        .log_prob(batch["graph"][GRAPH_REGISTRY_KEYS.EWT_KEY])
        .mean()
    )

    assert torch.allclose(loss_output.reconstruction_loss["reconstruction_loss"], expected_x_nll)
    assert torch.allclose(loss_output.kl_local[MODULE_KEYS.KL_M_KEY], expected_kl_m)
    assert torch.allclose(loss_output.kl_local[MODULE_KEYS.KL_B_KEY], expected_kl_b)
    assert torch.allclose(loss_output.extra_metrics["kl_v"], expected_kl_v)
    assert torch.allclose(loss_output.extra_metrics["g_nll"], expected_graph_nll)


def test_train_defaults_use_90_10_split_batch_128_and_rmsprop(general_package_model, monkeypatch):
    captured = {}

    class CapturingSplitter:
        def __init__(
            self, *args, train_size=None, validation_size=None, batch_size=None, **kwargs
        ):
            captured["splitter_train_size_arg"] = train_size
            captured["effective_train_size"] = 0.9 if train_size is None else train_size
            captured["validation_size"] = validation_size
            captured["batch_size"] = batch_size

    class CapturingPlan:
        def __init__(self, module, **kwargs):
            captured["plan_kwargs"] = kwargs

    class CapturingRunner:
        def __init__(
            self, *args, data_splitter=None, training_plan=None, max_epochs=None, **kwargs
        ):
            captured["runner_max_epochs"] = max_epochs

        def __call__(self):
            return "trained"

    monkeypatch.setattr(general_package_model, "_data_splitter_cls", CapturingSplitter)
    monkeypatch.setattr(general_package_model, "_training_plan_cls", CapturingPlan)
    monkeypatch.setattr(general_package_model, "_train_runner_cls", CapturingRunner)

    assert general_package_model.train(max_epochs=1) == "trained"
    assert captured["splitter_train_size_arg"] is None
    assert captured["effective_train_size"] == 0.9
    assert captured["validation_size"] is None
    assert captured["batch_size"] == 128
    assert captured["plan_kwargs"]["optimizer"] == "Custom"
    assert captured["plan_kwargs"]["optimizer_creator"] is create_rmsprop_optimizer


def test_compact_repro_wrapper_training_defaults_match_script():
    tree = ast.parse((ROOT / "scripts" / "run_kl_search_args_parity.py").read_text())
    train_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    ]
    assert len(train_calls) == 1
    train_kwargs = {}
    for keyword in train_calls[0].keywords:
        try:
            train_kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            continue

    assert train_kwargs["train_size"] == 0.8
    assert train_kwargs["validation_size"] == 0.2
    assert train_kwargs["batch_size"] == 32


def test_batch_one_hot_covariates_reach_both_decoder_branches():
    decoder = _small_decoder(n_cat_list=[2])
    captured = {}

    def capture_rxn_gene_layer(module, inputs):
        captured["rxn_gene_input"] = inputs[0].detach().clone()

    def capture_back_linear(module, inputs):
        captured["back_linear_input"] = inputs[0].detach().clone()

    decoder.rxn_gene_layer.register_forward_pre_hook(capture_rxn_gene_layer)
    decoder.back_linear.register_forward_pre_hook(capture_back_linear)

    decoder(
        "gene",
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.eye(2),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([[0], [1]]),
    )

    assert torch.equal(captured["rxn_gene_input"][:, -2:], torch.eye(2, dtype=torch.long))
    assert torch.equal(captured["back_linear_input"][:, -2:], torch.eye(2, dtype=torch.long))


def test_get_decoding_documents_missing_multi_sample_api(general_package_model):
    with pytest.raises(NotImplementedError, match="n_samples > 1"):
        general_package_model.get_decoding(indices=[0, 1], batch_size=2, n_samples=8)


def test_get_latent_representation_give_mean_returns_posterior_means(general_package_model):
    general_package_model.is_trained_ = True
    general_package_model.module.eval()
    indices = np.arange(5)

    metabolic, background, graph = general_package_model.get_latent_representation(
        indices=indices, batch_size=len(indices), give_mean=True
    )
    metabolic_mean, _, background_mean, _, graph_mean, _ = (
        general_package_model.get_latent_representation(
            indices=indices, batch_size=len(indices), return_dist=True
        )
    )

    assert np.allclose(metabolic, metabolic_mean)
    assert np.allclose(background, background_mean)
    assert np.allclose(graph, graph_mean)


def test_module_can_generate_priors_without_model_wrapper():
    metabolic_genes = pd.Series(["Metabolic", "Non-metabolic"], index=["g1", "g2"])
    module = MERNModule(
        genes=["g1", "g2"],
        vertices={0: "rn:R1"},
        rxn_to_genes={"rn:R1": ["g1"]},
        metabolic_genes=metabolic_genes,
        n_input=2,
        n_batch=1,
        n_hidden=8,
        n_metabolic_dim=2,
        n_background_dim=2,
        dropout_rate=0.0,
        gene_likelihood="nb",
        positive_met_dims=False,
    )
    outputs = module.generative(
        m=torch.zeros(1, 2),
        b=torch.zeros(1, 2),
        v=torch.zeros(1, 2),
        eidx=torch.tensor([[0], [0]]),
        ewt=torch.ones(1),
        esgn=torch.ones(1),
        metabolic_library=torch.zeros(1, 1),
        background_library=torch.zeros(1, 1),
        batch_index=torch.zeros(1, 1, dtype=torch.long),
    )
    assert torch.allclose(outputs[MODULE_KEYS.PM_KEY].loc, torch.zeros(1, 2))
    assert torch.allclose(outputs[MODULE_KEYS.PB_KEY].scale, torch.ones(1, 2))


def test_calculate_ddps_returns_full_linkage_above_ddp_threshold():
    reactions = ["r1", "r2", "r3"]
    graph = nx.Graph()
    graph.add_edges_from([("r1", "r2"), ("r2", "r3")])
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ],
        index=reactions,
        columns=reactions,
    )

    rxn_to_ddp, ddp_rxns, kept, linkage_matrix = calculate_ddps(
        graph,
        corr,
        min_corr=0.8,
        min_size=2,
    )

    assert kept == [["r1", "r2"]]
    assert ddp_rxns == {"ddp_0": ["r1", "r2"]}
    assert rxn_to_ddp.to_dict() == {"r1": "ddp_0", "r2": "ddp_0", "r3": None}
    assert linkage_matrix.shape == (len(reactions) - 1, 4)
    assert np.allclose(linkage_matrix[:, 2], [0.1, 0.7])


def test_calculate_cophenetic_corr_matrix_returns_reaction_matrix():
    labels = ["r1", "r2", "r3"]
    linkage_matrix = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [3.0, 2.0, 0.7, 3.0],
    ])

    cophenetic_corr = calculate_cophenetic_corr_matrix(linkage_matrix, labels)

    assert list(cophenetic_corr.index) == labels
    assert list(cophenetic_corr.columns) == labels
    assert np.allclose(np.diag(cophenetic_corr), 1.0)
    assert np.isclose(cophenetic_corr.loc["r1", "r2"], 0.9)
    assert np.isclose(cophenetic_corr.loc["r1", "r3"], 0.3)
    assert np.isclose(cophenetic_corr.loc["r2", "r3"], 0.3)


def test_compare_cophenetic_corr_returns_graph_edge_deltas():
    labels = ["r1", "r2", "r3"]
    graph = nx.Graph()
    graph.add_edges_from([("r1", "r2"), ("r2", "r3"), ("r3", "outside")])
    wt_linkage = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [3.0, 2.0, 0.7, 3.0],
    ])
    ko_linkage = np.array([
        [1.0, 2.0, 0.4, 2.0],
        [3.0, 0.0, 0.8, 3.0],
    ])

    comparison = compare_cophenetic_corr(wt_linkage, ko_linkage, labels, graph=graph)

    assert comparison["rxn_1"].tolist() == ["r1", "r2"]
    assert comparison["rxn_2"].tolist() == ["r2", "r3"]
    assert np.allclose(comparison["wt_cophenetic_corr"], [0.9, 0.3])
    assert np.allclose(comparison["ko_cophenetic_corr"], [0.2, 0.6])
    assert np.allclose(comparison["delta_wt_minus_ko"], [0.7, -0.3])


def test_calculate_ddp_structural_breaks_scores_wt_ddp_pairs():
    labels = ["r1", "r2", "r3"]
    wt_corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0],
        ],
        index=labels,
        columns=labels,
    )
    ko_corr = pd.DataFrame(
        [
            [1.0, 0.5, 0.4],
            [0.5, 1.0, 0.1],
            [0.4, 0.1, 1.0],
        ],
        index=labels,
        columns=labels,
    )
    wt_linkage = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [3.0, 2.0, 0.3, 3.0],
    ])
    ko_linkage = np.array([
        [1.0, 2.0, 0.6, 2.0],
        [3.0, 0.0, 0.8, 3.0],
    ])
    rxn_to_ddp = pd.Series({"r1": "ddp_0", "r2": "ddp_0", "r3": "ddp_0"})

    breaks = calculate_ddp_structural_breaks(
        rxn_to_ddp,
        wt_corr,
        wt_linkage,
        ko_corr,
        ko_linkage,
        labels=labels,
    )

    row = breaks.iloc[0]
    assert row["rank"] == 1
    assert row["ddp"] == "ddp_0"
    assert row["n_reactions"] == 3
    assert row["n_pairs"] == 3
    assert np.isclose(row["wt_mean_corr"], 0.8)
    assert np.isclose(row["ko_mean_corr"], 1 / 3)
    assert np.isclose(row["mean_corr_drop"], 0.8 - 1 / 3)
    assert np.isclose(row["wt_min_corr"], 0.7)
    assert np.isclose(row["ko_min_corr"], 0.1)
    assert np.isclose(row["min_corr_drop"], 0.6)
    assert row["largest_corr_drop_pair"] == ("r2", "r3")
    assert np.isclose(row["wt_mean_cophenetic_corr"], (0.9 + 0.7 + 0.7) / 3)
    assert np.isclose(row["ko_mean_cophenetic_corr"], (0.2 + 0.2 + 0.4) / 3)
    assert np.isclose(row["mean_cophenetic_drop"], 0.5)
    assert np.isclose(row["wt_min_cophenetic_corr"], 0.7)
    assert np.isclose(row["ko_min_cophenetic_corr"], 0.2)
    assert np.isclose(row["min_cophenetic_drop"], 0.5)
    assert row["largest_cophenetic_drop_pair"] == ("r1", "r2")


def test_calculate_ddp_structural_breaks_filters_weak_wt_ddps():
    labels = ["r1", "r2", "r3"]
    corr = pd.DataFrame(np.eye(3), index=labels, columns=labels)
    linkage = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [3.0, 2.0, 0.3, 3.0],
    ])

    breaks = calculate_ddp_structural_breaks(
        {"ddp_0": labels},
        corr,
        linkage,
        corr,
        linkage,
        labels=labels,
        min_wt_cophenetic_corr=0.75,
    )

    assert breaks.empty
    assert "largest_corr_drop_pair" in breaks.columns
    assert "largest_cophenetic_drop_pair" in breaks.columns
