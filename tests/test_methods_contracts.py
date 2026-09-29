import ast
import copy
from io import StringIO
from pathlib import Path
import pickle

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
    dataset.add_oxphos = True
    dataset.keep_isolates = keep_isolates
    dataset.rebuild_from_kegg = False
    dataset.rxns = [
        _Reaction("rn:R1", substrates=["cpd:A"], products=["cpd:B"]),
        _Reaction("rn:R2", substrates=["cpd:B"], products=["cpd:C"]),
        _Reaction("rn:R3", substrates=["cpd:X"], products=["cpd:Y"]),
        _Reaction("rn:R1", substrates=["cpd:B"], products=["cpd:D"]),
        _Reaction("rn:R4", substrates=["cpd:Q"], products=["cpd:R"]),
    ]
    dataset.kegg_rxns = dataset.rxns
    dataset.rxn_info = None
    dataset.rxn_genes = {
        "rn:R1": ["GeneA"],
        "rn:R2": ["GeneB"],
        "rn:R3": ["GeneC"],
        "rn:R4": ["GeneD"],
    }
    dataset._bundled_graph = nx.DiGraph(
        [
            ("rn:R1", "rn:R2"),
            ("rn:R2", "rn:R1"),
        ]
    )
    dataset._bundled_graph.add_nodes_from(["rn:R3", "rn:R4"])
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

    dataset.rebuild_from_kegg = True
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


@pytest.mark.parametrize(
    ("filename", "nodes", "edges", "isolates"),
    [
        ("mouse_metabolic_graph.pkl", 1249, 5582, 36),
        ("human_metabolic_graph.pkl", 1268, 5604, 43),
    ],
)
def test_bundled_kegg_graphs(filename, nodes, edges, isolates):
    path = ROOT / "src" / "mern" / "support" / "data" / "kegg" / filename
    with path.open("rb") as f:
        graph = pickle.load(f)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == nodes
    assert graph.number_of_edges() == edges
    assert len(list(nx.isolates(graph))) == isolates
    assert nx.number_of_selfloops(graph) == 0
    assert set(KeggKGMLMetabolicDataset.OXPHOS_RXNS).issubset(graph)
    assert all(data == {"weight": 1.0, "sign": 1} for _, _, data in graph.edges(data=True))


def test_kegg_cache_writes_are_atomic(tmp_path):
    text_path = tmp_path / "test.kgml"
    pickle_path = tmp_path / "test.pkl"

    KeggKGMLMetabolicDataset._atomic_write_text(text_path, "complete")
    KeggKGMLMetabolicDataset._atomic_pickle_dump({"complete": True}, pickle_path)

    assert text_path.read_text() == "complete"
    with pickle_path.open("rb") as f:
        assert pickle.load(f) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_kegg_cache_write_keeps_existing_file(tmp_path, monkeypatch):
    from mern.support import _metabolic_datasets

    target = tmp_path / "test.pkl"
    target.write_bytes(b"existing")

    def fail_after_partial_write(value, f, protocol):
        f.write(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(_metabolic_datasets.pickle, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="interrupted"):
        KeggKGMLMetabolicDataset._atomic_pickle_dump({"complete": False}, target)

    assert target.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.tmp"))


def test_reaction_metadata_download_uses_current_kegg_reactions(tmp_path, monkeypatch):
    from mern.support import _metabolic_datasets

    dataset = _fake_kegg_dataset()
    dataset.species = "mouse"
    dataset.data_dir = str(tmp_path)
    dataset.kegg_species = "mmu"
    dataset.kegg_rxns = [_Reaction("rn:R12658 rn:R13426")]

    requested = []

    class _Response:
        def read(self):
            return "reaction///"

    def fake_kegg_get(reactions):
        requested.extend(reactions.split("+"))
        return _Response()

    monkeypatch.setattr(_metabolic_datasets.REST, "kegg_get", fake_kegg_get)
    monkeypatch.setattr(dataset, "parse_rxn", lambda _: {"id": "R12658"})

    dataset.get_kgml_rxn_info()

    assert requested == ["rn:R12658", "rn:R13426"]


@pytest.mark.parametrize("rebuild_from_kegg", [False, True])
@pytest.mark.parametrize("capitalize", [False, True])
def test_reaction_gene_mapping_reuses_selected_file(
    tmp_path, monkeypatch, rebuild_from_kegg, capitalize
):
    from mern.support import _metabolic_datasets

    dataset = _fake_kegg_dataset()
    dataset.species = "mouse"
    dataset.kegg_species = "mmu"
    dataset.capitalize_genes = capitalize
    dataset.data_dir = str(tmp_path / "cache")
    dataset.package_data_dir = str(tmp_path / "package")
    files = {}
    for directory, gene in [
        (dataset.data_dir, "CachedGene"),
        (dataset.package_data_dir, "BundledGene"),
    ]:
        path = Path(directory) / "mmu_kgml_rxn_genes.pkl"
        path.parent.mkdir()
        files[path] = pickle.dumps({"rn:R1": [gene]})
        path.write_bytes(files[path])

    def fail_network(*args, **kwargs):
        raise AssertionError("existing reaction-gene mappings must not access the network")

    monkeypatch.setattr(_metabolic_datasets.REST, "kegg_link", fail_network)
    monkeypatch.setattr(_metabolic_datasets.mygene, "MyGeneInfo", fail_network)

    result = dataset.get_rxn_genes_all(rebuild_from_kegg=rebuild_from_kegg)

    gene = "CachedGene" if rebuild_from_kegg else "BundledGene"
    assert result == {"rn:R1": [gene.upper() if capitalize else gene]}
    assert all(path.read_bytes() == original for path, original in files.items())


def test_reaction_gene_mapping_download_is_cached(tmp_path, monkeypatch):
    from mern.support import _metabolic_datasets

    dataset = _fake_kegg_dataset()
    dataset.species = "mouse"
    dataset.kegg_species = "mmu"
    dataset.capitalize_genes = True
    dataset.data_dir = str(tmp_path / "cache")
    dataset.package_data_dir = str(tmp_path / "package")
    dataset.kegg_rxns = [_Reaction("rn:R1")]

    links = {
        ("enzyme", "reaction"): "rn:R1\tec:1.1.1.1\n",
        ("mmu", "enzyme"): "ec:1.1.1.1\tmmu:123\n",
    }
    monkeypatch.setattr(
        _metabolic_datasets.REST, "kegg_link", lambda *args: StringIO(links[args])
    )

    class FakeMyGene:
        def querymany(self, genes, species):
            assert genes == ["123"]
            assert species == "mouse"
            return [{"query": "123", "symbol": "GeneA"}]

    monkeypatch.setattr(_metabolic_datasets.mygene, "MyGeneInfo", FakeMyGene)
    assert dataset.get_rxn_genes_all(rebuild_from_kegg=True) == {"rn:R1": ["GENEA"]}

    cache_path = Path(dataset.data_dir) / "mmu_kgml_rxn_genes.pkl"
    cached_bytes = cache_path.read_bytes()
    assert pickle.loads(cached_bytes) == {"rn:R1": ["GeneA"]}
    assert not Path(dataset.package_data_dir).exists()

    def fail_network(*args, **kwargs):
        raise AssertionError("the downloaded mapping must be reused on the next call")

    monkeypatch.setattr(_metabolic_datasets.REST, "kegg_link", fail_network)
    monkeypatch.setattr(_metabolic_datasets.mygene, "MyGeneInfo", fail_network)
    dataset.capitalize_genes = False
    assert dataset.get_rxn_genes_all(rebuild_from_kegg=True) == {"rn:R1": ["GeneA"]}
    assert cache_path.read_bytes() == cached_bytes


def test_constructor_selects_current_kegg_data(tmp_path, monkeypatch):
    from mern.support import _metabolic_datasets

    current_reactions = [_Reaction("rn:R1", substrates=["cpd:A"], products=["cpd:B"])]
    current_info = {"R1": {"name": "Reaction 1"}}
    rebuild_gene_args = []

    monkeypatch.setattr(_metabolic_datasets.config, "KEGG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        KeggKGMLMetabolicDataset,
        "_load_or_download_kgml",
        lambda self: type("KGML", (), {"reactions": current_reactions})(),
    )
    monkeypatch.setattr(
        KeggKGMLMetabolicDataset, "get_kgml_rxn_info", lambda self: current_info
    )
    monkeypatch.setattr(
        KeggKGMLMetabolicDataset,
        "get_rxn_genes_all",
        lambda self, rebuild_from_kegg=False: rebuild_gene_args.append(rebuild_from_kegg) or {},
    )
    monkeypatch.setattr(KeggKGMLMetabolicDataset, "get_compound_info", lambda self: {})

    dataset = KeggKGMLMetabolicDataset(
        "mouse", add_oxphos=False, rebuild_from_kegg=True
    )

    assert dataset.rxns is dataset.kegg_rxns
    assert dataset.rxn_info is current_info
    assert rebuild_gene_args == [True]


def test_default_constructor_does_not_load_species_kgml(tmp_path, monkeypatch):
    from mern.support import _metabolic_datasets

    monkeypatch.setattr(_metabolic_datasets.config, "KEGG_CACHE_DIR", str(tmp_path))

    def fail_download(self):
        raise AssertionError("default construction should not load the species KGML")

    monkeypatch.setattr(KeggKGMLMetabolicDataset, "_load_or_download_kgml", fail_download)
    monkeypatch.setattr(
        KeggKGMLMetabolicDataset,
        "get_rxn_genes_all",
        lambda self, rebuild_from_kegg=False: {},
    )
    monkeypatch.setattr(KeggKGMLMetabolicDataset, "get_compound_info", lambda self: {})

    dataset = KeggKGMLMetabolicDataset("mouse")

    assert dataset.kgml is None
    assert dataset.kegg_rxns == []
    assert dataset.rxns is dataset._bundled_rxns


def test_bundled_reaction_annotations_do_not_download(mouse_kegg_dataset, monkeypatch):
    rna = _small_adata()
    graph = mouse_kegg_dataset.metabolic_topology(rna)

    def fail_download():
        raise AssertionError("bundled annotations should not download reaction metadata")

    monkeypatch.setattr(mouse_kegg_dataset, "get_kgml_rxn_info", fail_download)
    mouse_kegg_dataset.add_rxn_module_info(rna, graph)

    expected = {
        node: graph.nodes[node]["reaction_info"]
        for node in graph.nodes
    }
    assert rna.uns["Reaction Info"].to_dict("index") == expected


def test_live_reaction_annotations_download_on_demand(monkeypatch):
    dataset = _fake_kegg_dataset()
    graph = nx.DiGraph()
    graph.add_node("rn:R1")
    downloaded = {"R1": {"name": "Reaction 1"}}
    monkeypatch.setattr(dataset, "get_kgml_rxn_info", lambda: downloaded)

    rna = _small_adata()
    dataset.add_rxn_module_info(rna, graph)

    assert dataset.rxn_info is downloaded
    assert rna.uns["Reaction Info"].loc["rn:R1", "Rxn Name"] == "Reaction 1"


def test_isolate_removal_and_retention_contract():
    rna = _small_adata()

    no_isolates = _fake_kegg_dataset(keep_isolates=False).metabolic_topology(rna)
    assert {"rn:R3", "rn:R4"}.isdisjoint(no_isolates.nodes)

    keep_isolates = _fake_kegg_dataset(keep_isolates=True).metabolic_topology(rna)
    assert "rn:R4" in keep_isolates.nodes
    assert "rn:R3" not in keep_isolates.nodes


def test_oxphos_reactions_are_added_with_expected_compounds(mouse_kegg_dataset):
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

    for reaction, compounds in expected.items():
        assert set(KeggKGMLMetabolicDataset.OXPHOS_RXNS[reaction]["substrates"]) == compounds[
            "substrates"
        ]
        assert set(KeggKGMLMetabolicDataset.OXPHOS_RXNS[reaction]["products"]) == compounds[
            "products"
        ]

    rna = _small_adata()
    with_oxphos = mouse_kegg_dataset.metabolic_topology(rna)
    without_oxphos_dataset = copy.copy(mouse_kegg_dataset)
    without_oxphos_dataset.add_oxphos = False
    without_oxphos = without_oxphos_dataset.metabolic_topology(rna)

    assert set(expected).issubset(with_oxphos)
    assert set(expected).isdisjoint(without_oxphos)
    assert set(with_oxphos) - set(without_oxphos) == set(expected)


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
