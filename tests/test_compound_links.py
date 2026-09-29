from io import StringIO
from types import SimpleNamespace

import networkx as nx
import pytest
from Bio.KEGG.KGML import KGML_parser

from mern.support import KeggKGMLMetabolicDataset
from mern.support import _metabolic_datasets as datasets
from mern.support import _plots as plots


@pytest.fixture(params=["mouse", "human"])
def dataset(request, monkeypatch, tmp_path):
    monkeypatch.setattr(datasets.config, "KEGG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(KeggKGMLMetabolicDataset, "get_compound_info", lambda self: {})

    def no_download(*args, **kwargs):
        raise AssertionError("bundled compound links must load offline")

    monkeypatch.setattr(datasets.REST, "kegg_get", no_download)
    monkeypatch.setattr(datasets.REST, "kegg_link", no_download)
    return KeggKGMLMetabolicDataset(request.param)


def test_compound_links_restore_topology_without_changing_training(dataset):
    expected = {
        ("rn:R01655", "cpd:C00234", "cpd:C00445"),
        ("rn:R02101", "cpd:C00365", "cpd:C00364"),
        ("rn:R03720", "cpd:C01794", "cpd:C05122"),
        ("rn:R07770", "cpd:C16241", "cpd:C16238"),
    }
    original = {
        (name, sub, prod)
        for name, data in dataset._bundled_graph.nodes(data=True)
        for sub in data["substrates"]
        for prod in data["products"]
    }
    loaded = {
        (reaction.name, sub.name, prod.name)
        for reaction in dataset.rxns
        for sub in reaction.substrates
        for prod in reaction.products
    }
    # Exact equality also rejects unintended substrate/product cross-pairs.
    assert loaded == original | expected
    directed = dataset.compound_metabolic_topology(directed=True)
    for reaction, sub, prod in expected:
        assert directed.edges[sub, prod]["reaction"] == reaction
    compound = dataset.compound_metabolic_topology()
    assert (len(compound), compound.number_of_edges()) == {
        "mouse": (1211, 1334),
        "human": (1236, 1353),
    }[dataset.species]

    graph = dataset.metabolic_topology(None)
    bundled = dataset._bundled_graph.copy()
    bundled.remove_nodes_from(list(nx.isolates(bundled)))
    assert list(graph.nodes(data=True)) == list(bundled.nodes(data=True))
    assert list(graph.edges) == list(bundled.edges)
    assert "rn:R07770" not in graph


def test_custom_plot_includes_duplicate_and_isolated_reaction_links(dataset, monkeypatch):
    coordinates = {name: (i, i) for i, name in enumerate(sorted(dataset.all_compounds))}
    monkeypatch.setattr(plots, "graphviz_layout", lambda graph: coordinates)
    figure = plots.custom_pathway_plot(["rn:R02101", "rn:R07770"], dataset, rna=None)
    edges = [trace for trace in figure.data if trace.mode == "lines"]
    assert len(edges) == 3
    assert any(
        set(trace.x) - {None} == {coordinates[c][0] for c in ("cpd:C00365", "cpd:C00364")}
        for trace in edges
    )
    assert any(
        set(trace.x) - {None} == {coordinates[c][0] for c in ("cpd:C16241", "cpd:C16238")}
        for trace in edges
    )


@pytest.mark.parametrize(
    "kgml_name,in_pathway",
    [
        ("rn:R07770", True),
        ("rn:R00001 rn:R07770 rn:R00002", True),
        ("rn:R00001", False),
        ("rn:R077700", False),
    ],
)
def test_kegg_plot_filters_extra_only_reaction_using_pathway_kgml(
    dataset, monkeypatch, kgml_name, in_pathway
):
    coordinates = {name: (i, i) for i, name in enumerate(sorted(dataset.all_compounds))}
    monkeypatch.setattr(
        dataset,
        "_get_kgml_entry",
        lambda name, pathway: SimpleNamespace(
            graphics=[SimpleNamespace(x=coordinates[name][0], y=coordinates[name][1])]
        ),
    )
    # The same cached pathway supplies membership and coordinates; no extra metadata file.
    dataset.pathway_kgmls["rn00120"] = KGML_parser.read(
        StringIO(
            '<pathway name="path:rn00120" org="rn" number="00120">'
            f'<entry id="1" name="{kgml_name}" type="ortholog"/>'
            f'<reaction id="1" name="{kgml_name}" type="irreversible"/>'
            '</pathway>'
        )
    )
    figure = plots.kegg_pathway_plot("rn00120", dataset, rna=None)
    edges = [trace for trace in figure.data if trace.mode == "lines"]
    isolated_pair = {coordinates[c][0] for c in ("cpd:C16241", "cpd:C16238")}
    assert any(set(trace.x) - {None} == isolated_pair for trace in edges) == in_pathway
