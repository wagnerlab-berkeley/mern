import networkx as nx


def test_mouse_intestine_fixture_shape_and_annotations(mouse_intestine_100):
    assert mouse_intestine_100.shape == (100, 12963)
    assert "counts" in mouse_intestine_100.layers
    assert "X_umap" in mouse_intestine_100.obsm
    assert {"Gene Reactions", "Metabolic Gene"}.issubset(mouse_intestine_100.var.columns)
    assert {"highly_variable_metabolic", "highly_variable_background"}.issubset(
        mouse_intestine_100.var.columns
    )
    assert (mouse_intestine_100.var["Metabolic Gene"] == "Metabolic").sum() > 0


def test_mouse_intestine_mern_inputs(mouse_intestine_100_mern_inputs):
    graph = mouse_intestine_100_mern_inputs["graph"]
    rxn_to_genes = mouse_intestine_100_mern_inputs["rxn_to_genes"]

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert isinstance(rxn_to_genes, dict)
    assert set(graph.nodes()).issubset(rxn_to_genes)

    edge_data = next(iter(graph.edges(data=True)))[2]
    assert edge_data["weight"] == 1.0
    assert edge_data["sign"] == 1
