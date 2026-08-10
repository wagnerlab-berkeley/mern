import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mern.support._plots as plots
from mern.support._plots import (
    _SpatialGeometryIndex,
    _ddp_overlays,
    _label_box_size,
    _rect_bounds,
    _segment_rect_distance,
    custom_pathway_plot,
)


class _Compound:
    def __init__(self, name):
        self.name = name


class _Reaction:
    def __init__(self, name, substrate, product):
        self.name = name
        self.substrates = [_Compound(substrate)]
        self.products = [_Compound(product)]


class _PathwayDataset:
    def __init__(self):
        self.rxns = [
            _Reaction("rn:R1", "cpd:C00001", "cpd:C00002"),
            _Reaction("rn:R2", "cpd:C00002", "cpd:C00003"),
        ]
        self.compound_info = {
            "C00001": {"name": "Water"},
            "C00002": {"name": "ATP"},
            "C00003": {"name": "ADP"},
        }


def test_custom_pathway_plot_accepts_publication_style_parameters(monkeypatch):
    monkeypatch.setattr(
        plots,
        "graphviz_layout",
        lambda graph: {
            "cpd:C00001": (0.0, 0.0),
            "cpd:C00002": (1.0, 0.0),
            "cpd:C00003": (2.0, 0.0),
        },
    )
    figure = custom_pathway_plot(
        ["rn:R1", "rn:R2"],
        _PathwayDataset(),
        rna=None,
        user_labels=pd.Series({"rn:R1": "DDP 1", "rn:R2": np.nan}),
        edge_width=2.5,
        unassigned_edge_width=1.0,
        unassigned_edge_color="#AAAAAA",
        node_size=4.2,
        node_outline_color="#666666",
        node_outline_width=0.4,
        node_label_font_family="Arial",
        node_label_font_size=9,
        node_label_color="#222222",
        show_legend=False,
        show_colorbars=False,
        layout={
            "title": None,
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    )

    edge_traces = [trace for trace in figure.data if trace.mode == "lines"]
    assert len(edge_traces) == 2
    assert edge_traces[0].line.width == pytest.approx(2.5)
    assert edge_traces[1].line.width == pytest.approx(1.0)
    assert edge_traces[1].line.color == "#AAAAAA"

    node_trace = next(trace for trace in figure.data if trace.mode == "markers+text")
    assert list(node_trace.marker.size) == pytest.approx([4.2, 4.2, 4.2])
    assert node_trace.marker.line.color == "#666666"
    assert node_trace.marker.line.width == pytest.approx(0.4)
    assert node_trace.textfont.family == "Arial"
    assert node_trace.textfont.size == pytest.approx(9)
    assert node_trace.textfont.color == "#222222"
    assert list(node_trace.text) == ["Water", "ATP", "ADP"]

    assert not any(trace.showlegend for trace in figure.data)
    assert not any(getattr(trace.marker, "showscale", False) for trace in figure.data)
    assert figure.layout.title.text is None
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_ddp_contour_label_box_does_not_overlap_reaction_segment():
    graph = nx.Graph()
    graph.add_edge("a", "b", reaction="rn:R1")
    pos = {"a": (0.0, 0.0), "b": (100.0, 0.0)}
    ddp_labels = pd.Series({"rn:R1": "ddp_0"})

    _, annotations, _ = _ddp_overlays(
        graph,
        pos,
        ddp_labels=ddp_labels,
        ddp_contour_pad=24.0,
        ddp_style="contours",
    )

    assert len(annotations) == 1
    assert annotations[0]["text"] == "ddp_0"
    assert abs(annotations[0]["x"] - 50.0) < 8.0
    assert abs(annotations[0]["y"]) > 10.0
    assert annotations[0]["xanchor"] == "center"
    label_bounds = _rect_bounds(
        (annotations[0]["x"], annotations[0]["y"]),
        _label_box_size(annotations[0]["text"]),
    )
    assert _segment_rect_distance(pos["a"], pos["b"], label_bounds) > 0.0


def test_ddp_halo_label_box_does_not_overlap_reaction_segment():
    graph = nx.Graph()
    graph.add_edge("a", "b", reaction="rn:R1")
    graph.add_edge("c", "d", reaction="rn:R2")
    pos = {
        "a": (0.0, 0.0),
        "b": (10.0, 0.0),
        "c": (100.0, 0.0),
        "d": (140.0, 0.0),
    }
    ddp_labels = pd.Series({"rn:R1": "ddp_0", "rn:R2": "ddp_0"})

    _, annotations, shapes = _ddp_overlays(
        graph,
        pos,
        ddp_labels=ddp_labels,
        ddp_contour_pad=10.0,
        ddp_style="halos",
    )

    assert len(shapes) == 1
    assert len(annotations) == 1
    assert annotations[0]["x"] == pytest.approx(120.0)
    assert abs(annotations[0]["y"]) > 10.0
    label_bounds = _rect_bounds(
        (annotations[0]["x"], annotations[0]["y"]),
        _label_box_size(annotations[0]["text"]),
    )
    assert _segment_rect_distance(pos["c"], pos["d"], label_bounds) > 0.0


def test_ddp_label_box_avoids_nearby_non_ddp_graph_edges():
    graph = nx.Graph()
    graph.add_edge("a", "b", reaction="rn:R1")
    graph.add_edge("c", "d", reaction="rn:R2")
    pos = {
        "a": (0.0, 0.0),
        "b": (100.0, 0.0),
        "c": (50.0, 16.0),
        "d": (120.0, 16.0),
    }
    ddp_labels = pd.Series({"rn:R1": "ddp_0"})

    _, annotations, _ = _ddp_overlays(
        graph,
        pos,
        ddp_labels=ddp_labels,
        ddp_contour_pad=24.0,
        ddp_style="contours",
    )

    label_bounds = _rect_bounds(
        (annotations[0]["x"], annotations[0]["y"]),
        _label_box_size(annotations[0]["text"]),
    )
    assert _segment_rect_distance(pos["a"], pos["b"], label_bounds) > 0.0
    assert _segment_rect_distance(pos["c"], pos["d"], label_bounds) > 0.0


def test_spatial_geometry_index_queries_only_nearby_geometry():
    node_points = [(5.0, 5.0), (500.0, 500.0)]
    edge_segments = [
        ((0.0, 0.0), (20.0, 0.0)),
        ((500.0, 500.0), (520.0, 500.0)),
    ]
    index = _SpatialGeometryIndex(node_points, edge_segments, cell_size=50.0)

    nearby_nodes, nearby_edges = index.query((0.0, 30.0, -10.0, 10.0), pad=5.0)

    assert nearby_nodes == [node_points[0]]
    assert nearby_edges == [edge_segments[0]]
