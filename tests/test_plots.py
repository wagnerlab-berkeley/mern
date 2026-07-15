import networkx as nx
import pandas as pd
import pytest

from mern.support._plots import (
    _SpatialGeometryIndex,
    _ddp_overlays,
    _label_box_size,
    _rect_bounds,
    _segment_rect_distance,
)


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
