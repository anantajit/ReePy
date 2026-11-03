from itertools import combinations
from math import dist

import networkx as nx
from reepy import ReebGraph


def graph_connectivity(points, epsilon):
    groups = {}
    for point in points:
        groups.setdefault(point["graph"], {})[point["id"]] = point["weight"]
    groups = list(groups.values())
    return all(group.keys() == groups[0].keys() for group in groups) and all(
        sum(abs(a[node] - b[node]) for node in a) / len(a) < epsilon
        for a, b in combinations(groups, 2)
    )


def graph_sequence(graphs, epsilon, store_points=False):
    graphs = list(graphs)
    builder = ReebGraph(
        level=lambda point: point["depth"],
        connectivity=lambda points: graph_connectivity(points, epsilon),
        continuity=lambda a, b: (
            a["graph"] == b["graph"] and graphs[a["graph"]].has_edge(a["id"], b["id"])
        ),
        store_points=store_points,
    )
    for graph_id, graph in enumerate(graphs):
        for node, attributes in graph.nodes(data=True):
            builder.add({**attributes, "id": node, "graph": graph_id})
    return builder.to_networkx()


def activation_graph(graph, epsilon=0.5, store_points=False):
    builder = ReebGraph(
        level=lambda point: graph.nodes[point]["depth"],
        connectivity=lambda points: all(
            dist(graph.nodes[a]["activation"], graph.nodes[b]["activation"]) <= epsilon
            for a, b in combinations(points, 2)
        ),
        continuity=graph.has_edge,
        store_points=store_points,
    )
    for node in graph:
        builder.add(node)
    return builder.to_networkx()


def test_y_graphs():
    graphs = []
    for weights in [(0, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0)]:
        graph = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
        for node, (depth, weight) in enumerate(zip((0, 0, 1, 2), weights)):
            graph.nodes[node].update(depth=depth, weight=weight)
        graphs.append(graph)
    result = graph_sequence(graphs, epsilon=0.51, store_points=True)
    layers = {}
    for node, data in result.nodes(data=True):
        layers.setdefault(data["level"], []).append(node)
    assert result.number_of_nodes() == 4
    assert result.number_of_edges() == 3
    assert {depth: len(nodes) for depth, nodes in layers.items()} == {0: 2, 1: 1, 2: 1}
    junction, end = layers[1][0], layers[2][0]
    assert all(result.has_edge(node, junction) for node in layers[0])
    assert result.has_edge(junction, end)
    assert {
        frozenset(point["graph"] for point in result.nodes[node]["points"])
        for node in layers[0]
    } == {frozenset({0, 2}), frozenset({1})}
    assert all(
        "graph" not in data for graph in graphs for _, data in graph.nodes(data=True)
    )


def test_graph_connectivity_matches_ids_and_averages_weight_differences():
    points = [
        {"graph": 0, "id": "a", "weight": 0.0},
        {"graph": 0, "id": "b", "weight": 0.4},
        {"graph": 1, "id": "b", "weight": 0.4},
        {"graph": 1, "id": "a", "weight": 0.15},
    ]
    assert graph_connectivity(points, epsilon=0.1)
    assert not graph_connectivity(points, epsilon=0.05)
    assert not graph_connectivity(points[:-1], epsilon=0.1)


def test_graph_connectivity_checks_every_pair():
    points = [
        {"graph": graph, "id": "a", "weight": weight}
        for graph, weight in enumerate((0.4, 0.2, 0.6))
    ]
    assert not graph_connectivity(points, epsilon=0.3)


def test_graph_with_class_activation_vectors():
    vectors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0.5, 0.5, 0),
        (0.4, 0.6, 0),
        (0, 0, 1),
        (0.3, 0.3, 0.4),
    ]
    graph = nx.DiGraph([(0, 3), (1, 4), (2, 5), (3, 6), (4, 6), (5, 6)])
    for node, (depth, vector) in enumerate(zip((0, 0, 0, 1, 1, 1, 2), vectors)):
        graph.nodes[node].update(depth=depth, activation=vector)
    result = activation_graph(graph, epsilon=0.5, store_points=True)
    groups = {frozenset(data["points"]): node for node, data in result.nodes(data=True)}
    assert result.number_of_nodes() == 5
    assert result.number_of_edges() == 4
    assert set(groups) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3, 4}),
        frozenset({6}),
    }
    assert set(result.edges()) == {
        (groups[frozenset(a)], groups[frozenset(b)])
        for a, b in [({0}, {3, 4}), ({1}, {3, 4}), ({3, 4}, {6}), ({2}, {6})]
    }
