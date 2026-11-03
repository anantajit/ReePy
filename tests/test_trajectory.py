from itertools import combinations, pairwise
from math import dist

import pytest
from reepy import ReebGraph
from reepy.utils.clustering import Clustering, EuclideanClustering


class ExactClustering(Clustering):
    def __init__(self, cluster=None):
        super().__init__(
            cluster
            or (
                lambda points: [
                    [point for point in points if point == value]
                    for value in dict.fromkeys(points)
                ]
            )
        )

    def connectivity(self):
        return lambda points: len(set(points)) <= 1


def trajectory_graph(trajectories, epsilon, interpolate=True, store_points=False):
    samples = [
        (trajectory_id, index, time, *position)
        for trajectory_id, trajectory in enumerate(trajectories)
        for index, (time, position) in enumerate(trajectory)
    ]
    ties = {point: set() for point in samples}
    levels = sorted({point[2] for point in samples}) if interpolate else []
    for trajectory_id, trajectory in enumerate(trajectories):
        for index, ((start, a), (end, b)) in enumerate(pairwise(trajectory)):
            segment = (trajectory_id, index)
            ties[(trajectory_id, index, start, *a)].add(segment)
            ties[(trajectory_id, index + 1, end, *b)].add(segment)
            for time in levels:
                if start < time < end:
                    t = (time - start) / (end - start)
                    position = tuple(x + t * (y - x) for x, y in zip(a, b))
                    crossing = (trajectory_id, None, time, *position)
                    ties.setdefault(crossing, set()).add(segment)
                    samples.append(crossing)
    graph = ReebGraph(
        level=lambda point: point[2],
        connectivity=lambda points: all(
            dist(a[3:], b[3:]) < epsilon for a, b in combinations(points, 2)
        ),
        continuity=lambda a, b: not ties[a].isdisjoint(ties[b]),
        store_points=store_points,
    )
    for point in samples:
        graph.add(point)
    result = graph.to_networkx()
    if store_points:
        for _, data in result.nodes(data=True):
            data["points"] = [point for point in data["points"] if point[1] is not None]
    return result


def test_two_trajectories_split_and_rejoin():
    trajectories = [
        [(0, (0,)), (1, (0,)), (2, (1,)), (3, (0,)), (4, (0,))],
        [(0, (0,)), (1, (0,)), (2, (-1,)), (3, (0,)), (4, (0,))],
    ]
    graph = trajectory_graph(trajectories, epsilon=0.5, store_points=True)
    nodes = {data["level"]: node for node, data in graph.nodes(data=True)}

    assert graph.number_of_nodes() == 4
    assert set(nodes) == {0, 1, 3, 4}
    assert graph.number_of_edges() == 4
    assert graph.number_of_edges(nodes[0], nodes[1]) == 1
    assert graph.number_of_edges(nodes[1], nodes[3]) == 2
    assert graph.number_of_edges(nodes[3], nodes[4]) == 1
    for time, node in nodes.items():
        assert graph.nodes[node]["points"] == [
            (0, time, time, 0),
            (1, time, time, 0),
        ]


@pytest.mark.parametrize(
    "interpolate, edges", [(False, 3), (True, 4)], ids=["dropout", "interpolated"]
)
def test_trajectories_with_different_sampling_rates(interpolate, edges):
    trajectories = [
        [(0, (1,)), (1, (2,)), (2, (3,))],
        [(0, (1,)), (1, (2,)), (2, (1,))],
        [(0, (1,)), (2, (1,))],
    ]
    graph = trajectory_graph(
        trajectories, epsilon=0.5, interpolate=interpolate, store_points=True
    )
    nodes = {
        (data["level"], data["points"][0][3]): node
        for node, data in graph.nodes(data=True)
    }
    expected = {
        (nodes[0, 1], nodes[1, 2]),
        (nodes[1, 2], nodes[2, 1]),
        (nodes[1, 2], nodes[2, 3]),
    }
    if interpolate:
        expected.add((nodes[0, 1], nodes[2, 1]))
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == edges
    assert set(nodes) == {(0, 1), (1, 2), (2, 1), (2, 3)}
    assert set(graph.edges()) == expected
    assert {point[0] for point in graph.nodes[nodes[1, 2]]["points"]} == {0, 1}
    assert {point[0] for point in graph.nodes[nodes[2, 1]]["points"]} == {1, 2}
    assert all(
        point[1] is not None
        for _, data in graph.nodes(data=True)
        for point in data["points"]
    )


def test_sequential_trajectories_use_indices_and_preserve_branches():
    from reepy.trajectory import SequentialReebGraph

    graph = SequentialReebGraph(
        [[0, 0, 1, 0, 0], [0, 0, -1, 0, 0]],
        connectivity=lambda points: len(set(points)) == 1,
        store_points=True,
    )
    assert isinstance(graph, ReebGraph)
    nodes = {data["level"]: node for node, data in graph.nodes(data=True)}
    assert set(nodes) == {0, 1, 3, 4}
    assert graph.number_of_edges() == 4
    assert graph.number_of_edges(nodes[1], nodes[3]) == 2
    assert graph.nodes[nodes[0]]["points"] == [(0, 0, 0), (1, 0, 0)]
    assert graph.validate()


def test_sequential_trajectory_validation_and_add():
    from reepy.trajectory import SequentialReebGraph

    graph = SequentialReebGraph([[0, 1]], connectivity=lambda points: True)
    with pytest.raises(AssertionError, match="equal lengths"):
        graph.add_trajs([[0]])
    with pytest.raises(AssertionError, match="sized"):
        graph.add_trajs([iter([0, 1])])
    graph.add((1, 0, 0))
    graph.add((1, 1, 1))
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.nodes[0]["points"] == [(0, 0, 0), (1, 0, 0)]


def test_spatial_trajectories_cluster_before_following_paths():
    from reepy.trajectory import SpatialReebGraph

    calls = []

    def cluster(points):
        calls.append(points)
        return [
            [point for point in points if point == value]
            for value in dict.fromkeys(points)
        ]

    graph = SpatialReebGraph(
        ([0, 1, 2], iter([0, 3, 2])), ExactClustering(cluster), store_points=True
    )
    assert isinstance(graph, ReebGraph)
    assert calls == [[(0,), (1,), (2,), (0,), (3,), (2,)]]
    assert set(graph) == {0, 2}
    assert graph.number_of_edges(0, 2) == 2
    assert graph.nodes[0]["points"] == [(0, 0, 0, 0), (1, 0, 0, 0)]
    assert "level" not in graph.nodes[0]
    assert graph.to_networkx().number_of_edges(0, 2) == 2
    graph.add_trajs([iter([0, 4, 2])])
    assert graph.number_of_edges(0, 2) == 3
    assert len(calls) == 2
    assert graph.validate()


def test_spatial_trajectories_preserve_cycles_and_membership_changes():
    from reepy.trajectory import SpatialReebGraph

    cycle = SpatialReebGraph([[0, 1, 2, 0]], clustering=ExactClustering())
    assert cycle.number_of_nodes() == cycle.number_of_edges() == 1
    assert next(iter(cycle.edges()))[0] == next(iter(cycle.edges()))[1]
    graph = SpatialReebGraph([[0, 1, 2], [1, 2]], clustering=ExactClustering())
    assert set(graph) == {0, 1, 2}
    assert set(graph.edges()) == {(0, 1), (1, 2)}


def test_empty_trajectories_and_spatial_partition_validation():
    from reepy.trajectory import SequentialReebGraph, SpatialReebGraph

    assert len(SequentialReebGraph([], connectivity=lambda points: True)) == 0
    assert len(SpatialReebGraph([iter([])], clustering=ExactClustering())) == 0
    with pytest.raises(AssertionError, match="partition"):
        SpatialReebGraph(
            [[0, 1]], clustering=ExactClustering(lambda points: [[points[0]]])
        )


def test_spatial_euclidean_clustering_validates_collapsed_components():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph(
        [[(0,), (1,), (2,)], [(0.1,), (1.1,), (2.1,)]],
        EuclideanClustering(0.2),
    )
    assert graph.number_of_nodes() == 2
    assert len(graph._components[None]) == 3
    assert graph.validate()
    graph._components[None][1].append((99, 0, 0, 9))
    assert not graph.validate()


def test_sequential_batch_add_matches_individual_adds():
    from reepy.trajectory import SequentialReebGraph

    trajectories = [[0, 0, 1, 0, 0], [0, 0, -1, 0, 0]]
    batch = SequentialReebGraph([], connectivity=lambda points: len(set(points)) == 1)
    batch.add_trajs(iter(trajectories))
    single = SequentialReebGraph([], connectivity=lambda points: len(set(points)) == 1)
    for trajectory_id, trajectory in enumerate(trajectories):
        for index, point in enumerate(trajectory):
            single.add((trajectory_id, index, point))
    assert list(batch.nodes(data=True)) == list(single.nodes(data=True))
    assert list(batch.edges(keys=True)) == list(single.edges(keys=True))
    with pytest.raises(AssertionError, match="equal lengths"):
        batch.add_trajs([[0] * 5, [0]])
    assert batch._points == single._points
    assert batch.validate()


def test_spatial_batch_add_clusters_once():
    from reepy.trajectory import SpatialReebGraph

    calls = []

    def cluster(points):
        calls.append(points)
        return [
            [point for point in points if point == value]
            for value in dict.fromkeys(points)
        ]

    graph = SpatialReebGraph([], ExactClustering(cluster))
    graph.add_trajs(iter([iter([0, 1, 2]), iter([0, 3, 2])]))
    assert calls == [[(0,), (1,), (2,), (0,), (3,), (2,)]]
    assert graph.number_of_edges(0, 2) == 2
    assert graph.validate()


@pytest.mark.parametrize("kind", ["sequential", "spatial"])
def test_trajectory_add_preserves_the_atomic_base_contract(kind):
    from reepy.trajectory import SequentialReebGraph, SpatialReebGraph

    graph = (
        SequentialReebGraph([], lambda points: True)
        if kind == "sequential"
        else SpatialReebGraph([], ExactClustering())
    )
    points = [(7, 1, 2), (7, 0, 1)]
    for point in points:
        graph.add(
            (point[0], point[1], None, *point[2:]) if kind == "spatial" else point
        )
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert {
        (point[0], point[1], point[-1])
        for _, data in graph.nodes(data=True)
        for point in data["points"]
    } == set(points)
    assert graph.validate()


def test_spatial_atoms_require_adjacent_sample_indices():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([], ExactClustering())
    graph.add((0, 0, None, 1))
    graph.add((0, 2, None, 3))
    assert graph.number_of_edges() == 0
    graph.add((0, 1, None, 2))
    assert graph.number_of_edges() == 1
    assert graph.number_of_nodes() == 2


def test_batch_ids_do_not_collide_with_existing_atoms():
    from reepy.trajectory import SequentialReebGraph, SpatialReebGraph

    graphs = [
        SequentialReebGraph([], lambda points: True),
        SpatialReebGraph([], ExactClustering()),
    ]
    for graph in graphs:
        graph.add(
            (0, 0, 1) if isinstance(graph, SequentialReebGraph) else (0, 0, None, 1)
        )
        graph.add_trajs([[(2,)]])
        graph.add_trajs([[(3,)]])
        points = (
            graph._points[0]
            if isinstance(graph, SequentialReebGraph)
            else graph._points
        )
        assert [(point[0], point[1], point[-1]) for point in points] == [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 3),
        ]


def test_trajectory_storage_matches_each_modality():
    from reepy.trajectory import SequentialReebGraph, SpatialReebGraph

    sequential = SequentialReebGraph([], lambda points: True)
    spatial = SpatialReebGraph([], ExactClustering())
    for point in [(1, 1, 2), (0, 0, 1), (1, 0, 1)]:
        sequential.add(point)
        spatial.add((point[0], point[1], None, *point[2:]))
    assert sequential._points == {1: [(1, 1, 2)], 0: [(0, 0, 1), (1, 0, 1)]}
    assert spatial._points == [(0, 0, 0, 1), (1, 0, 0, 1), (1, 1, 1, 2)]
    assert sequential.validate() and spatial.validate()


def test_flatten_sequential_trajectories_produces_atoms_without_adding_them():
    from reepy.trajectory import SequentialReebGraph

    graph = SequentialReebGraph([], lambda points: True)
    assert graph.flatten([]) == []
    atoms = graph.flatten([[(1, 2), (3, 4)]])
    assert atoms == [(0, 0, 1, 2), (0, 1, 3, 4)]
    assert len(graph) == 0
    for atom in atoms:
        graph.add(atom)
    assert graph.number_of_edges() == 1
    assert graph.flatten(iter([iter([(5, 6)])])) == [(1, 0, 5, 6)]
    assert graph.validate()


def test_spatial_depths_and_flatten_trace_the_built_graph():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([[0, 1, 2, 3], [0, 3], [1, 4], [2, 5]], ExactClustering())
    assert {node: data["depth"] for node, data in graph.nodes(data=True)} == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
    }
    points = list(graph._points)
    atoms = graph.flatten([[0, 3]])
    assert atoms == [(4, 0, 0, 0), (4, 1, 3, 3)]
    assert graph._points == points
    for atom in atoms:
        graph.add(atom)
    assert 4 in graph[0][3][0]["trajs"]
    with pytest.raises(NotImplementedError, match="Cycles and more complex paths"):
        graph.flatten([[3, 0]])
    assert graph.flatten([[4, 5]]) == [(5, 0, 2, 4), (5, 1, 3, 5)]
    assert graph.validate()


def test_spatial_flatten_requires_a_built_graph_and_rejects_cycles():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([], ExactClustering())
    assert graph.flatten([]) == []
    with pytest.raises(ValueError, match="before flattening"):
        graph.flatten([[0]])
    cycle = SpatialReebGraph([[0, 1, 2, 0]], ExactClustering())
    assert next(iter(cycle.nodes(data=True)))[1]["depth"] == 0
    with pytest.raises(NotImplementedError, match="Cycles and more complex paths"):
        cycle.flatten([[0]])


def test_spatial_flatten_retains_bundles_and_interpolates_gaps():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([[0, 1, 2]], ExactClustering())
    assert graph.flatten([[0, 1, 2]]) == [(1, 0, 0, 0), (1, 1, 0.5, 1), (1, 2, 1, 2)]
    assert graph.flatten([[0, 8, 2]]) == [(1, 0, 0, 0), (1, 1, 0.5, 8), (1, 2, 1, 2)]
    assert graph.flatten([[0, 8, 9]]) == [(1, 0, 0, 0), (1, 1, 1, 8), (1, 2, 2, 9)]
    assert graph.flatten([[8, 9]]) == [(1, 0, 0, 8), (1, 1, 1, 9)]
    assert graph.flatten([[8, 0]]) == [(1, 0, -1, 8), (1, 1, 0, 0)]
    with pytest.raises(NotImplementedError, match="Cycles and more complex paths"):
        graph.flatten([[2, 1, 0]])


def test_spatial_continuity_is_adjacency_within_each_trajectory():
    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([], ExactClustering())
    a, b, c = (0, 0, 0, 1), (0, 1, 0.5, 2), (0, 2, 3, 3)
    assert graph.continuity(a, b)
    assert graph.continuity(b, a)
    assert graph.continuity(b, c)
    assert not graph.continuity(a, c)
    assert not graph.continuity(a, (1, 0, 0, 1))


@pytest.mark.parametrize(
    "trajectories",
    [
        [[0, 0, 1, 0, 0], [0, 0, -1, 0, 0]],
        [[1, 2, 3], [1, 2, 1]],
    ],
)
def test_sequential_matches_the_base_implementation(trajectories):
    import networkx as nx
    from reepy.trajectory import SequentialReebGraph

    sequential = SequentialReebGraph(trajectories, lambda points: len(set(points)) == 1)
    base = ReebGraph(sequential.level, sequential.connectivity, sequential.continuity)
    for points in sequential._points.values():
        for point in points:
            base.add(point)
    assert nx.is_isomorphic(
        base,
        sequential,
        node_match=lambda a, b: (
            a["level"] == b["level"] and set(a["points"]) == set(b["points"])
        ),
    )
    assert base.validate() and sequential.validate()


def test_spatial_matches_the_base_on_depth_ordered_trajectories():
    import networkx as nx
    from reepy.trajectory import SpatialReebGraph

    trajectories = [[0, 1, 2, 3], [0, 4, 2, 3]]
    spatial = SpatialReebGraph(trajectories, ExactClustering())
    atoms = spatial.flatten(trajectories)
    base = ReebGraph(spatial.level, spatial.connectivity, spatial.continuity)
    for point in atoms:
        base.add(point)
    assert nx.is_isomorphic(
        base,
        spatial,
        node_match=lambda a, b: (
            a["level"] == b["depth"]
            and {point[3:] for point in a["points"]}
            == {point[3:] for point in b["points"]}
        ),
    )
    assert base.number_of_nodes() == 3
    assert base.number_of_edges() == 3
    assert base.validate() and spatial.validate()


def test_spatial_flatten_does_not_change_graph_or_callback_state():
    from copy import deepcopy

    from reepy.trajectory import SpatialReebGraph

    graph = SpatialReebGraph([[0, 1, 2]], ExactClustering())
    before = deepcopy(
        (
            graph._points,
            graph._components,
            dict(graph.nodes(data=True)),
            list(graph.edges(keys=True, data=True)),
        )
    )
    callback = graph.continuity
    first = graph.flatten([[0, 8, 2]])
    graph.flatten([[0, 9, 10, 2]])
    assert (
        graph._points,
        graph._components,
        dict(graph.nodes(data=True)),
        list(graph.edges(keys=True, data=True)),
    ) == before
    assert not hasattr(graph, "_adjacent")
    assert graph.continuity is callback
    assert callback(first[0], first[1])
    assert not callback(first[0], first[2])
    assert graph.level(first[1]) == 0.5
    assert graph.order() == len(graph)
