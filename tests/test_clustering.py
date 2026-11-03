from itertools import combinations

import pytest
from reepy import ReebGraph
from reepy.utils.clustering import Clustering, EuclideanClustering, HaversineClustering


@pytest.mark.parametrize(
    "method, points",
    [
        (EuclideanClustering(1), [(0,), (0.9,), (1.8,), (0,)]),
        (HaversineClustering(30), [(0, 179.9), (0, -179.9), (0, 0)]),
    ],
)
def test_clusters_match_their_connectivity_rule(method, points):
    clusters = method.cluster(iter(points))
    connected = method.connectivity()
    assert sorted(map(id, points)) == sorted(
        id(point) for cluster in clusters for point in cluster
    )
    assert len(clusters) == 2
    assert all(connected(cluster) for cluster in clusters)
    assert all(not connected(a + b) for a, b in combinations(clusters, 2))


def test_haversine_uses_degrees_and_kilometers():
    connected = HaversineClustering(112).connectivity()
    assert connected([(0, 0), (0, 1)])
    assert not HaversineClustering(111).connectivity()([(0, 0), (0, 1)])


@pytest.mark.parametrize("epsilon", [-1, float("nan"), float("inf")])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        EuclideanClustering(epsilon)


def test_clustering_base_requires_a_connectivity_rule():
    with pytest.raises(TypeError):
        Clustering(lambda points: [points])


def test_validate_checks_both_internal_connectivity_and_separation():
    graph = ReebGraph(
        lambda point: 0, lambda points: len(set(points)) <= 1, lambda a, b: False
    )
    graph.add(0)
    graph.add(1)
    assert graph.validate()
    graph.connectivity = lambda points: True
    assert not graph.validate()
    graph.connectivity = lambda points: False
    assert not graph.validate()
