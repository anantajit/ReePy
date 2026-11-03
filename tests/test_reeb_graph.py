import networkx as nx
from reepy import ReebGraph


def test_only_consecutive_levels_are_linked():
    builder = ReebGraph(lambda point: point, lambda group: True, lambda a, b: True)
    for point in [0, 1, 2]:
        builder.add(point)
    graph = builder.to_networkx()
    assert [data["level"] for _, data in graph.nodes(data=True)] == [0, 2]
    assert graph.number_of_edges() == 1


def test_multidigraph_export_removes_builder_metadata():
    builder = ReebGraph(lambda point: point, lambda points: True, lambda a, b: True)
    builder.add(0)
    builder.add(1)
    assert isinstance(builder, nx.MultiDiGraph)
    assert nx.is_directed_acyclic_graph(builder)
    builder.graph["source"] = "test"
    exported = builder.to_networkx()
    assert type(exported) is nx.MultiDiGraph
    assert exported.graph == {}
    assert not hasattr(exported, "_points")
    assert not hasattr(exported, "continuity")
    assert list(exported.nodes(data=True)) == [(0, {"level": 0}), (1, {"level": 1})]
    assert list(exported.edges(keys=True)) == list(builder.edges(keys=True))
    detailed = builder.to_networkx(store_points=True)
    detailed.nodes[0]["points"].clear()
    assert builder.nodes[0]["points"] == [0]
    exported.clear()
    assert len(builder) == 2
    builder.add(2)
    assert len(builder.to_networkx()) == 2


def test_flatten_preserves_base_points_without_adding_them():
    graph = ReebGraph(lambda point: point, lambda points: True, lambda a, b: True)
    assert graph.flatten(iter([1, 2])) == [1, 2]
    assert graph.flatten([]) == []
    assert len(graph) == 0
