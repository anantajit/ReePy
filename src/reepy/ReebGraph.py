from itertools import combinations

from networkx import MultiDiGraph


class ReebGraph(MultiDiGraph):
    def __init__(self, level, connectivity, continuity, store_points=False):
        super().__init__()
        self.level = level
        self.connectivity = connectivity
        self.continuity = continuity
        self.store_points = store_points
        self._points = []
        self._components = {}

    def add(self, point):
        self._points.append(point)
        self._build()

    def flatten(self, data):
        return list(data)

    def _build(self, level_sets=None):
        if level_sets is None:
            level_sets = {}
            for point in self._points:
                level_sets.setdefault(self.level(point), []).append(point)

        graph = self
        graph.clear()
        self._components = {}
        previous = []
        for level in sorted(level_sets):
            current = []
            remaining = level_sets[level]
            while remaining:
                selected = next(
                    indices
                    for size in range(len(remaining), 0, -1)
                    for indices in combinations(range(len(remaining)), size)
                    if size == 1 or self.connectivity([remaining[i] for i in indices])
                )
                points = [remaining[i] for i in selected]
                self._components.setdefault(level, []).append(points)
                selected = set(selected)
                remaining = [
                    point for i, point in enumerate(remaining) if i not in selected
                ]
                node = graph.number_of_nodes()
                graph.add_node(node, level=level, points=points)
                current.append(node)
                for source in previous:
                    if any(
                        self.continuity(a, b)
                        for a in graph.nodes[source]["points"]
                        for b in points
                    ):
                        graph.add_edge(source, node)
            previous = current

        for node in list(graph.nodes):
            if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
                source = next(iter(graph.predecessors(node)))
                target = next(iter(graph.successors(node)))
                graph.add_edge(source, target)
                graph.remove_node(node)

    def validate(self):
        return all(
            all(self.connectivity(points) for points in clusters)
            and all(not self.connectivity(a + b) for a, b in combinations(clusters, 2))
            for clusters in self._components.values()
        )

    def to_networkx(self, store_points=None):
        if store_points is None:
            store_points = self.store_points
        graph = MultiDiGraph()
        for node, attributes in self.nodes(data=True):
            graph.add_node(
                node,
                **{key: value for key, value in attributes.items() if key != "points"},
            )
            if store_points:
                graph.nodes[node]["points"] = list(attributes["points"])
        graph.add_edges_from(self.edges(keys=True))
        return graph
