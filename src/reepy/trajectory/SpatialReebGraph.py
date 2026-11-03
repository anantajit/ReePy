from bisect import bisect_right, insort
from collections import Counter
from itertools import groupby, pairwise

from networkx import condensation, topological_sort

from ..ReebGraph import ReebGraph
from ._points import trajectory_points


class SpatialReebGraph(ReebGraph):
    def __init__(self, trajectories, clustering, store_points=False):
        connected = clustering.connectivity()
        super().__init__(
            lambda point: point[2],
            lambda points: connected([point[3:] for point in points]),
            lambda a, b: a[0] == b[0] and abs(a[1] - b[1]) == 1,
            store_points,
        )
        self.clustering = clustering
        self.add_trajs(trajectories)

    def add(self, point):
        assert len(point) >= 4, "Expected (trajectory_id, index, depth, *coordinates)"
        insort(self._points, point, key=lambda point: point[:2])
        self._build()

    def flatten(self, trajectories):
        trajectories = list(trajectories)
        if not trajectories:
            return []
        if not self:
            raise ValueError("Construct the spatial Reeb graph before flattening")
        result = []
        atoms = trajectory_points(trajectories, self._points)
        for trajectory_id, trajectory in groupby(atoms, key=lambda point: point[0]):
            trajectory = list(trajectory)
            anchors = []
            for index, point in enumerate(trajectory):
                matches = [
                    label
                    for label, cluster in enumerate(self._components[None])
                    if self.connectivity(
                        [(point[0], point[1], None, *point[2:])] + cluster
                    )
                ]
                if len(matches) > 1:
                    raise NotImplementedError(
                        "Ambiguous matches and more complex paths are not yet implemented"
                    )
                if not matches:
                    continue
                label = matches[0]
                depth = self._bundle_depths[label]
                if label in self._cyclic or (anchors and depth < anchors[-1][1]):
                    raise NotImplementedError(
                        "Cycles and more complex paths are not yet implemented"
                    )
                anchors.append((index, depth))
            indices = [index for index, _ in anchors]
            for index, point in enumerate(trajectory):
                right = bisect_right(indices, index)
                if not anchors:
                    depth = index
                elif right == 0:
                    depth = anchors[0][1] + index - anchors[0][0]
                elif right == len(anchors):
                    depth = anchors[-1][1] + index - anchors[-1][0]
                else:
                    a, start = anchors[right - 1]
                    b, end = anchors[right]
                    depth = start + (end - start) * (index - a) / (b - a)
                result.append((trajectory_id, index, depth, *point[2:]))
        return result

    def add_trajs(self, trajectories):
        self._points.extend(
            (trajectory_id, index, None, *coordinates)
            for trajectory_id, index, *coordinates in trajectory_points(
                trajectories, self._points
            )
        )
        self._points.sort(key=lambda point: point[:2])
        self._build()

    def _build(self):
        self.clear()
        self._components = {}
        self._cyclic = set()
        self._bundle_depths = {}
        # Cluster.
        points = [point[3:] for point in self._points]
        atoms = dict(zip(map(id, points), self._points))
        if not points:
            return
        clusters = [list(cluster) for cluster in self.clustering.cluster(points)]
        assert all(clusters) and Counter(map(id, points)) == Counter(
            id(point) for cluster in clusters for point in cluster
        ), "Clustering must partition the input points"
        self._components = {
            None: [[atoms[id(point)] for point in cluster] for cluster in clusters]
        }
        labels = {}
        for label, cluster in enumerate(clusters):
            self.add_node(label, points=self._components[None][label])
            for point in cluster:
                assert id(point) not in labels or labels[id(point)] == label, (
                    "Identical points must share a cluster"
                )
                labels[id(point)] = label
        # Connect.
        for trajectory_id, trajectory in groupby(
            zip(self._points, points), key=lambda pair: pair[0][0]
        ):
            for (a, point_a), (b, point_b) in pairwise(trajectory):
                source, target = labels[id(point_a)], labels[id(point_b)]
                if b[1] == a[1] + 1 and source != target:
                    if not self.has_edge(source, target):
                        self.add_edge(source, target, trajs=set(), via=[source, target])
                    self[source][target][0]["trajs"].add(trajectory_id)
        # Simplify.
        for node in list(self):
            if self.in_degree(node) == 1 and self.out_degree(node) == 1:
                source, _, incoming = next(iter(self.in_edges(node, data=True)))
                _, target, outgoing = next(iter(self.out_edges(node, data=True)))
                if (
                    source != node
                    and target != node
                    and incoming["trajs"] == outgoing["trajs"]
                ):
                    self.add_edge(
                        source,
                        target,
                        trajs=set(incoming["trajs"]),
                        via=incoming["via"] + outgoing["via"][1:],
                    )
                    self.remove_node(node)

        # Node depths.
        dag = condensation(self)
        depths = {}
        for component in topological_sort(dag):
            depths[component] = max(
                (depths[parent] + 1 for parent in dag.predecessors(component)),
                default=0,
            )
            members = dag.nodes[component]["members"]
            for node in members:
                self.nodes[node]["depth"] = depths[component]
                if len(members) > 1 or self.has_edge(node, node):
                    self._cyclic.add(node)

        # Bundle depths.
        self._bundle_depths = {
            node: data["depth"] for node, data in self.nodes(data=True)
        }
        for source, target, data in self.edges(data=True):
            start, end = self.nodes[source]["depth"], self.nodes[target]["depth"]
            path = data["via"]
            for index, label in enumerate(path):
                self._bundle_depths[label] = start + (end - start) * index / (
                    len(path) - 1
                )
            if dag.graph["mapping"][source] == dag.graph["mapping"][target]:
                self._cyclic.update(path)

        # Atom depths.
        updated = {
            id(point): (point[0], point[1], self._bundle_depths[label], *point[3:])
            for label, cluster in enumerate(self._components[None])
            for point in cluster
        }
        self._points = [updated[id(point)] for point in self._points]
        for cluster in self._components[None]:
            cluster[:] = [updated[id(point)] for point in cluster]
