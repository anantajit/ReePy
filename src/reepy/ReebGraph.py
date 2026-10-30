from typing import Iterable

from networkx import DiGraph


class _Bundle:
    def __init__(self, point, trajectory_id):
        self.point = point
        self.trajectory_ids = {trajectory_id}

    def add_trajectory(self, trajectory_id):
        self.trajectory_ids.add(trajectory_id)


class ReebGraph(DiGraph):
    """
    orderer: f(x) returns a real, scalar value
    equivalence: f(x, y) returns True if two classes are equivalent,
                 False otherwise. Must be reflexive and symmetric
    """

    def __init__(self, orderer=None, equivalence=None):
        super().__init__()

        self.orderer = (lambda x: x) if orderer is None else orderer
        self.equivalence = (lambda x, y: x == y) if equivalence is None else equivalence

        self.bundles = {}
        self._trajectory_count = 0

    """
    traj: an iterable of points (input into orderer/equivalence). Conceptually,
          we consider these points to be "continuous" with respect to each other
    """

    def append_trajectory(self, traj):
        trajectory_id = self._trajectory_count
        self._trajectory_count += 1

        level_sets = {}

        # compute level sets
        for point in traj:
            level = self.orderer(point)
            if level not in level_sets:
                level_sets[level] = []
            level_sets[level].append(point)

        # once level sets are computed, construct local bundles
        for level, points in level_sets.items():
            if level not in self.bundles:
                self.bundles[level] = []

            for point in points:
                connected = False

                # check if point matches any bundle
                # by equivalence relation property, bundle can connect
                # to at most one bundle
                for bundle in self.bundles[level]:
                    if self.equivalence(bundle.point, point):
                        bundle.add_trajectory(trajectory_id)
                        connected = True
                        break

                if not connected:
                    bundle = _Bundle(point, trajectory_id)
                    self.bundles[level].append(bundle)

        self.__build()

    def append_trajectories(self, trajs: Iterable):
        for traj in trajs:
            self.append_trajectory(traj)
        self.__build()

    def __build(self):
        # clear existing graph nodes/edges so build is idempotent
        self.clear()

        # states[traj_id] -> the node id of the last node that trajectory belonged to
        states = {}
        # node_bundles[node_id] -> the bundle at that node (for trajectory_ids lookup)
        node_bundles = {}

        nodec = 0

        # iterate bundles in sorted level order
        disappear_level = max(self.bundles.keys())
        for level in sorted(self.bundles.keys()):
            for bundle in self.bundles[level]:
                curr_ids = bundle.trajectory_ids

                # check if connectivity changed vs previous bundle
                first_traj = next(iter(curr_ids))
                prev_node = states.get(first_traj)
                prev_ids = (
                    node_bundles[prev_node].trajectory_ids
                    if prev_node is not None
                    else None
                )

                if curr_ids != prev_ids or level == disappear_level:
                    # create a new node
                    self.add_node(
                        nodec,
                        level=level,
                        point=bundle.point,
                        trajs=bundle.trajectory_ids,
                    )
                    node_bundles[nodec] = bundle

                    predecessor_counts = {}
                    for traj in curr_ids:
                        if traj in states:
                            pred = states[traj]
                            predecessor_counts[pred] = (
                                predecessor_counts.get(pred, 0) + 1
                            )
                        states[traj] = nodec

                    for pred, count in predecessor_counts.items():
                        weight = count / len(node_bundles[pred].trajectory_ids)
                        self.add_edge(pred, nodec, weight=weight)

                    nodec += 1
