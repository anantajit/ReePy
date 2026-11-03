from collections.abc import Sized
from itertools import chain

from ..ReebGraph import ReebGraph
from ._points import trajectory_points


class SequentialReebGraph(ReebGraph):
    def __init__(self, trajectories, connectivity, store_points=False):
        super().__init__(
            level=lambda point: point[1],
            connectivity=lambda points: connectivity([point[2:] for point in points]),
            continuity=lambda a, b: a[0] == b[0] and abs(b[1] - a[1]) == 1,
            store_points=store_points,
        )
        self._points = {}
        self._size = None
        self.add_trajs(trajectories)

    def add(self, point):
        self._points.setdefault(point[1], []).append(point)
        self._build()

    def _build(self):
        super()._build(self._points)

    def flatten(self, trajectories):
        return list(
            trajectory_points(trajectories, chain.from_iterable(self._points.values()))
        )

    def add_trajs(self, trajectories):
        trajectories = list(trajectories)
        assert all(isinstance(trajectory, Sized) for trajectory in trajectories), (
            "Sequential trajectories must be sized"
        )
        sizes = {len(trajectory) for trajectory in trajectories}
        if self._size is not None:
            sizes.add(self._size)
        assert len(sizes) <= 1, "Sequential trajectories must have equal lengths"
        for point in self.flatten(trajectories):
            self._points.setdefault(point[1], []).append(point)
        self._size = next(iter(sizes), None)
        self._build()
