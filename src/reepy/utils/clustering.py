from abc import ABC, abstractmethod
from itertools import combinations
from math import asin, cos, dist, isfinite, radians, sin, sqrt


class Clustering(ABC):
    def __init__(self, cluster):
        self._cluster = cluster

    def cluster(self, points):
        return self._cluster(list(points))

    @abstractmethod
    def connectivity(self):
        pass


class _MetricClustering(Clustering):
    def __init__(self, epsilon, distance):
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        self.epsilon = epsilon
        self.distance = distance
        super().__init__(self._partition)

    def connectivity(self):
        return lambda points: all(
            self.distance(a, b) <= self.epsilon for a, b in combinations(points, 2)
        )

    def _partition(self, points):
        clusters = []
        for point in points:
            for cluster in clusters:
                if all(
                    self.distance(point, other) <= self.epsilon for other in cluster
                ):
                    cluster.append(point)
                    break
            else:
                clusters.append([point])
        return clusters


class EuclideanClustering(_MetricClustering):
    def __init__(self, epsilon):
        super().__init__(epsilon, dist)


class HaversineClustering(_MetricClustering):
    # Points are (latitude, longitude) in degrees; epsilon and radius are in km.
    def __init__(self, epsilon, radius=6371.0088):
        if not isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        self.radius = radius
        super().__init__(epsilon, self._distance)

    def _distance(self, a, b):
        lat_a, lon_a = map(radians, a)
        lat_b, lon_b = map(radians, b)
        h = (
            sin((lat_b - lat_a) / 2) ** 2
            + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
        )
        return 2 * self.radius * asin(sqrt(min(1, max(0, h))))
