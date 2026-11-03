from collections.abc import Iterable


def trajectory_points(trajectories, points):
    used = {point[0] for point in points}
    trajectory_id = 0
    for trajectory in trajectories:
        while trajectory_id in used:
            trajectory_id += 1
        for index, point in enumerate(trajectory):
            coordinates = tuple(point) if isinstance(point, Iterable) else (point,)
            yield (trajectory_id, index, *coordinates)
        trajectory_id += 1
