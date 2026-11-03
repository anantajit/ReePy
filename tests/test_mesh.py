from itertools import combinations, permutations
from math import cos, pi, sin

import networkx as nx
import pytest
from reepy import ReebGraph


def torus(n=4):
    points = []
    for i in range(n):
        u = 2 * pi * i / n
        for j in range(n):
            v = 2 * pi * j / n
            points.append(
                tuple(
                    round(x, 10)
                    for x in (
                        (2 + cos(v)) * cos(u),
                        (2 + cos(v)) * sin(u),
                        sin(v),
                    )
                )
            )
    faces = []
    for i in range(n):
        for j in range(n):
            a, b = i * n + j, ((i + 1) % n) * n + j
            c, d = ((i + 1) % n) * n + (j + 1) % n, i * n + (j + 1) % n
            faces.extend([(a, b, c), (a, c, d)])
    return points, faces


def inside(point, polygon):
    x, y = point
    enclosed = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) < 1e-9
            and min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9
            and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9
        ):
            return True
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            enclosed = not enclosed
    return enclosed


def mesh_graph(vertices, faces, axis, reverse=False):
    samples, labels, cells = [], {}, {}
    for level in sorted({point[axis] for point in vertices}):
        segments = nx.Graph()
        positions = {}
        for cell, face in enumerate(faces):
            hits = {(i,) for i in face if vertices[i][axis] == level}
            for a, b in combinations(face, 2):
                if (
                    min(vertices[a][axis], vertices[b][axis])
                    < level
                    < max(vertices[a][axis], vertices[b][axis])
                ):
                    hits.add(tuple(sorted((a, b))))
            segments.add_nodes_from(hits)
            segments.add_edges_from(combinations(hits, 2))
            for token in hits:
                sample = (level, token)
                cells.setdefault(sample, set()).add(cell)
                a, b = vertices[token[0]], vertices[token[-1]]
                t = 0 if len(token) == 1 else (level - a[axis]) / (b[axis] - a[axis])
                positions[token] = tuple(
                    a[d] + t * (b[d] - a[d]) for d in range(3) if d != axis
                )
        regions = nx.Graph()
        regions.add_nodes_from(segments)
        for cycle in nx.cycle_basis(segments):
            polygon = [positions[token] for token in cycle]
            covered = [token for token in segments if inside(positions[token], polygon)]
            regions.add_edges_from((covered[0], token) for token in covered[1:])
        for component, tokens in enumerate(nx.connected_components(regions)):
            for token in sorted(tokens):
                sample = (level, token)
                samples.append(sample)
                labels[sample] = component
    graph = ReebGraph(
        level=lambda point: point[0],
        connectivity=lambda group: len({labels[point] for point in group}) == 1,
        continuity=lambda a, b: not cells[a].isdisjoint(cells[b]),
        store_points=True,
    )
    for sample in reversed(samples) if reverse else samples:
        graph.add(sample)
    return graph.to_networkx()


@pytest.mark.parametrize(
    "axis, counts", [(0, (4, 4)), (1, (4, 4)), (2, (2, 1))], ids=["x", "y", "z"]
)
def test_torus(axis, counts):
    vertices, faces = torus()
    graph = mesh_graph(vertices, faces, axis)
    assert (graph.number_of_nodes(), graph.number_of_edges()) == counts
    assert nx.is_weakly_connected(graph)
    assert nx.is_directed_acyclic_graph(graph)
    reverse = mesh_graph(vertices, faces, axis, reverse=True)
    assert nx.is_isomorphic(
        graph, reverse, node_match=lambda a, b: a["level"] == b["level"]
    )
    if axis != 2:
        split = next(node for node in graph if graph.out_degree(node) == 2)
        merge = next(node for node in graph if graph.in_degree(node) == 2)
        assert graph.number_of_edges(split, merge) == 2
        assert any(len(token) == 2 for _, token in graph.nodes[split]["points"])


def human():
    blocks = [(x, z) for z in (0, 1) for x in (-1, 1)]
    blocks += [(x, 2) for x in (-1, 0, 1)]
    blocks += [(x, 3) for x in range(-2, 3)] + [(0, 4)]
    blocks += [(x, 5) for x in (-1, 0, 1)]
    vertices, indices, cells = [], {}, []
    for x, z in blocks:
        corners = {}
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    point = (x + dx - 0.5, (dy - 0.5) * 0.6, z + dz)
                    if point not in indices:
                        indices[point] = len(vertices)
                        vertices.append(point)
                    corners[dx, dy, dz] = indices[point]
        for order in permutations(range(3)):
            corner, cell = [0, 0, 0], [corners[0, 0, 0]]
            for axis in order:
                corner[axis] = 1
                cell.append(corners[tuple(corner)])
            cells.append(tuple(cell))
    faces = {}
    for cell in cells:
        for face in combinations(cell, 3):
            key = tuple(sorted(face))
            faces[key] = faces.get(key, 0) + 1
    return vertices, [face for face, count in faces.items() if count == 1], cells


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_block_person(axis):
    vertices, faces, _ = human()
    graph = mesh_graph(vertices, faces, axis)
    assert nx.is_weakly_connected(graph)
    assert nx.is_directed_acyclic_graph(graph)
