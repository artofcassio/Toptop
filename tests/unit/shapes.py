"""Malhas sintéticas para os testes sem Maya."""

from __future__ import annotations

import numpy as np

from toptop.core.mesh import Mesh


def grid_plane(nx: int, ny: int, sx: float = 1.0, sy: float = 1.0, z: float = 0.0) -> Mesh:
    """Plano em XY com nx por ny quads, normal +Z, tamanho sx por sy."""
    xs = np.linspace(0.0, sx, nx + 1)
    ys = np.linspace(0.0, sy, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    faces = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b = (i + 1) * (ny + 1) + j
            faces.append((a, b, b + 1, a + 1))
    return Mesh.from_faces(points, faces, "plane")


def cube(n: int = 1, size: float = 1.0) -> Mesh:
    """Cubo fechado e soldado com n por n quads em cada lado."""
    index: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []

    def vid(ijk: tuple[int, int, int]) -> int:
        if ijk not in index:
            index[ijk] = len(points)
            points.append(tuple(c * size / n for c in ijk))
        return index[ijk]

    faces = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for value, flip in ((n, False), (0, True)):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        ijk = [0, 0, 0]
                        ijk[axis] = value
                        ijk[u] = i + du
                        ijk[v] = j + dv
                        quad.append(vid(tuple(ijk)))
                    faces.append(tuple(reversed(quad)) if flip else tuple(quad))
    return Mesh.from_faces(points, faces, "cube")


def cylinder(
    sides: int, rings: int = 1, radius: float = 1.0, height: float = 2.0, caps: str = "none"
) -> Mesh:
    """Cilindro em Y. ``caps``: 'none' (aberto), 'ngon' ou 'fan' (leque de triângulos)."""
    theta = np.arange(sides) * (2.0 * np.pi / sides)
    points = []
    for r in range(rings + 1):
        y = height * r / rings
        for t in theta:
            points.append((radius * np.cos(t), y, radius * np.sin(t)))
    faces = []
    for r in range(rings):
        for s in range(sides):
            a = r * sides + s
            b = r * sides + (s + 1) % sides
            faces.append((a, a + sides, b + sides, b))
    top = rings * sides
    if caps == "ngon":
        faces.append(tuple(range(sides)))
        faces.append(tuple(top + s for s in reversed(range(sides))))
    elif caps == "fan":
        bottom_center = len(points)
        points.append((0.0, 0.0, 0.0))
        top_center = len(points)
        points.append((0.0, height, 0.0))
        for s in range(sides):
            nxt = (s + 1) % sides
            faces.append((bottom_center, s, nxt))
            faces.append((top_center, top + nxt, top + s))
    return Mesh.from_faces(points, faces, "cylinder")


def torus(n_major: int = 16, n_minor: int = 8, major: float = 2.0, minor: float = 0.5) -> Mesh:
    """Toro só de quads: fechado, todo vértice com valência 4."""
    points = []
    for i in range(n_major):
        a = 2.0 * np.pi * i / n_major
        for j in range(n_minor):
            b = 2.0 * np.pi * j / n_minor
            r = major + minor * np.cos(b)
            points.append((r * np.cos(a), minor * np.sin(b), r * np.sin(a)))
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = i * n_minor + (j + 1) % n_minor
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = ((i + 1) % n_major) * n_minor + j
            faces.append((a, b, c, d))
    return Mesh.from_faces(points, faces, "torus")


def combine(*meshes: Mesh) -> Mesh:
    """Junta malhas num só objeto, sem soldar (uma casca por malha)."""
    points, counts, connects = [], [], []
    base = 0
    for m in meshes:
        points.append(m.points)
        counts.append(m.counts)
        connects.append(m.connects + base)
        base += m.num_points
    return Mesh(np.concatenate(points), np.concatenate(counts), np.concatenate(connects), "combined")


def translated(mesh: Mesh, offset: tuple[float, float, float]) -> Mesh:
    """Cópia deslocada."""
    return mesh.with_points(mesh.points + np.asarray(offset, dtype=np.float64))


def cube_feature_edges(mesh: Mesh, size: float = 1.0) -> np.ndarray:
    """Pares de vértices das arestas da malha que ficam sobre as 12 arestas do cubo."""
    p = mesh.points
    on_side = (np.abs(p) < 1e-9) | (np.abs(p - size) < 1e-9)
    edges = mesh.topology.edges
    both = on_side[edges[:, 0]] & on_side[edges[:, 1]]
    return edges[both.sum(axis=1) >= 2]
