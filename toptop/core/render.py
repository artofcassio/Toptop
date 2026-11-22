"""Captura de wireframe sem Maya: rasterizador simples em NumPy que grava PNG.

O ``mayapy`` não tem viewport, então o playblast não serve para a Fase 0. Este
módulo desenha a malha sombreada com o wireframe por cima (com remoção de
linhas ocultas por z-buffer) e marca o que interessa para julgar a topologia:

* arestas pretas: wireframe;
* laranja: edge loop sobre crease forte; azul: sobre crease guia ou borda;
* pontos magenta: trecho de crease original SEM edge loop;
* quadrado ciano: polo de valência 3 (ou menor); quadrado roxo: valência 5+.
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np

from .mesh import Mesh, face_normals_areas, triangulate

BACKGROUND = (246, 246, 248)
FRONT = np.array([196.0, 204.0, 216.0])
BACK = np.array([150.0, 120.0, 118.0])
WIRE = (38, 40, 46)
STRONG = (232, 96, 20)
GUIDE = (30, 110, 225)
MISSING = (225, 0, 190)
POLE_LOW = (0, 170, 200)
POLE_HIGH = (130, 20, 190)


def write_png(path: str, rgb: np.ndarray) -> None:
    """Grava um array (H, W, 3) uint8 como PNG (só zlib e struct)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)))
        handle.write(chunk(b"IEND", b""))


def read_png(path: str) -> np.ndarray:
    """Lê de volta um PNG gravado por ``write_png`` (para testes)."""
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("não é PNG")
    pos = 8
    width = height = 0
    body = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif tag == b"IDAT":
            body += payload
        pos += 12 + length
    rows = np.frombuffer(zlib.decompress(body), dtype=np.uint8).reshape(height, width * 3 + 1)
    return rows[:, 1:].reshape(height, width, 3)


def _camera(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eixos da câmera ortográfica (direita, cima, para a câmera), com Y para cima."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    toward = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    right = np.array([toward[2], 0.0, -toward[0]])
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    return right, up, toward


class _Panel:
    """Um quadro: buffer de cor, z-buffer e a projeção da malha."""

    def __init__(self, mesh: Mesh, width: int, height: int, azimuth: float, elevation: float) -> None:
        self.width, self.height = width, height
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.image[:] = BACKGROUND
        self.depth = np.full((height, width), -np.inf)
        self.right, self.up, self.toward = _camera(azimuth, elevation)
        lo, hi = mesh.bbox()
        self.center = 0.5 * (lo + hi)
        rel = mesh.points[mesh.topology.used_vertex] - self.center
        if len(rel) == 0:
            rel = np.zeros((1, 3))
        sx, sy = rel @ self.right, rel @ self.up
        span_x = max(float(sx.max() - sx.min()), 1e-12)
        span_y = max(float(sy.max() - sy.min()), 1e-12)
        margin = 0.06
        self.scale = min(width * (1 - 2 * margin) / span_x, height * (1 - 2 * margin) / span_y)
        self.shift_x = 0.5 * width - self.scale * 0.5 * float(sx.max() + sx.min())
        self.shift_y = 0.5 * height + self.scale * 0.5 * float(sy.max() + sy.min())
        depth = rel @ self.toward
        self.depth_span = max(float(depth.max() - depth.min()), 1e-12)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pontos 3D -> (x de tela, y de tela, profundidade; maior = mais perto)."""
        rel = np.asarray(points, dtype=np.float64) - self.center
        x = rel @ self.right * self.scale + self.shift_x
        y = self.shift_y - rel @ self.up * self.scale
        return x, y, rel @ self.toward

    def fill(self, mesh: Mesh) -> None:
        """Rasteriza as faces com sombreamento chapado e preenche o z-buffer."""
        tris, tri_face = triangulate(mesh)
        if len(tris) == 0:
            return
        normals, _ = face_normals_areas(mesh)
        n_view = normals[tri_face]
        facing = n_view @ self.toward
        light = self.right * 0.35 + self.up * 0.55 + self.toward * 0.76
        light /= np.linalg.norm(light)
        diffuse = np.abs(n_view @ light)
        shade = 0.42 + 0.58 * diffuse
        colors = np.where((facing >= 0)[:, None], FRONT, BACK) * shade[:, None]
        colors = np.clip(colors, 0, 255).astype(np.uint8)

        x, y, z = self.project(mesh.points)
        tx, ty, tz = x[tris], y[tris], z[tris]
        area = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
        x0 = np.clip(np.floor(tx.min(axis=1)).astype(np.int64), 0, self.width - 1)
        x1 = np.clip(np.ceil(tx.max(axis=1)).astype(np.int64), 0, self.width - 1)
        y0 = np.clip(np.floor(ty.min(axis=1)).astype(np.int64), 0, self.height - 1)
        y1 = np.clip(np.ceil(ty.max(axis=1)).astype(np.int64), 0, self.height - 1)
        visible = (np.abs(area) > 1e-9) & (tx.max(axis=1) >= 0) & (tx.min(axis=1) < self.width)
        visible &= (ty.max(axis=1) >= 0) & (ty.min(axis=1) < self.height)
        inv = 1.0 / np.where(np.abs(area) > 1e-9, area, 1.0)
        # w0 = a0*x + b0*y + c0 (peso do vértice 0), idem para o vértice 1
        a0 = (ty[:, 1] - ty[:, 2]) * inv
        b0 = (tx[:, 2] - tx[:, 1]) * inv
        c0 = (tx[:, 1] * ty[:, 2] - tx[:, 2] * ty[:, 1]) * inv
        a1 = (ty[:, 2] - ty[:, 0]) * inv
        b1 = (tx[:, 0] - tx[:, 2]) * inv
        c1 = (tx[:, 2] * ty[:, 0] - tx[:, 0] * ty[:, 2]) * inv
        image, depth = self.image, self.depth
        eps = -1e-6
        for t in np.flatnonzero(visible).tolist():
            xa, xb, ya, yb = x0[t], x1[t], y0[t], y1[t]
            px = np.arange(xa, xb + 1) + 0.5
            py = (np.arange(ya, yb + 1) + 0.5)[:, None]
            w0 = a0[t] * px + b0[t] * py + c0[t]
            w1 = a1[t] * px + b1[t] * py + c1[t]
            w2 = 1.0 - w0 - w1
            zz = w0 * tz[t, 0] + w1 * tz[t, 1] + w2 * tz[t, 2]
            sub = depth[ya : yb + 1, xa : xb + 1]
            hit = (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (zz > sub)
            if hit.any():
                sub[hit] = zz[hit]
                image[ya : yb + 1, xa : xb + 1][hit] = colors[t]

    def lines(self, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
        """Desenha segmentos 3D visíveis (teste de profundidade contra o z-buffer)."""
        if len(a) == 0:
            return
        ax, ay, az = self.project(a)
        bx, by, bz = self.project(b)
        length = np.hypot(bx - ax, by - ay)
        n = np.maximum(2, np.ceil(length / 0.6).astype(np.int64) + 1)
        seg = np.repeat(np.arange(len(n)), n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        t = local / (n[seg] - 1)
        self._plot(ax[seg] + (bx - ax)[seg] * t, ay[seg] + (by - ay)[seg] * t, az[seg] + (bz - az)[seg] * t, color, thickness)

    def dots(self, points: np.ndarray, color: tuple[int, int, int], size: int) -> None:
        """Desenha quadrados em pontos 3D visíveis."""
        if len(points) == 0:
            return
        x, y, z = self.project(points)
        self._plot(x, y, z, color, size)

    def _plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, color: tuple[int, int, int], size: int) -> None:
        """Pinta os pixels (com espessura) que passam no teste de profundidade."""
        ix = np.floor(x).astype(np.int64)
        iy = np.floor(y).astype(np.int64)
        inside = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        ix, iy, z = ix[inside], iy[inside], z[inside]
        # folga: o segmento está sobre a superfície; o z interpolado difere um pouco
        near = np.full(len(ix), -np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sx = np.clip(ix + dx, 0, self.width - 1)
                sy = np.clip(iy + dy, 0, self.height - 1)
                near = np.maximum(near, -self.depth[sy, sx])
        seen = z >= -near - 0.004 * self.depth_span
        ix, iy = ix[seen], iy[seen]
        half = size // 2
        for dx in range(-half, size - half):
            for dy in range(-half, size - half):
                px = np.clip(ix + dx, 0, self.width - 1)
                py = np.clip(iy + dy, 0, self.height - 1)
                self.image[py, px] = color


def render_wireframe(
    mesh: Mesh,
    path: str,
    views: tuple[tuple[float, float], ...] = ((35.0, 24.0), (215.0, -20.0)),
    panel: tuple[int, int] = (1200, 900),
    strong_edges: np.ndarray | None = None,
    guide_edges: np.ndarray | None = None,
    missing_points: np.ndarray | None = None,
    show_poles: bool = True,
    supersample: int = 2,
) -> str:
    """Grava um PNG com a malha em wireframe sobre sombreado, um quadro por vista.

    ``views`` são pares (azimute, elevação) em graus; o padrão mostra a frente
    por cima e as costas por baixo. ``strong_edges`` e
    ``guide_edges`` são pares de vértices da própria malha (os loops que caíram
    sobre creases); ``missing_points`` são pontos da crease original sem loop.
    """
    width, height = panel[0] * supersample, panel[1] * supersample
    topo = mesh.topology
    edges = topo.edges[~topo.degenerate_edge]
    points = mesh.points
    panels = []
    for azimuth, elevation in views:
        p = _Panel(mesh, width, height, azimuth, elevation)
        p.fill(mesh)
        p.lines(points[edges[:, 0]], points[edges[:, 1]], WIRE, supersample)
        if guide_edges is not None and len(guide_edges):
            p.lines(points[guide_edges[:, 0]], points[guide_edges[:, 1]], GUIDE, 2 * supersample)
        if strong_edges is not None and len(strong_edges):
            p.lines(points[strong_edges[:, 0]], points[strong_edges[:, 1]], STRONG, 2 * supersample)
        if missing_points is not None and len(missing_points):
            p.dots(missing_points, MISSING, 3 * supersample)
        if show_poles:
            interior = topo.used_vertex & ~topo.boundary_vertex
            valence = topo.vertex_valence
            p.dots(points[interior & (valence < 4)], POLE_LOW, 5 * supersample)
            p.dots(points[interior & (valence > 4)], POLE_HIGH, 5 * supersample)
        image = p.image
        if supersample > 1:
            image = (
                image.reshape(panel[1], supersample, panel[0], supersample, 3).astype(np.float64).mean(axis=(1, 3))
            )
            image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        panels.append(image)
    write_png(path, np.concatenate(panels, axis=1))
    return path
