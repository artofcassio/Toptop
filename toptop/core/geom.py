"""Distâncias e amostragem em NumPy puro (sem SciPy, sem Maya).

O ponto central é ``PrimitiveGrid``: uma grade uniforme sobre segmentos ou
triângulos que responde "qual é o primitivo mais próximo de cada ponto" de
forma exata e vetorizada. É o que sustenta o desvio de forma (Hausdorff), a
fidelidade das quinas e a reprojeção.
"""

from __future__ import annotations

import numpy as np

_OFFSETS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64
)
_POINT_BLOCK = 16384
_PAIR_CHUNK = 400_000
_BUILD_CHUNK = 200_000


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produto escalar linha a linha."""
    return np.einsum("ij,ij->i", a, b)


def closest_on_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distância ao quadrado e ponto mais próximo de p[i] no segmento a[i]-b[i]."""
    ab = b - a
    den = _dot(ab, ab)
    t = _dot(p - a, ab) / np.where(den > 0, den, 1.0)
    t = np.where(den > 0, np.clip(t, 0.0, 1.0), 0.0)
    c = a + ab * t[:, None]
    d = p - c
    return _dot(d, d), c


def closest_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distância ao quadrado e ponto mais próximo de p[i] no triângulo a[i] b[i] c[i].

    Classificação por regiões de Voronoi (Ericson, Real-Time Collision Detection).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        den = va + vb + vc
        v_in = vb / den
        w_in = vc / den
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    zero = np.zeros_like(d1)
    one = np.ones_like(d1)
    v = np.select(conds, [zero, one, t_ab, zero, zero, 1.0 - t_bc], default=v_in)
    w = np.select(conds, [zero, zero, zero, one, t_ac, t_bc], default=w_in)
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    q = a + ab * v[:, None] + ac * w[:, None]
    d = p - q
    return _dot(d, d), q


def _split_segments(prims: np.ndarray, parents: np.ndarray, max_len: float) -> tuple[np.ndarray, np.ndarray]:
    """Divide segmentos longos em pedaços de no máximo ``max_len``."""
    a, b = prims[:, 0], prims[:, 1]
    length = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(length / max_len)).astype(np.int64)
    if int(n.max(initial=1)) == 1:
        return prims, parents
    seg = np.repeat(np.arange(len(n)), n)
    local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t0 = (local / n[seg])[:, None]
    t1 = ((local + 1) / n[seg])[:, None]
    d = (b - a)[seg]
    return np.stack([a[seg] + d * t0, a[seg] + d * t1], axis=1), parents[seg]


def _split_triangles(prims: np.ndarray, parents: np.ndarray, max_len: float) -> tuple[np.ndarray, np.ndarray]:
    """Divide triângulos grandes pela aresta mais longa até caberem em ``max_len``."""
    done_p: list[np.ndarray] = []
    done_i: list[np.ndarray] = []
    cur, cur_i = prims, parents
    limit = max_len * max_len
    for _ in range(64):
        if len(cur) == 0:
            break
        edges = np.stack([cur[:, 1] - cur[:, 0], cur[:, 2] - cur[:, 1], cur[:, 0] - cur[:, 2]], axis=1)
        l2 = np.einsum("ijk,ijk->ij", edges, edges)
        longest = l2.argmax(axis=1)
        big = l2.max(axis=1) > limit
        done_p.append(cur[~big])
        done_i.append(cur_i[~big])
        cur = cur[big]
        cur_i = cur_i[big]
        if len(cur) == 0:
            break
        rows = np.arange(len(cur))[:, None]
        cur = cur[rows, (longest[big][:, None] + np.arange(3)) % 3]
        mid = 0.5 * (cur[:, 0] + cur[:, 1])
        first = np.stack([cur[:, 0], mid, cur[:, 2]], axis=1)
        second = np.stack([mid, cur[:, 1], cur[:, 2]], axis=1)
        cur = np.concatenate([first, second])
        cur_i = np.concatenate([cur_i, cur_i])
    if len(cur):
        done_p.append(cur)
        done_i.append(cur_i)
    return np.concatenate(done_p), np.concatenate(done_i)


class PrimitiveGrid:
    """Grade uniforme sobre segmentos (P, 2, 3) ou triângulos (P, 3, 3).

    ``query`` devolve, para cada ponto, a distância ao primitivo mais próximo, o
    índice desse primitivo (no array passado ao construtor) e o ponto mais
    próximo. A busca olha as 27 células em volta do ponto; quando isso não
    garante o resultado, repete numa grade 4 vezes mais grossa, até cobrir tudo.
    """

    def __init__(self, prims: np.ndarray, cell: float | None = None, target_cells: int = 200_000) -> None:
        prims = np.asarray(prims, dtype=np.float64)
        if prims.ndim != 3 or prims.shape[1] not in (2, 3) or prims.shape[2] != 3:
            raise ValueError("prims deve ter forma (P, 2, 3) ou (P, 3, 3)")
        self.kind = prims.shape[1]
        self.num_input = len(prims)
        self.empty = len(prims) == 0
        self._levels: dict[int, tuple] = {}
        if self.empty:
            return
        flat = prims.reshape(-1, 3)
        self.lo = flat.min(axis=0)
        self.hi = flat.max(axis=0)
        extent = self.hi - self.lo
        self.diag = float(np.linalg.norm(extent))
        if cell is None:
            cell = self._auto_cell(prims, target_cells)
        floor = max(self.diag * 1e-5, float(extent.max()) / 1024.0, 1e-300)
        self.cell0 = float(max(cell, floor))
        parents = np.arange(len(prims), dtype=np.int64)
        if self.kind == 2:
            self.prims, self.parents = _split_segments(prims, parents, 2.0 * self.cell0)
        else:
            self.prims, self.parents = _split_triangles(prims, parents, 2.0 * self.cell0)
        # esfera envolvente de cada pedaço: descarta candidatos sem a conta exata
        self.centers = self.prims.mean(axis=1)
        self.radii = np.linalg.norm(self.prims - self.centers[:, None, :], axis=2).max(axis=1)

    def _auto_cell(self, prims: np.ndarray, target_cells: int) -> float:
        """Tamanho de célula que deixa poucos primitivos por célula ocupada."""
        count = len(prims)
        if self.kind == 2:
            total = float(np.linalg.norm(prims[:, 1] - prims[:, 0], axis=1).sum())
            cells = int(np.clip(count, 16, target_cells))
            size = 2.0 * total / cells
        else:
            cross = np.cross(prims[:, 1] - prims[:, 0], prims[:, 2] - prims[:, 0])
            total = 0.5 * float(np.linalg.norm(cross, axis=1).sum())
            cells = int(np.clip(count, 64, target_cells))
            size = float(np.sqrt(total / cells))
        if not np.isfinite(size) or size <= 0:
            size = self.diag / 32.0 if self.diag > 0 else 1.0
        return size

    def _level(self, level: int) -> tuple:
        """Constrói (uma vez) a tabela célula -> primitivos do nível pedido."""
        if level in self._levels:
            return self._levels[level]
        cell = self.cell0 * (4.0**level)
        origin = self.lo - 0.5 * cell
        plo = self.prims.min(axis=1)
        phi = self.prims.max(axis=1)
        ilo = np.floor((plo - origin) / cell).astype(np.int64)
        ihi = np.floor((phi - origin) / cell).astype(np.int64)
        dims = ihi.max(axis=0) + 1
        span = ihi - ilo + 1
        per_prim = span.prod(axis=1)
        key_parts: list[np.ndarray] = []
        pid_parts: list[np.ndarray] = []
        for s in range(0, len(per_prim), _BUILD_CHUNK):
            n = per_prim[s : s + _BUILD_CHUNK]
            total = int(n.sum())
            pid = np.repeat(np.arange(s, s + len(n), dtype=np.int64), n)
            local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
            sp = span[pid]
            cz = local % sp[:, 2]
            cy = (local // sp[:, 2]) % sp[:, 1]
            cx = local // (sp[:, 2] * sp[:, 1])
            base = ilo[pid]
            key = ((base[:, 0] + cx) * dims[1] + (base[:, 1] + cy)) * dims[2] + (base[:, 2] + cz)
            key_parts.append(key)
            pid_parts.append(pid.astype(np.int32))
        keys = np.concatenate(key_parts)
        pids = np.concatenate(pid_parts)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        pids = pids[order]
        ukeys, starts = np.unique(keys, return_index=True)
        counts = np.diff(np.append(starts, len(keys)))
        self._levels[level] = (cell, origin, dims, ukeys, starts, counts, pids)
        return self._levels[level]

    def _dist2(self, pts: np.ndarray, piece: np.ndarray) -> np.ndarray:
        """Distância ao quadrado de cada ponto ao pedaço de mesmo índice."""
        pr = self.prims[piece]
        if self.kind == 2:
            return closest_on_segments(pts, pr[:, 0], pr[:, 1])[0]
        return closest_on_triangles(pts, pr[:, 0], pr[:, 1], pr[:, 2])[0]

    def _nearest27(self, level: int, q: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Melhor candidato nas 27 células em volta de cada ponto.

        Devolve distância ao quadrado, pedaço e quantos pontos foram processados
        antes de o orçamento de pares acabar.
        """
        cell, origin, dims, ukeys, starts, counts, pids = self._level(level)
        m = len(q)
        out_d = np.full(m, np.inf)
        out_p = np.full(m, -1, dtype=np.int64)
        used = 0
        last = len(ukeys) - 1
        for b0 in range(0, m, _POINT_BLOCK):
            qb = q[b0 : b0 + _POINT_BLOCK]
            ic = np.floor((qb - origin) / cell).astype(np.int64)
            c = ic[:, None, :] + _OFFSETS[None, :, :]
            valid = ((c >= 0) & (c < dims)).all(axis=2)
            key = (c[..., 0] * dims[1] + c[..., 1]) * dims[2] + c[..., 2]
            pos = np.minimum(np.searchsorted(ukeys, key), last)
            found = valid & (ukeys[pos] == key)
            cnt = np.where(found, counts[pos], 0)
            st = starts[pos]
            per_pt = cnt.sum(axis=1)
            cum = np.cumsum(per_pt)
            s = 0
            nb = len(qb)
            while s < nb:
                base = int(cum[s - 1]) if s else 0
                room = budget - used
                e = int(np.searchsorted(cum, base + min(_PAIR_CHUNK, max(room, 0)), side="right"))
                if e <= s:
                    # o próximo ponto sozinho passa do limite: só segue se couber
                    # no orçamento (ou se for o primeiro, para sempre haver avanço)
                    if used > 0 and int(per_pt[s]) > room:
                        return out_d, out_p, b0 + s
                    e = s + 1
                npairs = int(cum[e - 1]) - base
                if npairs:
                    used += npairs
                    n = cnt[s:e].reshape(-1)
                    stt = st[s:e].reshape(-1)
                    nz = n > 0
                    n = n[nz]
                    stt = stt[nz]
                    flat = np.arange(npairs, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n) + np.repeat(stt, n)
                    cand = pids[flat].astype(np.int64)
                    pp = per_pt[s:e]
                    grp = np.repeat(np.arange(s, e, dtype=np.int64), pp)
                    has = pp > 0
                    gst = (np.cumsum(pp) - pp)[has]
                    # filtro barato: a distância ao centro do pedaço é um limite
                    # superior; quem não pode ganhar dele nem entra na conta exata
                    pts_g = qb[grp]
                    to_center = np.linalg.norm(pts_g - self.centers[cand], axis=1)
                    upper = np.minimum.reduceat(to_center, gst)
                    keep = (to_center - self.radii[cand]) <= np.repeat(upper, pp[has])
                    kept = np.add.reduceat(keep.astype(np.int64), gst)
                    cand = cand[keep]
                    grp = grp[keep]
                    d2 = self._dist2(pts_g[keep], cand)
                    best = np.minimum.reduceat(d2, np.cumsum(kept) - kept)
                    hit = np.flatnonzero(d2 == np.repeat(best, kept))
                    _, first = np.unique(grp[hit], return_index=True)
                    tgt = b0 + s + np.flatnonzero(has)
                    out_d[tgt] = best
                    out_p[tgt] = cand[hit[first]]
                s = e
        return out_d, out_p, m

    def query(
        self,
        points: np.ndarray,
        max_dist: float | None = None,
        pair_budget: int = 60_000_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Primitivo mais próximo de cada ponto.

        Devolve ``(dist, index, closest, exact)``. Com ``max_dist``, pontos sem
        nenhum primitivo a essa distância ficam com ``dist = inf`` e ``index =
        -1``. ``exact`` é falso só quando o orçamento de pares acabou em pontos
        muito distantes; nesse caso ``dist`` é um limite, não o valor exato.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        m = len(pts)
        d2 = np.full(m, np.inf)
        piece = np.full(m, -1, dtype=np.int64)
        exact = np.zeros(m, dtype=bool)
        closest = np.full((m, 3), np.nan)
        if self.empty or m == 0:
            return np.sqrt(d2), piece, closest, exact

        union = np.linalg.norm(np.maximum(self.hi, pts.max(axis=0)) - np.minimum(self.lo, pts.min(axis=0)))
        pending = np.arange(m, dtype=np.int64)
        level = 0
        while pending.size:
            cell = self.cell0 * (4.0**level)
            dd, pp, done = self._nearest27(level, pts[pending], pair_budget)
            better = dd < d2[pending]
            d2[pending[better]] = dd[better]
            piece[pending[better]] = pp[better]
            covers_all = cell >= union
            skipped = pending[done:]
            pending = pending[:done]
            ok = (d2[pending] <= cell * cell) | covers_all
            exact[pending[ok]] = True
            pending = pending[~ok]
            if skipped.size:
                # orçamento esgotado: fica o melhor valor conhecido ou o limite inferior
                lower = (cell / 4.0) ** 2 if level else 0.0
                d2[skipped] = np.where(np.isfinite(d2[skipped]), d2[skipped], lower)
            if covers_all or (max_dist is not None and cell >= max_dist):
                break
            level += 1

        if max_dist is not None:
            far = ~(d2 <= max_dist * max_dist)
            d2[far] = np.inf
            piece[far] = -1
            exact[far] = True
        has = piece >= 0
        if has.any():
            pr = self.prims[piece[has]]
            if self.kind == 2:
                closest[has] = closest_on_segments(pts[has], pr[:, 0], pr[:, 1])[1]
            else:
                closest[has] = closest_on_triangles(pts[has], pr[:, 0], pr[:, 1], pr[:, 2])[1]
        index = np.where(has, self.parents[np.maximum(piece, 0)], -1)
        return np.sqrt(d2), index, closest, exact


def point_grid(points: np.ndarray, cell: float | None = None) -> PrimitiveGrid:
    """Grade de pontos soltos (segmentos de comprimento zero) para "vértice mais próximo"."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if cell is None and len(pts):
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        cell = diag / max(1.0, np.sqrt(len(pts))) if diag > 0 else 1.0
    return PrimitiveGrid(np.stack([pts, pts], axis=1), cell=cell)


def sample_triangles(
    tri_points: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Amostras proporcionais à área (estratificadas) sobre triângulos (T, 3, 3)."""
    tri_points = np.asarray(tri_points, dtype=np.float64)
    if len(tri_points) == 0 or count <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    a, b, c = tri_points[:, 0], tri_points[:, 1], tri_points[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cum = np.cumsum(area)
    total = float(cum[-1])
    if total <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    u = (np.arange(count) + rng.random(count)) / count * total
    tri = np.minimum(np.searchsorted(cum, u, side="right"), len(area) - 1)
    r1 = np.sqrt(rng.random(count))[:, None]
    r2 = rng.random(count)[:, None]
    pts = (1.0 - r1) * a[tri] + r1 * (1.0 - r2) * b[tri] + r1 * r2 * c[tri]
    return pts, tri


def sample_segments(
    a: np.ndarray, b: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amostras regulares ao longo de segmentos.

    Devolve os pontos, o segmento de cada ponto e o comprimento que cada ponto
    representa (para somar cobertura em unidades de comprimento).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(a) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0)
    length = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(length / max(spacing, 1e-300))).astype(np.int64)
    seg = np.repeat(np.arange(len(n), dtype=np.int64), n)
    local = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    t = ((local + 0.5) / n[seg])[:, None]
    pts = a[seg] + (b - a)[seg] * t
    return pts, seg, length[seg] / n[seg]
