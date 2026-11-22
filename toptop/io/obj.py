"""Leitura e escrita de OBJ (só geometria: vértices e faces).

O leitor aceita o que os exportadores reais produzem: ``v/vt/vn`` em qualquer
combinação, índices negativos (relativos), linhas continuadas com ``\\``,
CRLF, cores depois do ``xyz``, vários ``o``/``g`` no mesmo arquivo (viram uma
malha só) e faces com qualquer número de lados. O que não é geometria é
ignorado. A ordem e os índices dos vértices são preservados, o que mantém
válido o arquivo de features que acompanha o OBJ.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from ..core.mesh import Mesh


class ObjError(ValueError):
    """Arquivo OBJ inválido (a mensagem traz o número da linha)."""


def _logical_lines(text: str) -> Iterable[tuple[int, str]]:
    """Linhas do arquivo já com as continuações (``\\`` no fim) emendadas."""
    pending = ""
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not pending:
            start = number
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        yield start, pending + line
        pending = ""
    if pending:
        yield start, pending


def parse_obj(text: str, name: str = "") -> Mesh:
    """Converte o texto de um OBJ em ``Mesh``."""
    verts: list[tuple[float, float, float]] = []
    counts: list[int] = []
    connects: list[int] = []
    for number, line in _logical_lines(text):
        if len(line) < 2:
            continue
        head = line[0]
        if head == "v" and line[1] in " \t":
            parts = line.split()
            try:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except (IndexError, ValueError):
                raise ObjError(f"linha {number}: vértice inválido: {line!r}") from None
        elif head == "f" and line[1] in " \t":
            parts = line.split()[1:]
            if len(parts) < 3:
                continue  # face degenerada (ponto ou linha): ignorada
            total = len(verts)
            for token in parts:
                try:
                    index = int(token.split("/", 1)[0])
                except ValueError:
                    raise ObjError(f"linha {number}: índice de face inválido: {token!r}") from None
                resolved = index - 1 if index > 0 else total + index
                if index == 0 or not 0 <= resolved < total:
                    raise ObjError(f"linha {number}: índice de vértice fora do intervalo: {token!r}")
                connects.append(resolved)
            counts.append(len(parts))
    points = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    return Mesh(points, np.asarray(counts, dtype=np.int64), np.asarray(connects, dtype=np.int64), name)


def read_obj(path: str | os.PathLike) -> Mesh:
    """Lê um arquivo OBJ; o nome da malha é o nome do arquivo sem extensão."""
    with open(path, "rb") as handle:
        text = handle.read().decode("utf-8", errors="replace")
    return parse_obj(text, os.path.splitext(os.path.basename(os.fspath(path)))[0])


def format_obj(mesh: Mesh, precision: int = 12, comment: str | None = None) -> str:
    """Texto OBJ da malha (índices a partir de 1, vértices na ordem original)."""
    lines: list[str] = []
    if comment:
        lines.extend("# " + part for part in comment.splitlines())
    fmt = f"v %.{precision}g %.{precision}g %.{precision}g"
    lines.extend(fmt % (x, y, z) for x, y, z in mesh.points.tolist())
    connects = (mesh.connects + 1).tolist()
    if mesh.num_faces and int(mesh.counts.min()) == 3 and int(mesh.counts.max()) == 3:
        it = iter(connects)
        lines.extend("f %d %d %d" % tri for tri in zip(it, it, it))
    else:
        offsets = mesh.offsets.tolist()
        lines.extend(
            "f " + " ".join(map(str, connects[offsets[i] : offsets[i + 1]]))
            for i in range(mesh.num_faces)
        )
    return "\n".join(lines) + "\n"


def write_obj(
    path: str | os.PathLike, mesh: Mesh, precision: int = 12, comment: str | None = None
) -> None:
    """Grava a malha em OBJ (ASCII, quebras de linha Unix)."""
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as handle:
        handle.write(format_obj(mesh, precision, comment))
