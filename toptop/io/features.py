"""Arquivo de arestas de feature (``.feat``), o par do OBJ.

Formato de texto, uma aresta por linha::

    # toptop-features 1
    # v0 v1 crease   (índices de vértice a partir de 0, os mesmos do OBJ)
    12 13 3
    40 41 1

``crease`` é opcional (sem ele vale o crease forte padrão). O nível (forte ou
guia) sai da comparação com o limiar na hora de usar, não fica gravado.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.creases import STRONG_VALUE, FeatureSet

HEADER = "# toptop-features 1"
EXTENSION = ".feat"


class FeatureFileError(ValueError):
    """Arquivo de features inválido (a mensagem traz o número da linha)."""


def features_path(obj_path: str | os.PathLike) -> str:
    """Caminho do arquivo de features que acompanha um OBJ."""
    return os.path.splitext(os.fspath(obj_path))[0] + EXTENSION


def parse_features(text: str, num_points: int | None = None) -> FeatureSet:
    """Converte o texto de um arquivo de features em ``FeatureSet``."""
    edges: list[tuple[int, int]] = []
    crease: list[float] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) not in (2, 3):
            raise FeatureFileError(f"linha {number}: esperado 'v0 v1 [crease]': {raw!r}")
        try:
            a, b = int(parts[0]), int(parts[1])
            value = float(parts[2]) if len(parts) == 3 else STRONG_VALUE
        except ValueError:
            raise FeatureFileError(f"linha {number}: número inválido: {raw!r}") from None
        if a < 0 or b < 0 or (num_points is not None and max(a, b) >= num_points):
            raise FeatureFileError(f"linha {number}: índice de vértice fora do intervalo: {raw!r}")
        edges.append((a, b))
        crease.append(value)
    return FeatureSet(np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(crease))


def read_features(path: str | os.PathLike, num_points: int | None = None) -> FeatureSet:
    """Lê um arquivo de features; se ele não existir, devolve um conjunto vazio."""
    if not os.path.exists(path):
        return FeatureSet.empty()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse_features(handle.read(), num_points)


def format_features(features: FeatureSet, comment: str | None = None) -> str:
    """Texto do arquivo de features."""
    lines = [HEADER, "# v0 v1 crease   (indices de vertice a partir de 0, os mesmos do OBJ)"]
    if comment:
        lines.extend("# " + part for part in comment.splitlines())
    lines.extend(
        "%d %d %.6g" % (a, b, c)
        for (a, b), c in zip(features.edges.tolist(), features.crease.tolist())
    )
    return "\n".join(lines) + "\n"


def write_features(path: str | os.PathLike, features: FeatureSet, comment: str | None = None) -> None:
    """Grava as arestas de feature."""
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as handle:
        handle.write(format_features(features, comment))
