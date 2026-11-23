"""Motores de remesh atrás de uma interface única (ver ``base.py``).

Trocar de motor é trocar de adaptador::

    from toptop.engine import get_engine
    quads = get_engine("quadwild").remesh(malha, arestas_de_feature, 5000)
"""

from __future__ import annotations

from typing import Any

from .base import Density, Engine, EngineError, EngineUnavailable

ENGINE_NAMES = ("quadwild", "maya_retopo", "quadremesher")


def get_engine(name: str, **options: Any) -> Engine:
    """Cria o adaptador pelo nome (``quadwild``, ``maya_retopo`` ou ``quadremesher``)."""
    if name == "quadwild":
        from .quadwild import QuadWildEngine

        return QuadWildEngine(**options)
    if name == "maya_retopo":
        from .maya_retopo import MayaRetopoEngine

        return MayaRetopoEngine(**options)
    if name == "quadremesher":
        from .quadremesher import QuadRemesherEngine

        return QuadRemesherEngine(**options)
    raise ValueError(f"motor desconhecido: {name!r} (opções: {', '.join(ENGINE_NAMES)})")


__all__ = ["Density", "Engine", "EngineError", "EngineUnavailable", "ENGINE_NAMES", "get_engine"]
