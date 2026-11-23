"""Adaptador do Retopologize do Maya (comando ``polyRetopo``).

As creases chegam ao motor como **hard edges**: a malha temporária fica toda
suave, só as arestas de feature ficam duras, e o comando roda com
``preserveHardEdges``. Nada da malha original é tocado.

Flags usados (nomes conferidos para o Maya 2024/2025): ``targetFaceCount``,
``targetFaceCountTolerance``, ``preserveHardEdges``, ``edgesByAngle``,
``preprocessMesh``, ``faceUniformity``, ``anisotropy``, ``topologyRegularity``,
``replaceOriginal`` e ``constructionHistory``. O Maya 2022/2023 não tem
``edgesByAngle`` nem ``preprocessMesh``; por isso a lista de flags é filtrada
pelo que o ``help polyRetopo`` da versão em uso declara.

Limitações: o comando não pode ser cancelado no meio e o tempo limite não tem
como ser imposto de dentro do processo. ``tests/run_phase0.py`` roda cada
execução num ``mayapy`` separado justamente para poder matá-la. O Maya avisa
que muitas hard edges deixam o cálculo muito lento.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from ..core.creases import FeatureSet
from ..core.mesh import Mesh
from .base import (
    CancelFn,
    Density,
    Engine,
    EngineCancelled,
    EngineError,
    EngineUnavailable,
    ProgressFn,
    RunInfo,
    as_feature_pairs,
)


def supported_flags(help_text: str) -> set[str]:
    """Nomes longos dos flags listados por ``cmds.help('polyRetopo')``."""
    return {long for _, long in re.findall(r"-(\w+)\s+-(\w+)", help_text or "")}


class MayaRetopoEngine(Engine):
    """Retopologize do Maya; só roda dentro do Maya/mayapy."""

    name = "maya_retopo"
    label = "Maya Retopologize"
    requires_maya = True

    def __init__(
        self,
        timeout_s: float = 600.0,
        preserve_hard_edges: bool = True,
        preprocess_mesh: bool = True,
        face_uniformity: float = 0.0,
        anisotropy: float | None = None,
        topology_regularity: float | None = None,
        tolerance_percent: int | None = None,
        extra_flags: dict[str, Any] | None = None,
    ) -> None:
        """Parâmetros.

        ``face_uniformity``: 0 deixa faces menores onde a curvatura é alta, 1
        deixa tudo do mesmo tamanho. ``tolerance_percent``: folga do número de
        faces (padrão: a folga da ``Density`` pedida). ``anisotropy`` e
        ``topology_regularity`` ficam no padrão do Maya quando são ``None``.
        ``extra_flags`` entra por último e pode sobrescrever qualquer flag.
        """
        super().__init__(timeout_s)
        self.preserve_hard_edges = preserve_hard_edges
        self.preprocess_mesh = preprocess_mesh
        self.face_uniformity = face_uniformity
        self.anisotropy = anisotropy
        self.topology_regularity = topology_regularity
        self.tolerance_percent = tolerance_percent
        self.extra_flags = dict(extra_flags or {})

    def availability(self) -> tuple[bool, str]:
        """Disponível se estamos dentro do Maya e o comando existe."""
        try:
            from maya import cmds
        except ImportError:
            return False, "o Retopologize só roda dentro do Maya (use o mayapy)"
        if not hasattr(cmds, "about"):
            return False, "o Maya não foi iniciado (falta maya.standalone.initialize())"
        if not hasattr(cmds, "polyRetopo"):
            return False, "esta versão do Maya não tem o comando polyRetopo"
        return True, "comando polyRetopo"

    def build_flags(self, density: Density, has_features: bool, supported: set[str]) -> dict[str, Any]:
        """Flags do ``polyRetopo`` para esta execução, já filtrados pela versão."""
        tolerance = self.tolerance_percent
        if tolerance is None:
            tolerance = int(round(density.tolerance * 100))
        flags: dict[str, Any] = {
            "targetFaceCount": int(density.target_quads),
            "targetFaceCountTolerance": int(tolerance),
            "preserveHardEdges": bool(self.preserve_hard_edges and has_features),
            "edgesByAngle": False,
            "preprocessMesh": bool(self.preprocess_mesh),
            "faceUniformity": float(self.face_uniformity),
            "replaceOriginal": True,
            "constructionHistory": False,
        }
        if self.anisotropy is not None:
            flags["anisotropy"] = float(self.anisotropy)
        if self.topology_regularity is not None:
            flags["topologyRegularity"] = float(self.topology_regularity)
        flags.update(self.extra_flags)
        if supported:
            flags = {key: value for key, value in flags.items() if key in supported}
        return flags

    def remesh(
        self,
        mesh: Mesh,
        feature_edges: "FeatureSet | np.ndarray | None",
        density: "Density | int",
        *,
        cancel: CancelFn | None = None,
        progress: ProgressFn | None = None,
    ) -> Mesh:
        """Cria uma malha temporária, roda o ``polyRetopo`` nela e lê o resultado."""
        ok, reason = self.availability()
        if not ok:
            raise EngineUnavailable(reason)
        from maya import cmds

        from .. import maya_bridge

        density = Density.coerce(density)
        pairs = as_feature_pairs(feature_edges)
        if cancel is not None and cancel():
            raise EngineCancelled("execução cancelada pelo usuário")
        try:
            supported = supported_flags(cmds.help("polyRetopo"))
        except Exception:  # sem a ajuda do comando, tenta todos os flags
            supported = set()
        flags = self.build_flags(density, len(pairs) > 0, supported)
        info = RunInfo(details={"flags": dict(flags), "hard_edges": int(len(pairs))})
        self.last_run = info

        start = time.perf_counter()
        temp = None
        roots_before = set(cmds.ls(assemblies=True, long=True) or [])
        created: list[str] = []
        try:
            if progress is not None:
                progress(0.05, "Retopologize: preparando a malha")
            temp = maya_bridge.create_mesh(mesh, "toptopRetopoInput")
            maya_bridge.set_hard_edges(temp, pairs)
            if progress is not None:
                progress(0.1, "Retopologize: calculando (não dá para cancelar)")
            cmds.polyRetopo(temp, **flags)
            created = [n for n in cmds.ls(assemblies=True, long=True) or [] if n not in roots_before and n != temp]
            result = maya_bridge.mesh_from_maya(temp, world=True)
            if _same_mesh(result, mesh):
                # replaceOriginal não pegou: a malha nova, se existir, é um objeto novo na cena
                result = self._read_created(created, mesh)
        except EngineError:
            raise
        except Exception as error:
            info.seconds = time.perf_counter() - start
            info.log = str(error)
            raise EngineError(f"o polyRetopo falhou: {error}", log=str(error)) from error
        finally:
            for node in ([temp] if temp is not None else []) + created:
                if cmds.objExists(node):
                    cmds.delete(node)
        info.seconds = time.perf_counter() - start
        if result.num_faces == 0:
            raise EngineError("o polyRetopo devolveu uma malha vazia")
        if info.seconds > self.timeout_s:
            info.details["warning"] = f"passou do tempo limite de {self.timeout_s:.0f} s"
        if progress is not None:
            progress(1.0, "Retopologize concluído")
        return Mesh(result.points, result.counts, result.connects, mesh.name)

    @staticmethod
    def _read_created(created: list[str], mesh: Mesh) -> Mesh:
        """Malha que o ``polyRetopo`` criou como objeto separado (quando não substituiu a temporária)."""
        from .. import maya_bridge

        for node in created:
            try:
                candidate = maya_bridge.mesh_from_maya(node, world=True)
            except ValueError:  # não é malha
                continue
            if candidate.num_faces and not _same_mesh(candidate, mesh):
                return candidate
        raise EngineError(
            "o polyRetopo terminou sem mudar a malha (replaceOriginal não foi aplicado e nenhum objeto "
            "novo com a malha refeita apareceu na cena)"
        )


def _same_mesh(a: Mesh, b: Mesh) -> bool:
    """Mesma topologia e mesmos pontos: o comando não mexeu na malha."""
    return (
        a.num_points == b.num_points
        and np.array_equal(a.counts, b.counts)
        and np.array_equal(a.connects, b.connects)
        and np.allclose(a.points, b.points)
    )
