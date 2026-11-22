"""Adaptador do Quad Remesher (Exoside) dentro do Maya.

AVISO DE LICENÇA: a licença do Quad Remesher cobre o uso **manual** do plugin
por quem a comprou; ela não cobre uso como SDK nem processo automatizado. Este
adaptador existe só para o dono da licença disparar o plugin na própria
máquina, um modelo por vez, durante a comparação da Fase 0. Não rode em lote,
não distribua e confirme com a Exoside antes de qualquer uso além disso (ver
seção 14 do plano).

Há dois caminhos:

1. **Manual (recomendado, sempre dentro da licença)**: ``prepare_manual``
   coloca o modelo na cena com as creases como hard edges e devolve o número
   de quads a digitar; você roda o Quad Remesher pela janela dele;
   ``collect_manual`` grava o resultado em OBJ para o ``run_phase0`` medir.
2. **Chamada pelo script** (``QuadRemesherEngine.remesh``): dispara o plugin
   pelo módulo Python dele. Os nomes abaixo NÃO foram verificados nesta sessão
   (o site da Exoside estava inacessível e o plugin não é público), por isso
   são todos configuráveis. Para conferir, abra os ``.py`` da pasta
   ``Contents/scripts`` do plugin instalado
   (``C:/ProgramData/Autodesk/ApplicationPlugins/...``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
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
    EngineTimeout,
    EngineUnavailable,
    ProgressFn,
    RunInfo,
    as_feature_pairs,
)

# TODO verificar: nomes do módulo, da classe e do método que dispara o remesh
# no plugin do Maya. São palpites; troque pelos nomes reais do plugin instalado.
MODULE_NAME = os.environ.get("TOPTOP_QR_MODULE", "QuadRemesher")
CLASS_NAME = os.environ.get("TOPTOP_QR_CLASS", "QuadRemesher")
METHOD_NAME = os.environ.get("TOPTOP_QR_METHOD", "remeshIt")

# TODO verificar: nomes das opções do plugin (optionVars). Deixe vazio para
# usar o que estiver na janela do Quad Remesher. Para descobrir os nomes, mude
# um valor na janela e rode no Script Editor:
#     [v for v in cmds.optionVar(list=True) if "QR" in v or "uad" in v]
# Os valores podem usar "{target_quads}", que é trocado pelo número pedido.
OPTION_VARS: dict[str, Any] = {}

MANUAL_PREFIX = "toptopQR_"


def _mesh_transforms() -> set[str]:
    """Transforms de todas as malhas da cena."""
    from maya import cmds

    shapes = cmds.ls(type="mesh", long=True, noIntermediate=True) or []
    if not shapes:
        return set()
    return set(cmds.listRelatives(shapes, parent=True, fullPath=True) or [])


def prepare_manual(mesh: Mesh, feature_edges: "FeatureSet | np.ndarray | None", target_quads: int, name: str) -> str:
    """Põe o modelo na cena pronto para o Quad Remesher manual; devolve o transform.

    As creases viram hard edges (ligue "Use Normals Splitting"/hard edges na
    janela do plugin) e o objeto fica selecionado.
    """
    from maya import cmds

    from .. import maya_bridge

    transform = maya_bridge.create_mesh(mesh, MANUAL_PREFIX + name)
    maya_bridge.set_hard_edges(transform, as_feature_pairs(feature_edges))
    cmds.select(transform, replace=True)
    print(f"[TopTop] {name}: rode o Quad Remesher com Target Quad Count = {int(target_quads)} "
          "e hard edges ligadas; depois selecione o resultado e chame collect_manual().")
    return transform


def collect_manual(path: str, result: str | None = None) -> Mesh:
    """Grava em OBJ o resultado do Quad Remesher (o objeto dado ou o selecionado)."""
    from .. import maya_bridge
    from ..io.obj import write_obj

    node = result or maya_bridge.selected_mesh()
    if node is None:
        raise ValueError("selecione a malha gerada pelo Quad Remesher")
    mesh = maya_bridge.mesh_from_maya(node, world=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_obj(path, mesh, comment="resultado manual do Quad Remesher")
    return mesh


class QuadRemesherEngine(Engine):
    """Dispara o Quad Remesher pelo script do plugin (uso pessoal, um modelo por vez).

    Leia o aviso de licença no topo deste módulo antes de usar.
    """

    name = "quadremesher"
    label = "Quad Remesher"
    requires_maya = True

    def __init__(
        self,
        timeout_s: float = 600.0,
        module_name: str = MODULE_NAME,
        class_name: str = CLASS_NAME,
        method_name: str = METHOD_NAME,
        option_vars: dict[str, Any] | None = None,
        allow_batch: bool = False,
    ) -> None:
        super().__init__(timeout_s)
        self.module_name = module_name
        self.class_name = class_name
        self.method_name = method_name
        self.option_vars = dict(OPTION_VARS if option_vars is None else option_vars)
        self.allow_batch = allow_batch

    def availability(self) -> tuple[bool, str]:
        """Só no Maya com interface e com o módulo do plugin importável."""
        try:
            from maya import cmds

            batch = bool(cmds.about(batch=True))
        except ImportError:
            return False, "o Quad Remesher só roda dentro do Maya"
        except Exception:  # mayapy antes do maya.standalone.initialize(): sem comandos
            batch = True
        if batch and not self.allow_batch:
            return False, (
                "o Quad Remesher é de uso manual: rode pela janela do Maya e grave o resultado com "
                "toptop.engine.quadremesher.collect_manual (veja HANDOFF_NEXT.md)"
            )
        try:
            found = importlib.util.find_spec(self.module_name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return False, f"módulo '{self.module_name}' do Quad Remesher não encontrado (plugin instalado?)"
        return True, f"módulo {self.module_name}"

    def _entry_point(self) -> Any:
        """Função do plugin que dispara o remesh do objeto selecionado."""
        module = importlib.import_module(self.module_name)
        owner = getattr(module, self.class_name, None)
        target = None
        if owner is not None:
            holder = owner() if isinstance(owner, type) else owner
            target = getattr(holder, self.method_name, None)
        if target is None:
            target = getattr(module, self.method_name, None)
        if not callable(target):
            names = sorted(n for n in dir(module) if "remesh" in n.lower() or "retopo" in n.lower())
            raise EngineError(
                f"não achei {self.module_name}.{self.class_name}.{self.method_name}. "
                f"Nomes parecidos no módulo: {names or 'nenhum'}. Ajuste MODULE_NAME/CLASS_NAME/"
                "METHOD_NAME em toptop/engine/quadremesher.py (ou as variáveis TOPTOP_QR_*)."
            )
        return target

    def remesh(
        self,
        mesh: Mesh,
        feature_edges: "FeatureSet | np.ndarray | None",
        density: "Density | int",
        *,
        cancel: CancelFn | None = None,
        progress: ProgressFn | None = None,
    ) -> Mesh:
        """Põe a malha na cena, dispara o plugin e lê a malha nova que aparecer."""
        ok, reason = self.availability()
        if not ok:
            raise EngineUnavailable(reason)
        from maya import cmds

        from .. import maya_bridge

        density = Density.coerce(density)
        info = RunInfo()
        self.last_run = info
        start = time.perf_counter()
        temp = None
        created: set[str] = set()
        try:
            temp = prepare_manual(mesh, feature_edges, density.target_quads, "input")
            for key, value in self.option_vars.items():
                if isinstance(value, str):
                    value = value.format(target_quads=density.target_quads)
                    cmds.optionVar(stringValue=(key, value))
                elif isinstance(value, bool) or isinstance(value, int):
                    cmds.optionVar(intValue=(key, int(value)))
                else:
                    cmds.optionVar(floatValue=(key, float(value)))
            before = _mesh_transforms()
            entry = self._entry_point()
            if progress is not None:
                progress(0.1, "Quad Remesher: calculando")
            entry()
            deadline = time.monotonic() + self.timeout_s
            while True:  # o plugin pode terminar depois de a chamada voltar
                created = _mesh_transforms() - before
                if created:
                    break
                if cancel is not None and cancel():
                    raise EngineCancelled("execução cancelada pelo usuário")
                if time.monotonic() > deadline:
                    raise EngineTimeout(f"o Quad Remesher não criou malha em {self.timeout_s:.0f} s")
                cmds.refresh()
                time.sleep(0.25)
            result = maya_bridge.mesh_from_maya(sorted(created)[0], world=True)
        except EngineError:
            raise
        except Exception as error:
            raise EngineError(f"falha ao chamar o Quad Remesher: {error}", log=str(error)) from error
        finally:
            info.seconds = time.perf_counter() - start
            for node in [temp] + sorted(created):
                if node is not None and cmds.objExists(node):
                    cmds.delete(node)
        if result.num_faces == 0:
            raise EngineError("o Quad Remesher devolveu uma malha vazia")
        if progress is not None:
            progress(1.0, "Quad Remesher concluído")
        return Mesh(result.points, result.counts, result.connects, mesh.name)
