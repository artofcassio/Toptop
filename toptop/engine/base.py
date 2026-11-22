"""Interface comum dos motores de remesh.

Um motor é uma caixa-preta com uma única operação::

    malha_quads = motor.remesh(malha, arestas_de_feature, densidade)

Trocar de motor é trocar de adaptador. Este módulo é Python puro: os
adaptadores que precisam do Maya só o importam dentro de ``remesh``.
"""

from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from ..core.creases import FeatureSet
from ..core.mesh import Mesh

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class EngineError(RuntimeError):
    """Falha do motor. ``workdir`` aponta a pasta mantida para diagnóstico."""

    def __init__(self, message: str, workdir: str | None = None, log: str = "") -> None:
        if workdir:
            message = f"{message}\nPasta de diagnóstico mantida em: {workdir}"
        super().__init__(message)
        self.workdir = workdir
        self.log = log


class EngineUnavailable(EngineError):
    """O motor não está instalado ou não pode rodar neste ambiente."""


class EngineTimeout(EngineError):
    """O motor passou do tempo limite e foi encerrado."""


class EngineCancelled(EngineError):
    """O usuário cancelou a execução."""


@dataclass
class Density:
    """Densidade pedida ao motor: número de quads desejado e a folga aceita."""

    target_quads: int
    tolerance: float = 0.15

    @classmethod
    def coerce(cls, value: "Density | int | float") -> "Density":
        """Aceita um ``Density`` ou só o número de quads."""
        if isinstance(value, Density):
            return value
        return cls(target_quads=max(1, int(round(float(value)))))

    def accepts(self, quads: int) -> bool:
        """Verdadeiro se ``quads`` está dentro da folga."""
        return abs(quads - self.target_quads) <= self.tolerance * self.target_quads


@dataclass
class RunInfo:
    """O que aconteceu na última execução de um motor."""

    seconds: float = 0.0
    workdir: str | None = None
    log: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def as_feature_pairs(features: "FeatureSet | np.ndarray | Sequence | None") -> np.ndarray:
    """Arestas de feature como array (K, 2) de índices de vértice."""
    if features is None:
        return np.zeros((0, 2), dtype=np.int64)
    if isinstance(features, FeatureSet):
        return features.edges
    return FeatureSet(np.asarray(features, dtype=np.int64).reshape(-1, 2)).edges


@dataclass
class ScaleTransform:
    """Normalização de escala: leva a malha para um tamanho padrão e desfaz depois."""

    center: np.ndarray
    factor: float

    @classmethod
    def for_mesh(cls, mesh: Mesh, target_diagonal: float = 100.0) -> "ScaleTransform":
        """Centro da caixa na origem e diagonal igual a ``target_diagonal``."""
        lo, hi = mesh.bbox()
        diagonal = float(np.linalg.norm(hi - lo))
        factor = target_diagonal / diagonal if diagonal > 0 else 1.0
        return cls(center=0.5 * (lo + hi), factor=factor)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Coordenadas originais -> normalizadas."""
        return (np.asarray(points, dtype=np.float64) - self.center) * self.factor

    def invert(self, points: np.ndarray) -> np.ndarray:
        """Coordenadas normalizadas -> originais."""
        return np.asarray(points, dtype=np.float64) / self.factor + self.center


class Engine(ABC):
    """Base de todos os adaptadores de motor."""

    #: identificador curto (usado em linha de comando, CSV e nomes de pasta)
    name: ClassVar[str] = ""
    #: nome para exibir
    label: ClassVar[str] = ""
    #: verdadeiro se o motor só roda dentro do Maya
    requires_maya: ClassVar[bool] = False

    def __init__(self, timeout_s: float = 600.0) -> None:
        self.timeout_s = float(timeout_s)
        self.last_run = RunInfo()

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        """(disponível?, explicação). Nunca levanta exceção."""

    def is_available(self) -> bool:
        """Atalho para ``availability()[0]``."""
        return self.availability()[0]

    @abstractmethod
    def remesh(
        self,
        mesh: Mesh,
        feature_edges: "FeatureSet | np.ndarray | None",
        density: "Density | int",
        *,
        cancel: CancelFn | None = None,
        progress: ProgressFn | None = None,
    ) -> Mesh:
        """Devolve a malha de quads. ``mesh`` nunca é alterada.

        ``feature_edges`` são pares de índices de vértice de ``mesh`` (arestas
        reais dela). Levanta ``EngineError`` (ou uma subclasse) em falha,
        tempo limite ou cancelamento; os detalhes ficam em ``last_run``.
        """


def run_process(
    command: Sequence[str],
    cwd: str,
    log_path: str,
    timeout_s: float,
    cancel: CancelFn | None = None,
    poll_s: float = 0.1,
    env: dict[str, str] | None = None,
) -> int:
    """Roda um processo filho com tempo limite e cancelamento.

    A saída (stdout + stderr) vai para ``log_path``. Devolve o código de saída;
    levanta ``EngineTimeout`` ou ``EngineCancelled`` depois de matar o processo.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    deadline = time.monotonic() + timeout_s
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(command) + "\n").encode("utf-8", errors="replace"))
        log.flush()
        try:
            proc = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
                env=env,
            )
        except OSError as error:
            raise EngineError(f"não foi possível iniciar {command[0]}: {error}") from error
        try:
            while True:
                code = proc.poll()
                if code is not None:
                    return code
                if cancel is not None and cancel():
                    raise EngineCancelled("execução cancelada pelo usuário")
                if time.monotonic() > deadline:
                    raise EngineTimeout(f"tempo limite de {timeout_s:.0f} s excedido")
                time.sleep(poll_s)
        finally:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass


def read_log_tail(log_path: str, max_chars: int = 4000) -> str:
    """Fim do arquivo de log (para mensagens de erro)."""
    try:
        with open(log_path, "rb") as handle:
            data = handle.read()
    except OSError:
        return ""
    return data[-max_chars:].decode("utf-8", errors="replace")
