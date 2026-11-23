# TopTop

Retopologia automática de hard surface para **Maya 2025+ (Windows)**: 100% quads,
e cada aresta marcada com crease vira um edge loop. A entrada são booleanas e
malhas de CAD (até ~500 mil faces); a saída é sempre um objeto novo.

O motor de remesh é uma **caixa-preta** atrás de uma interface única; toda a
inteligência (creases, limpeza, reprojeção, métricas) fica em Python.

> Estado: **Fase 0** (comparar motores antes de usar a ferramenta de verdade).
> O passo a passo para o Windows está em [`HANDOFF_NEXT.md`](HANDOFF_NEXT.md).

## Estrutura

| Caminho | O que faz |
| --- | --- |
| `toptop/core/mesh.py` | Malha em arrays NumPy, topologia, triangulação, cascas, subdivisão linear |
| `toptop/core/geom.py` | Distância exata ponto–triângulo/segmento com grade uniforme, amostragem |
| `toptop/core/quality.py` | Métricas da seção 11, metas, relatório, mapa de calor, comparação com referência |
| `toptop/core/creases.py` | Dois níveis de crease, sugestão por ângulo, "Verificar entrada" (+ camada Maya) |
| `toptop/core/cleanup.py` | Soldar, faces degeneradas, non-manifold, bordas como feature, cascas, redução |
| `toptop/core/reproject.py` | Vértices novos para a superfície, loops para a crease, cantos exatos |
| `toptop/core/pipeline.py` | Os 9 passos do plano, densidade, hard edges, relatório (+ camada Maya) |
| `toptop/core/phase0.py`, `render.py` | Lógica da Fase 0 (linhas, CSV/JSON, portão) e capturas de wireframe em PNG |
| `toptop/io/` | OBJ e arquivo de arestas de feature (`.feat`) |
| `toptop/engine/` | Interface dos motores e os adaptadores (QuadWild, Retopologize, Quad Remesher) |
| `toptop/maya_bridge.py` | Malha do Maya <-> arrays, arestas, hard edges, material, desfazer |
| `toptop/ui/window.py`, `toptop/shelf.py` | Janela PySide6 e botão na shelf |
| `tests/unit/` | Testes pytest que rodam **sem Maya** |
| `tests/make_test_models.py` | Gera os 10 modelos de teste no Maya |
| `tests/run_phase0.py` | Fase 0: modelo x motor, mede e grava CSV/JSON + wireframes |
| `tests/run_headless.py` | Autoverificação: pipeline completo + comparação com a referência (piora > 10% reprova) |

## No Maya

Instalar o botão (uma vez, no Script Editor, aba Python):

```python
import sys
TOPTOP = r"C:/dev/Toptop"          # a pasta deste repositório
if TOPTOP not in sys.path:
    sys.path.insert(0, TOPTOP)
import toptop.shelf
toptop.shelf.install()
```

Pelo script, sem a janela (a primeira linha é dispensável se o repositório já
estiver no `sys.path`, pelo botão ou pelo `userSetup.py`):

```python
import sys; sys.path.insert(0, r"C:/dev/Toptop")
from toptop.core.pipeline import PipelineOptions, maya_retopologize
novo, resultado = maya_retopologize("pCube1", PipelineOptions(mode="game", detail=5))
print(resultado.report.passed)
```

O QuadWild precisa estar em `third_party/quadwild-bimdf/` (ou na variável
`TOPTOP_QUADWILD_DIR`); veja `HANDOFF_NEXT.md`.

## Testes sem Maya

```
pip install numpy pytest          # PySide6 opcional: sem ele os testes da janela são pulados
pytest
```

Variáveis opcionais: `TOPTOP_QUADWILD_DIR` (roda também os testes com o motor
real) e `TOPTOP_MAYA_CMDS_PKL` (confere os flags do Maya usados na shelf).

## Métricas (resumo)

Tudo em `toptop.core.quality.evaluate(malha_nova, malha_original, features)`:

| Métrica | Como é medida | Meta inicial |
| --- | --- | --- |
| Quads | faces de 4 lados / total | 100% |
| Cobertura das creases | comprimento de crease com um edge loop alinhado em cima | >= 99% |
| Fidelidade das quinas | maior distância (vértices dos loops -> crease; cantos -> vértice mais próximo) / diagonal | <= 0,1% |
| Desvio da forma | Hausdorff por amostragem nos dois sentidos / diagonal | <= 0,5% |
| Proporção dos quads | lado maior / lado menor | mediana <= 1,5; 95% <= 3 |
| Polos | vértices internos com valência != 4 | <= 5%; nenhum sobre crease forte fora dos cantos |
| Tempo | segundos do motor | <= 2 min por 100 mil triângulos |

Licença do código deste repositório: uso pessoal. O QuadWild Bi-MDF é GPL3 e roda
como executável separado; ele não é distribuído aqui.
