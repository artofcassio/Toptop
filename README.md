# TopTop

Retopologia automática de hard surface para **Maya 2025+ (Windows)**: 100% quads,
e cada aresta marcada com crease vira um edge loop. A entrada são booleanas e
malhas de CAD (até ~500 mil faces); a saída é sempre um objeto novo.

O motor de remesh é uma **caixa-preta** atrás de uma interface única; toda a
inteligência (creases, limpeza, reprojeção, métricas) fica em Python.

> Estado: **Fase 0** (comparar motores antes de construir a ferramenta).
> Veja `HANDOFF_NEXT.md` para o passo a passo no Windows.

## Estrutura

| Caminho | O que faz |
| --- | --- |
| `toptop/core/mesh.py` | Malha em arrays NumPy, topologia, triangulação, cascas |
| `toptop/core/geom.py` | Distância ponto-triângulo/segmento com grade uniforme, amostragem |
| `toptop/core/quality.py` | Métricas da seção 11, metas, relatório, comparação com referência |
| `toptop/core/creases.py` | Dois níveis de crease, sugestão por ângulo, "Verificar entrada" |
| `toptop/io/` | OBJ e arquivo de arestas de feature |
| `toptop/engine/` | Interface dos motores e os adaptadores (QuadWild, Retopologize, Quad Remesher) |
| `tests/unit/` | Testes pytest que rodam **sem Maya** |
| `tests/make_test_models.py` | Gera os 10 modelos de teste no Maya |
| `tests/run_phase0.py` | Roda modelo x motor, mede e grava CSV/JSON + wireframes |

## Testes sem Maya

```
pip install numpy pytest
pytest
```

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
