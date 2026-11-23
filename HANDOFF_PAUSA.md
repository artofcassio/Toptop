# TopTop: pausa (2026-11-23)

O trabalho foi pausado a pedido. Branch: `claude/new-session-vw85ux`. O último commit
de código é `1d576c2`. Tudo está enviado e os testes estão verdes: 257 passam no pytest,
com o QuadWild real e o checador de flags do Maya ligados.

Este arquivo tem duas partes: **o que falta agora** e **o resumo de tudo que foi feito**.
As instruções completas da Fase 0 no Windows continuam em [`HANDOFF_NEXT.md`](HANDOFF_NEXT.md).

---

## 1. O que falta fazer agora

> **Atualização (sessão seguinte):** 1.1 e 1.2 feitas. Os 10 achados foram aplicados
> (o nº 1 também no código: `run_headless.py` usa o motor da referência e recusa
> comparar com outro). A camada Maya passou por uma auditoria voltada ao Maya 2026;
> o resultado e o que ainda precisa ser conferido no Maya estão na seção 1.5. Testes:
> 264 passam, com o QuadWild real e as flags do Maya 2025; as chamadas `cmds` do
> pacote inteiro e do `make_test_models.py` agora também são conferidas (2022 a 2025).

### 1.1 Corrigir a documentação (achados da última verificação, aplicados)

Um revisor independente conferiu o `HANDOFF_NEXT.md` contra o código, rodando os
comandos. A verificação foi interrompida na pausa e estes 10 achados ficaram sem aplicar:

| # | Onde | Problema | Correção |
| --- | --- | --- | --- |
| 1 | HANDOFF_NEXT.md ~242 (**importante**) | `run_headless.py` usa sempre `--engine quadwild` por padrão. Se a Fase 0 escolher outro motor, a referência é gravada e conferida com o motor errado, e o código não compara o motor gravado no `headless.json`. | Mostrar `--engine <motor escolhido>` nos dois comandos. Também vale fazer `run_headless.py` avisar ou reprovar quando o `--engine` diferir do motor da referência. |
| 2 | ~326 | O texto diz "nome", mas a chave do `models.json` é `name`. Com `"nome"`, a execução dá `KeyError`. | Escrever as chaves literais: `name`, `obj`, `features`, `target_quads`. Só `name` é obrigatória. |
| 3 | ~189 | O worker do Retopologize só é morto em `--timeout` + 600 s (20 min no padrão), não em `--timeout`. | Documentar isso, ou reduzir `WORKER_MARGIN_S` em `tests/run_phase0.py`. |
| 4 | ~97 | `set TOPTOP_QUADWILD_DIR` vale só na janela do cmd. O Maya aberto pelo menu Iniciar não enxerga a variável. | Usar `setx` ou `Maya.env`, ou o caminho mais simples: `third_party\quadwild-bimdf\`. |
| 5 | ~81 | `set`/`%MAYAPY%` só funcionam no cmd.exe. No PowerShell, padrão do Windows 11, falham. | Dizer "abra o Prompt de Comando (cmd.exe)" ou dar também a forma do PowerShell. |
| 6 | ~306 | Os trechos de uso no Maya não põem o repositório no `sys.path`, e em outra sessão o import falha. | Abrir cada trecho com `sys.path.insert(0, r"C:\dev\Toptop")` ou documentar o `userSetup.py` (o texto está no docstring de `toptop/shelf.py`). |
| 7 | ~47 | A tolerância do loop está descrita de forma imprecisa. | Pontas: min(1% da diagonal, 0,3 × menor aresta nova no vértice). Meio da aresta: min(1% da diagonal, 0,3 × o próprio comprimento). |
| 8 | ~58 | A amostragem do Hausdorff está descrita de forma imprecisa. | Novo→original: vértices + centros de face + 50k amostras. Original→novo: vértices + 50k amostras. Vértices e centros limitados a 50k cada. |
| 9 | ~107 | A linha "QUALIFICA" do selftest não é a última da saída. | Dizer que deve aparecer `quadwild ... passou em 3/3 ... QUALIFICA` e depois `Motor escolhido pelas métricas: quadwild`. |
| 10 | ~180 / ~209 | Entradas com mais de 300 mil triângulos (a 09_500k_faces) não ganham captura `_entrada`. "Seção 3" aponta para o plano original, não para este arquivo. | Acrescentar a exceção e escrever "seção 3 do plano (HANDOFF original)". |

A segunda metade da verificação (comparar a entrega com cada item do brief original)
**não terminou**. Vale rodá-la de novo depois das correções.

### 1.2 Fechar a entrega

1. Aplicar 1.1, rodar `pytest` com o motor real e as flags do Maya ligados:
   `TOPTOP_QUADWILD_DIR=<pasta do quadwild> TOPTOP_MAYA_CMDS_PKL=<cmds2025.pkl> pytest`.
   Depois, commit e push no branch.
2. Abrir o PR de `claude/new-session-vw85ux` para `main`. Ainda não foi aberto.

### 1.3 No Windows (só você pode fazer, porque precisa do Maya)

Passo a passo em `HANDOFF_NEXT.md` seção 2:

1. Baixar o QuadWild Bi-MDF v0.0.2 (`windows-binaries.zip`) para `third_party\quadwild-bimdf\`.
2. Rodar `%MAYAPY% tests\run_phase0.py --selftest`.
3. Rodar `%MAYAPY% tests\make_test_models.py`, que gera os 10 modelos.
4. Rodar `%MAYAPY% tests\run_phase0.py`, que grava o CSV/JSON e os wireframes. O Quad Remesher é manual.
5. Escolher o motor pelo portão (≥ 8/10 modelos). Depois rodar
   `%MAYAPY% tests\run_headless.py --engine <motor> --save-baseline`.
6. Partes que nunca rodaram num Maya de verdade: toda a camada Maya (`maya_bridge`,
   `maya_*` em creases/pipeline/cleanup, Retopologize, a janela e a shelf) foi conferida
   só offline, por um checador de flags sobre o cache do Maya 2025. Espere ajustes
   pequenos no primeiro uso.

### 1.4 TODOs técnicos conhecidos (detalhes em HANDOFF_NEXT.md seção 3)

- Os argumentos do Quad Remesher por script estão marcados `# TODO verificar`, configuráveis
  pelas variáveis `TOPTOP_QR_*`. No lote ele é só manual, por causa da licença.
- QuadWild: os argumentos do `prep.txt`/`main.txt` foram confirmados no binário Linux, mas o
  `.exe` do Windows não foi testado. No Linux, o binário dá segfault ao sair; isso já é
  tratado (`KILLED_AFTER_DONE`).
- Os controles da Fase 2/3 na janela (simetria, modo film completo e outros) estão desativados.

### 1.5 Auditoria da camada Maya (Maya 2026)

Corrigido, com testes (API do Maya simulada):

- `maya_bridge.create_mesh` confere se o Maya criou o mesmo número de vértices e
  faces. Se a numeração mudasse, hard edges, creases e mapa de calor cairiam nos
  vértices errados sem aviso. O pipeline agora compacta as cascas antes de juntar.
- `maya_read_creases`: `getCreaseEdges` que falha numa malha sem creases vira
  "sem creases"; só as arestas com crease são lidas (antes, uma chamada da API por
  aresta da malha toda).
- Retopologize: se o `replaceOriginal` não for aplicado, o resultado é lido do
  objeto novo criado pelo `polyRetopo`; nada do motor fica na cena; malha intacta
  vira erro claro, em vez de "resultado" igual à entrada.
- Shelf: um separador na prateleira derrubava o `install()` (o `exists` do
  `shelfButton` não olha o tipo); o botão salvo sem `docTag` agora é achado pelo
  rótulo + comando (sem isso, reinstalar duplicava); o retorno do `addNewShelfTab`
  não é mais usado como nome da aba.
- Janela: a estimativa de quads (a cada troca de seleção) usa `polyEvaluate` +
  `exactWorldBoundingBox` em vez de copiar a malha para o Python, o que travava
  o Maya ao clicar numa peça pesada.
- `make_test_models.py`: nomes de nó sem dígito no começo e sem `-`.
- Caminhos do Maya 2026 na documentação (`Maya2026`, `maya\2026\scripts`).

Ainda a conferir no Maya 2026 (não dá para verificar fora dele; a documentação da
Autodesk não estava acessível desta sessão):

- `polyRetopo` existe no `mayapy` sem carregar plugin? Se o selftest disser "esta
  versão do Maya não tem o comando polyRetopo", é um plugin não carregado.
- `polyRetopo` com `replaceOriginal` + `constructionHistory=False` substitui a
  malha? O adaptador agora cobre os dois casos, mas vale olhar o primeiro resultado.
- O `polyReduce` mantém os valores de crease? Se não, o pipeline transfere as
  creases do original pela geometria (nota "transferidas do original" no relatório).
- Ler 500 mil faces (`mesh_from_maya`, `edge_vertex_pairs`, Sugerir/Verificar)
  converte a API ponto a ponto em Python: alguns segundos por leitura.
- Desfazer/refazer do objeto criado e as cores do mapa de calor (já listados na
  seção 3 do HANDOFF_NEXT).

---

## 2. Resumo do que foi feito

**Base (`toptop/core`)**
- `mesh.py`: malha em arrays NumPy, topologia vetorizada, triangulação por ear clipping, cascas e subdivisão.
- `geom.py`: distância exata ponto–triângulo/segmento com grade uniforme e amostragem.
- `quality.py`: todas as métricas (quads, cobertura de crease, fidelidade das quinas,
  Hausdorff, proporção, polos, tempo), relatório, metas, mapa de calor e comparação com a referência (10%).
- `creases.py`: dois níveis (forte/guia), sugestão por ângulo diedro, "Verificar entrada", camada Maya.
- `cleanup.py`: solda, faces degeneradas, non-manifold, separação de cascas, redução prévia (polyReduce).
- `reproject.py`: reprojeção dos vértices na superfície, loops na crease, cantos exatos.
- `pipeline.py`: o pipeline completo de 9 passos, densidade por casca, hard edges e relatório.
  A camada Maya cria um objeto novo num único undo chunk, sem tocar no original.
- `render.py`: rasterizador próprio que grava os wireframes em PNG.
- `phase0.py`: lógica da Fase 0 (linhas, CSV de 44 colunas, JSON, portão).

**Entrada e saída:** `toptop/io/obj.py` e `features.py` (arquivo `.feat` de arestas de feature).

**Motores (`toptop/engine`):** interface `Engine` com três adaptadores.
- QuadWild Bi-MDF: processo externo com pasta temporária, tempo limite, cancelamento,
  pasta de diagnóstico mantida quando falha, normalização de escala e densidade em duas passadas.
- Retopologize do Maya: flags filtradas por versão.
- Quad Remesher: aviso de licença e fluxo manual.

**Maya e interface:** `maya_bridge.py`, janela PySide6 (`toptop/ui/window.py`, com botão de
mapa de calor) e instalador da shelf (`toptop/shelf.py`).

**Scripts de teste**
- `tests/make_test_models.py`: os 10 modelos.
- `tests/run_phase0.py`: modelo × motor, com worker separado, selftest e CSV para o Excel BR.
- `tests/run_headless.py`: autoverificação contra a referência.

**Verificação**
- 257 testes pytest sem Maya, também verdes no Python 3.11 + NumPy 1.26.
- QuadWild real (Linux) validado de ponta a ponta: selftest 3/3, 74 mil triângulos em 7,5 s.
- Flags do Maya conferidas contra o cache do pymel das versões 2022 a 2025.
- Duas revisões adversariais: 13 + 16 achados confirmados e corrigidos.

**Documentação:** `README.md` e `HANDOFF_NEXT.md` (passo a passo da Fase 0 no Windows,
legenda dos resultados, TODOs e uso).
