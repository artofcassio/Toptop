# HANDOFF_NEXT — TopTop

Estado do repositório depois da primeira sessão e o passo a passo para rodar a
**Fase 0** no Windows. Branch: `claude/new-session-vw85ux`.

## 1. O que ficou pronto

| Item do HANDOFF | Arquivo(s) | Estado |
| --- | --- | --- |
| 1. Esqueleto + README | `pyproject.toml`, `README.md`, `toptop/` | pronto |
| 2. Métricas (a peça mais valiosa) | `toptop/core/quality.py` (+ `mesh.py`, `geom.py`) | pronto e testado |
| 3. OBJ + arquivo de features | `toptop/io/obj.py`, `toptop/io/features.py` | pronto e testado |
| 4. Interface + 3 adaptadores | `toptop/engine/base.py`, `quadwild.py`, `maya_retopo.py`, `quadremesher.py` | QuadWild testado com o binário real; os dois do Maya só com um Maya simulado |
| 5. Os 10 modelos de teste | `tests/make_test_models.py` | escrito, **nunca executado** (não havia Maya) |
| 6. Fase 0 | `tests/run_phase0.py`, `toptop/core/phase0.py`, `toptop/core/render.py` | testado de ponta a ponta com o QuadWild real (`--selftest`) |
| 7. Creases | `toptop/core/creases.py` | lógica pura testada; camada Maya fina por cima |
| 8. Extras | `toptop/core/cleanup.py`, `reproject.py`, `pipeline.py`, `toptop/ui/window.py`, `toptop/shelf.py` | prontos e testados sem Maya (a janela em PySide6 sem tela, com um Maya simulado); **nunca rodaram dentro do Maya** |
| Seção 10 | `tests/run_headless.py` (autoverificação) | pronto; testado com o QuadWild real |
| Fase 2 (adiantado) | mapa de calor de desvio, estimativa de quads, Verificar entrada, redução de malhas grandes | prontos (a redução usa `polyReduce` e não foi testada no Maya) |

Testes: `pytest` na raiz (sem Maya): 257 testes, verdes em Python 3.13/NumPy 2.x
e em Python 3.11/NumPy 1.26 (o par do Maya 2025). Sem as variáveis
`TOPTOP_QUADWILD_DIR` e `TOPTOP_MAYA_CMDS_PKL` alguns testes são pulados.

### O que foi verificado nesta sessão (e o que não foi)

- **QuadWild real** (binário Linux v0.0.2 do mesmo release): adaptador,
  `run_phase0.py --selftest`, `run_headless.py --selftest` e o pipeline
  completo rodaram de ponta a ponta, com todas as metas cumpridas nas peças
  sintéticas. O binário Windows não pôde ser executado aqui.
- **Comandos do Maya**: as 120 chamadas `maya.cmds` do repositório foram
  conferidas automaticamente contra a lista de comandos e flags do Maya 2025
  (cache do pymel 1.5.0). Os flags do `polyRetopo` também foram conferidos de
  2022 a 2025; o adaptador filtra pelo que a versão em uso declara.
- **Revisão adversarial**: o código que não roda aqui (Maya, mayapy, Windows)
  passou por 4 revisores independentes com verificação de cada achado (13
  confirmados e corrigidos); os módulos de limpeza, reprojeção, pipeline e
  janela, por mais 2 revisores (16 achados, todos corrigidos com testes).
- **Nada rodou dentro do Maya.** `make_test_models.py`, o adaptador do
  Retopologize, a janela, a shelf e a camada Maya do pipeline precisam da
  primeira execução sua; espere pequenos ajustes.

### Como as métricas são medidas (resumo de `quality.evaluate`)

- **Quads**: faces de 4 lados / total.
- **Cobertura das creases**: cada crease original é dividida em trechos de 0,2% da
  diagonal; um trecho conta como coberto quando existe uma aresta nova que (a)
  está sobre a crease — pontas e meio a menos de `min(1% da diagonal, 0,3 x o
  comprimento da própria aresta)` —, (b) corre na direção dela e (c) tem essa
  crease como a mais próxima. Isso impede que, numa chapa fina, o loop de cima
  "cubra" a crease de baixo, e que numa malha densa o loop vizinho, paralelo,
  seja tomado pelo loop da crease.
- **Fidelidade das quinas**: o maior entre (a) a distância dos vértices desses
  loops até a crease e (b) a distância de cada canto original (junção, ponta ou
  virada > 50°) até o vértice novo mais próximo. Dividido pela diagonal.
  `feature_chord_dev` (a flecha das cordas em creases curvas) sai só como
  informação.
- **Desvio da forma**: Hausdorff por amostragem nos dois sentidos (vértices,
  centros de face e 50 mil amostras por área em cada malha), distância exata
  ponto–triângulo. O sentido original -> nova acusa partes faltando.
- **Proporção**: lado maior / menor de cada quad (mediana e percentil 95).
- **Polos**: vértices internos com valência != 4; os que estão sobre crease forte
  e longe dos cantos são contados à parte (meta: zero).
- **Tempo**: segundos do motor; a meta de 2 min vale para 100 mil triângulos e
  cresce proporcionalmente até o tempo limite de 10 min.
- Bordas abertas da original entram como feature de nível guia.

## 2. Fase 0 no Windows, passo a passo

Os comandos supõem o repositório em `C:\dev\Toptop` e o Maya 2025 no lugar
padrão. Troque os caminhos se for diferente.

### 2.1 Preparar

```bat
cd C:\dev
git clone https://github.com/artofcassio/Toptop.git
cd Toptop
git checkout claude/new-session-vw85ux

set MAYAPY="C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe"
%MAYAPY% -c "import numpy; print(numpy.__version__)"
```

Se o último comando falhar (NumPy ausente no mayapy), instale num prompt de
administrador: `%MAYAPY% -m pip install numpy`. O código foi testado com
NumPy 1.26 (Python 3.11, igual ao do Maya 2025) e com NumPy 2.x.

### 2.2 Baixar o QuadWild Bi-MDF

1. Abra https://github.com/cgg-bern/quadwild-bimdf/releases/tag/v0.0.2 (é um
   *pre-release*; não existe "latest") e baixe **`windows-binaries.zip`** (2,8 MB).
2. Extraia em `C:\dev\Toptop\third_party\quadwild-bimdf\` de modo que fique
   `third_party\quadwild-bimdf\quadwild.exe`, `quad_from_patches.exe` e a pasta
   `config\` lado a lado. (Uma subpasta a mais também é encontrada.) A pasta
   `third_party/` está no `.gitignore`: o binário GPL nunca entra no repositório.
3. Alternativa: deixe a pasta onde quiser e aponte a variável
   `set TOPTOP_QUADWILD_DIR=D:\ferramentas\quadwild-bimdf`, ou passe
   `--quadwild-dir` ao `run_phase0.py`.

Conferência rápida, sem modelos do Maya (3 peças sintéticas, ~20 s):

```bat
%MAYAPY% tests\run_phase0.py --selftest --engines quadwild
```

Deve terminar com `quadwild passou em 3/3 ... QUALIFICA`.

### 2.3 Gerar os 10 modelos de teste

```bat
%MAYAPY% tests\make_test_models.py
```

Grava em `tests\models\`: `<modelo>.mb` (cena com creases 3 = forte e 1 = guia,
hard edges nas fortes), `<modelo>.obj` (triangulado), `<modelo>.feat` (pares de
vértices + crease) e `models.json` (lista + quads pedidos a todos os motores).
Para refazer só alguns: `--only 01,09`. **Este script nunca rodou**: as chamadas
do Maya foram conferidas contra a lista de comandos e flags do Maya 2025, mas a
geometria das booleanas só se confirma rodando. Abra cada `.mb` e confira as
creases (Display > Polygons > Crease Edges) antes de seguir. Se um modelo falhar,
os outros continuam e o erro aparece no fim. Modelos:

| # | Nome | O que testa | Quads pedidos |
| --- | --- | --- | --- |
| 1 | `01_placa_furos` | furos, rebaixo, linha de painel (crease guia em x = 0) | 3000 |
| 2 | `02_cilindros_canos` | cano em T com flange e furos | 4000 |
| 3 | `03_booleana_ngons` | ressalto, rebaixo sextavado, rasgo inclinado | 2500 |
| 4 | `04_chanfros_fillets` | fillets de 6 segmentos + chanfro | 2500 |
| 5 | `05_simetrica` | peça simétrica em X | 3000 |
| 6 | `06_chapa_fina` | chapa em L de 0,15 com furos | 3000 |
| 7 | `07_varias_cascas` | 6 cascas no mesmo objeto | 3500 |
| 8 | `08_pecas_repetidas` | 6 parafusos idênticos | 4000 |
| 9 | `09_500k_faces` | ~500 mil triângulos | 6000 |
| 10 | `10_quinas_rasas` | vincos de ~20° (sugestão com 15°) | 1500 |

Você pode trocar os modelos pelos seus (booleanas/CAD reais): basta colocar
`.obj` + `.feat` em `tests\models\` e editar o `models.json`. Para gerar o
`.obj` + `.feat` de uma peça sua com creases já marcados no Maya, veja a seção 4.

### 2.4 Quad Remesher (manual, por causa da licença)

A licença do Quad Remesher não cobre automação, então o `run_phase0.py` nunca o
chama: ele só mede resultados feitos à mão. Para cada modelo, no Maya com
interface, cole no Script Editor (aba Python):

```python
import sys; sys.path.insert(0, r"C:\dev\Toptop")
from toptop.io.obj import read_obj
from toptop.io.features import read_features, features_path
from toptop.engine import quadremesher as qr

nome, quads = "01_placa_furos", 3000            # veja a tabela acima
obj = r"C:\dev\Toptop\tests\models\%s.obj" % nome
malha = read_obj(obj)
qr.prepare_manual(malha, read_features(features_path(obj), malha.num_points), quads, nome)
```

Rode o Quad Remesher pela janela dele com *Target Quad Count* = `quads` e o
uso de hard edges ligado (as creases viraram hard edges). Anote o tempo.
Selecione a malha gerada e grave:

```python
qr.collect_manual(r"C:\dev\Toptop\tests\output\phase0\manual\quadremesher\%s.obj" % nome)
open(r"C:\dev\Toptop\tests\output\phase0\manual\quadremesher\%s.time.txt" % nome, "w").write("42.0")  # segundos
```

### 2.5 Rodar a Fase 0

```bat
%MAYAPY% tests\run_phase0.py
```

Roda cada modelo em cada motor com a mesma densidade e grava em
`tests\output\phase0\`:

- `phase0_results.csv` (use `--excel-br` para abrir direto no Excel em
  português) e `phase0_results.json` (inclui a decisão do portão);
- `<motor>\<modelo>.obj` (o resultado) e `<motor>\<modelo>.png` (captura);
- `_entrada\<modelo>.png` (a peça original com as creases).

Opções úteis: `--engines quadwild,maya_retopo`, `--models 01,04`,
`--timeout 600`, `--quadwild-config flow_noalign`, `--quadwild-no-remesh` (o
pré-remesh do QuadWild reduz qualquer entrada a ~9 mil triângulos; desligue se
ele apagar detalhe), `--append` (mantém as linhas da execução anterior que não
forem refeitas; use depois de fazer só o Quad Remesher, por exemplo
`%MAYAPY% tests\run_phase0.py --engines quadremesher --append`).

O Retopologize roda num `mayapy` separado por execução (o comando não pode ser
cancelado; assim o tempo limite mata só aquela execução). Motor indisponível é
pulado com aviso.

### 2.6 Como ler o resultado

No fim aparece a tabela e o portão: segue o motor que cumprir as metas em
>= 8 de 10 modelos; desempate pela fidelidade média das quinas e depois pelo
tempo. **A nota visual vale tanto quanto as métricas**: abra os PNGs.

Legenda das capturas (duas vistas: frente por cima, costas por baixo):

- cinza com arestas pretas: a malha nova;
- **laranja**: edge loop sobre crease forte; **azul**: sobre crease guia ou borda;
- **pontos magenta**: trecho de crease original SEM edge loop (problema);
- quadrado **ciano**: polo de valência 3; **roxo**: valência 5 ou mais.

As metas estão em `toptop/core/quality.py` (`Targets`) e devem ser recalibradas
com o que a Fase 0 mostrar.

### 2.7 As três perguntas da seção 3, já respondidas para o QuadWild

1. **Aceita lista explícita de arestas de feature?** Sim: arquivo `.sharp`
   (`tipo,face,aresta` por linha), conferido no código-fonte
   (`LoadSharpFeatures`). Quando ele é passado, o motor não detecta nada por
   ângulo; as bordas abertas são acrescentadas pelo adaptador.
2. **Como a densidade é controlada?** Pelo `scaleFact` do config do segundo
   passo: lado do quad = aresta média da malha pré-remeshada x `scaleFact`. O
   adaptador calcula o valor para o número de quads pedido e faz uma segunda
   quadrangulação se sair mais de 15% fora.
3. **O que devolve quando falha?** Não há código de erro confiável (o binário
   Linux sai com falha de segmentação depois de gravar tudo; o original sai com
   0 em erro de arquivo). O adaptador decide pela existência e validade dos
   arquivos de saída e mantém a pasta temporária (com `log.txt`) para diagnóstico.

Medido nesta sessão (Linux, binário v0.0.2): cilindro de 74 mil triângulos ->
3 094 quads em 7,5 s, ou 15 900 quads em 11 s, todas as metas cumpridas.

## 3. O que ficou como TODO

**Para você, na Fase 0**

- Rodar `make_test_models.py` e conferir as 10 peças (seção 2.3). Se alguma
  booleana falhar, o erro sai no resumo; o mais provável é ajuste de
  coordenadas no `tests/make_test_models.py`.
- Conferir os nomes do Quad Remesher, se for usar o caminho por script:
  `MODULE_NAME`, `CLASS_NAME`, `METHOD_NAME` e `OPTION_VARS` em
  `toptop/engine/quadremesher.py` (ou as variáveis `TOPTOP_QR_MODULE`,
  `TOPTOP_QR_CLASS`, `TOPTOP_QR_METHOD`) são palpites marcados com
  `TODO verificar`; o site da Exoside não estava acessível e o plugin não é
  público. O caminho manual (seção 2.4) não depende deles.
- Dar a nota visual nos PNGs e recalibrar as metas em `Targets`
  (`toptop/core/quality.py`).
- Depois de escolher o motor: `%MAYAPY% tests\run_headless.py --save-baseline`
  e commitar `tests\baseline\headless.json`. Daí em diante, toda mudança roda
  `%MAYAPY% tests\run_headless.py` (código 1 = alguma métrica piorou > 10%).
  Atenção: o tempo também entra na comparação; numa máquina ocupada ele pode
  variar mais que 10% (há 1 s de folga).

**Coisas conhecidas do QuadWild (vistas aqui, no binário Linux)**

- O `quad_from_patches` sai com falha de segmentação depois de gravar tudo; o
  adaptador usa os arquivos, não o código de saída, e encerra o processo se
  ele gravar tudo e ficar pendurado (caixa de erro do Windows).
- O pré-remesh reduz qualquer entrada a ~9 mil triângulos: rápido, mas pode
  apagar detalhe pequeno (`--quadwild-no-remesh` / `do_remesh=False`).
- Com o pré-remesh ligado, ele **trava em casca perfeitamente plana**; o
  pipeline manda cascas planas sem pré-remesh e transforma cascas com menos de
  8 triângulos em quads direto (sem motor). Se uma casca ainda falhar, ela
  vira quads por subdivisão, o relatório reprova e as outras seguem.

**Fase 1 em diante (não feito)**

- Simetria (a opção existe e só gera uma nota "fica para a Fase 2").
- Normais ponderadas, support loops com verificação por subdivisão,
  círculos em múltiplos de 4, cópia de topologia entre peças repetidas, UVs
  (os controles aparecem desabilitados na janela).
- No modo Film, as creases dos loops são gravadas no resultado (crease
  transferido), mas não há support loops.
- O desfazer do objeto criado por `maya_bridge.create_mesh` (transform pelo
  `cmds`, geometria pela API) e as cores do mapa de calor (gravadas pela API,
  fora do desfazer) precisam ser conferidos no Maya.
- `maya_bridge.edge_vertex_pairs` percorre as arestas uma a uma pela API:
  alguns segundos numa peça de 500 mil faces.

## 4. Usar o TopTop no Maya (fora da Fase 0)

### Botão na shelf e janela

No Script Editor (aba Python), uma vez:

```python
import sys
TOPTOP = r"C:/dev/Toptop"
if TOPTOP not in sys.path:
    sys.path.insert(0, TOPTOP)
import toptop.shelf
toptop.shelf.install()
```

A aba "TopTop" ganha um botão que abre a janela. Ela segue a seleção e só
calcula ao clicar:

1. Selecione a peça. **Sugerir creases** seleciona as arestas acima do ângulo
   (nada é gravado); ajuste a seleção e clique **Gravar como forte** ou
   **Gravar como guia**. Creases marcados à mão também valem (forte >= 2,0).
2. **Verificar entrada** aponta pontas de crease soltas, excesso de arestas
   marcadas e quinas vivas sem crease, e seleciona os culpados.
3. Escolha modo, nível de detalhe (a estimativa de quads aparece ao lado) e
   motor; **Retopologizar** cria `<peça>_toptop` ao lado do original (que não é
   tocado), num único passo de desfazer, e mostra o relatório. **Cancelar**
   interrompe.
4. **Mapa de calor** pinta o resultado pelo desvio até o original (azul = em
   cima, vermelho = na meta de 0,5% da diagonal ou acima).

### Pelo script

```python
from toptop.core.pipeline import PipelineOptions, maya_retopologize
novo, resultado = maya_retopologize("pCube1", PipelineOptions(mode="game", detail=5))
from toptop.core.quality import format_report
print(format_report(resultado.report)); print("\n".join(resultado.notes))
```

### Exportar uma peça sua para os testes

```python
from toptop import maya_bridge
from toptop.core.creases import maya_read_creases
from toptop.io.obj import write_obj
from toptop.io.features import write_features, features_path

obj = r"C:\dev\Toptop\tests\models\minha_peca.obj"
write_obj(obj, maya_bridge.mesh_from_maya("minhaPeca"))
write_features(features_path(obj), maya_read_creases("minhaPeca"))
```

Depois acrescente a peça ao `tests\models\models.json` (nome, `obj`,
`features`, `target_quads`).
