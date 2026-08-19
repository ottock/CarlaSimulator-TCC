# Fase 6a — Modelo dual 180° + export ONNX

**Data:** 2026-08-19
**Branch base:** `feat/ai-fase4-pista-custom` (Fase 4 fechada)
**Pré-requisito:** `driving_track_v1.pt` dirige 3/3 pistas em malha fechada (ver `docs/ESTADO_IA.md`)

## Contexto

A Fase 6 leva o modelo para o **Jetson Nano** no RC 1:12. Ao analisar o hardware já validado
(`hardware/controle_teste.py`) contra o que o modelo aprendeu, apareceram três achados que
mudam o desenho — dois bons e um que obriga a retreinar.

**Achado 1 — o LiDAR real já fala a nossa língua.** O `CoinD6Parser.feed()` devolve
`(angle_deg, dist_m)`, que é exatamente a assinatura de
`shared/lidar_pipeline.scan_to_sectors_m(angles_deg, dist_m, ...)`. Isso não é sorte: esse módulo
foi escrito na Fase 0 já citando o COIN-D6 e mantido em Python 3.6 para rodar no Jetson.

**Achado 2 — o gap 3D×2D não é fatal.** O sim usa um ray-cast de 32 canais; o carro usa um 2D de
varredura 360°. Mas `points_to_sectors_m` **achata a nuvem 3D** (projeta no plano, mínimo por
setor): o modelo nunca viu 3D, viu **72 números**. Para uma parede vertical a distância horizontal
independe da elevação do feixe, então um LiDAR 2D produz o mesmo vetor.

**Achado 3 — a oclusão do próprio carro obriga a retreinar.** Para enxergar a parede (baixa), o
sensor precisa ser montado baixo; nessa altura **a carroceria bloqueia a traseira**. O modelo foi
treinado com 360° de visão. Mascarar a traseira só na inferência muda **25% dos valores do vetor,
com erro médio 0.177** (escala 0–1) — ou seja, entregaria à rede uma entrada que ela nunca viu,
que é **exatamente a condição da ablação** (0/3 pistas, bate na parede).

**Objetivo desta fase:** entregar um modelo dual **180° validado em malha fechada** e o **ONNX
pronto para o Jetson**, sem depender do hardware. A calibração de bancada e o runtime no carro
são a Fase 6b.

## Decisões (com a evidência que as sustenta)

| # | Decisão | Evidência / razão |
|---|---|---|
| 1 | **FOV limitado, parametrizado (`fov_deg`, default 180°)** | Máscara só na inferência = 0.177 de erro médio no vetor = condição da ablação. Tem de ser treinada. |
| 2 | **Manter 72 setores; mascarar como "livre"** — não encolher para 36 | O braço Conv1D usa **padding circular** porque o setor 71 é vizinho do 0 (ambos na frente). Num vetor de 36 posições as pontas (−90°/+90°) deixariam de ser vizinhas e o padding costuraria extrema-esquerda na extrema-direita — geometricamente errado. |
| 3 | **`fov_deg` viaja no checkpoint** | Descasamento silencioso entre a máscara do treino e a da inferência é catastrófico (= ablação). Não pode depender de repetir uma flag. |
| 4 | **Escala: `max_range = 12.0/12 = 1.0 m` no carro** | A normalização divide por `max_range`, então isso é **matematicamente idêntico** a multiplicar as leituras por 12 — sem multiplicador mágico e sem tocar no `shared/`. Parede real a 0.265 m → 0.265 normalizado, igual ao treino. |
| 5 | **Retreinar a partir do `dataset_track_v1` — sem recoletar** | O `lidar.npy` guarda os 72 setores **em metros**; a máscara é transformação de dados. ~20 min de GPU, zero CARLA. |
| 6 | **Controle longitudinal fora de escopo** | O `dataset_track_v1` tem **0% de frames com freio** (o Pure Pursuit ignora obstáculos), então o head de brake é inerte. Nem o throttle do modelo pararia o carro. Velocidade fixa na 6b. |
| 7 | **Export em opset 11** | Validado: exporta, `onnx.checker` OK, ops todas nativas do TensorRT, paridade 5.96e-08. |

### Medições que embasam o design

Sobre `dataset_track_v1` (14.400 frames):

```
ocupação      frente(180°) 45.1%  |  traseira(180°) 49.6%
mediana       frente 0.376 m real |  traseira 0.272 m real
máscara da traseira: 25% dos valores alterados, erro médio 0.177 (escala 0–1)

cone frontal ±10° em condução NORMAL:  pior 0.1% = 0.287 m real
   → trava a 0.25 m real dispararia em 0.00% dos 14.400 frames (0 falsos positivos)

ONNX: opset 11/13/16/17 OK · ops {Conv, Elu, Gemm, Flatten, Unsqueeze, Slice,
      Concat, Constant, Tanh, Sigmoid} · paridade PyTorch↔ONNX 5.96e-08
```

O padding circular **não** virou um `Pad(wrap)` exótico: o exportador o decompõe em `Slice`+`Concat`,
operadores triviais. TensorRT 8.x (JetPack 4.6) cobre até opset 13–14 — opset 11 passa com folga.

## Arquitetura

A máscara é uma função pura no módulo **compartilhado sim↔carro**, e é a mesma chamada nos dois
lados. Nada de arquitetura de rede muda.

```
                          shared/lidar_pipeline.py
                          ┌──────────────────────────────────┐
  sim:  nuvem CARLA  ───► │ points_to_sectors_m → 72 setores │
                          │ apply_fov_mask(fov_deg)          │ ──► normalize_sectors_m ──► rede
  carro: scan COIN-D6 ──► │ scan_to_sectors_m   → 72 setores │     (max_range 12.0 sim
                          └──────────────────────────────────┘      / 1.0 no carro)
```

**Nova função** (Python 3.6-safe, ao lado de `scan_to_sectors_m`):

```
apply_fov_mask(sectors_m, fov_deg=180.0, max_range=MAX_RANGE_M) -> np.ndarray
```
Setor `i` cobre `[i*360/n, (i+1)*360/n)`; seu centro é `(i+0.5)*360/n`. Mantém o setor se o centro,
normalizado para `[-180, +180]`, satisfaz `|ângulo| <= fov_deg/2`; senão recebe `max_range` (livre).
`fov_deg >= 360` é no-op.

### Arquivos

| Arquivo | Mudança |
|---|---|
| `src/ai/shared/lidar_pipeline.py` | + `apply_fov_mask` (puro, Py3.6) |
| `src/ai/dataset.py` | `DrivingDataset(..., fov_deg=None)` aplica ao carregar — dado cru intacto |
| `src/ai/train.py` | `--fov-deg`; **grava `fov_deg` no checkpoint** |
| `src/ai/model_policy.py` | `DrivingPolicy` **lê `fov_deg` do checkpoint** (parâmetro explícito sobrescreve) |
| `src/ai/eval_track.py` | nada — herda do checkpoint |
| `src/ai/export_onnx.py` | **novo**: exporta opset 11 + **teste de paridade obrigatório** |
| `requirements.txt` | + `onnx`, `onnxruntime` |
| `tests/` | testes da máscara, do checkpoint com `fov_deg` e da paridade |

## Verificação (ponta a ponta)

1. **`pytest -q`** — testes novos (máscara zera os setores certos; 360° é no-op; `DrivingPolicy` lê
   `fov_deg` do checkpoint) + os 97 atuais verdes.
2. **Retreinar** `driving_track_180.pt` no `dataset_track_v1` com `--fov-deg 180` (sem recoletar).
3. **`eval_track`** nas 3 pistas → **critério de aprovação: continuar 3/3 limpas**
   (`offlane=0`, `collisions=0`). Se degradar, é sinal de que 180° não basta — e descobrimos
   no simulador, não na bancada.
4. **`export_onnx`** → `onnx.checker` OK e **paridade < 1e-4** medida no split de validação
   (não em ruído aleatório).
5. Documentar em `docs/ESTADO_IA.md` e registrar a constante de escala real para a 6b.

## Fora de escopo (explícito)

- Frenagem / controle longitudinal aprendido (dataset tem 0% de frenagem).
- Runtime no Jetson, TensorRT, PCA9685 — **Fase 6b**.
- Medições de bancada — **Fase 6b** (mas o resultado delas realimenta o `fov_deg` desta fase).

## Fase 6b — o que virá depois (contexto, não especificado aqui)

Medições de bancada, **antes** do runtime:
1. **Arco real de oclusão** — carro em área aberta, logar o scan cru e ver quais ângulos devolvem
   distância curta constante (é a carroceria). Esse número vira o `fov_deg` do retreino.
2. **Zero e sentido do ângulo** — objeto em ângulo conhecido; se estiver espelhado, o mundo chega
   invertido na rede.
3. **Velocidade real a 1600 µs** — o modelo aprendeu a 2 m/s no sim = **0.167 m/s real**. Se o ESC
   no mínimo andar muito mais rápido, o carro corta as curvas.
4. **Altura da parede e do cone × plano do LiDAR** — se o feixe passar por cima, não há referência.

Runtime: câmera + LiDAR → `shared/*` (as mesmas funções) → engine TensorRT → PCA9685
(servo ch15 `1500 + steer*200` µs; ESC ch12 em PWM fixo), reusando `watchdog`/`ESC_ARMADO` do
`hardware/controle_teste.py`. **Montar o LiDAR no para-choque dianteiro** — no centro a carroceria
bloqueia nos dois sentidos do eixo.
