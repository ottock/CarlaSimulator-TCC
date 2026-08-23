# Fase 6b (parte 1) — Runtime no Jetson, primeiro teste com rodas no ar

**Data:** 2026-08-23
**Branch base:** `feat/ai-fase4-pista-custom` (Fase 6a fechada, commit `7472727`)
**Pré-requisito:** `driving_track_180.onnx` exportado e com paridade provada (ver `docs/ESTADO_IA.md`)

## Contexto

A Fase 6a entregou o modelo pronto para embarcar: `driving_track_180.pt` dirige as 3 pistas
com FOV de 180° (3/3 limpas) e o `driving_track_180.onnx` saiu em opset 11 com paridade
5.13e-07. O que falta é o outro lado da ponte — **nada no repositório roda o modelo no carro**.
O `hardware/controle_teste.py` já faz LiDAR + PCA9685 + watchdog, mas dirigido pelo **teclado**.

**Decisão do Rafael (2026-08-23):** antes de fechar as medições de bancada, ele quer **ver o
modelo rodando no carro**. O caminho seguro para isso é o **primeiro teste com as rodas no ar**:
ESC desarmado, só o servo responde. Esse teste entrega o que ele quer ver E, de brinde, as
medições #1 (arco de oclusão) e #2 (zero e sentido do ângulo) — com número, a partir de um log.

## Objetivo

Um runtime que leia câmera + LiDAR no Jetson, rode o modelo em TensorRT e comande o **servo**,
com o **ESC travado em neutro**, gravando um log que o PC analisa depois.

**Não** é objetivo o carro andar. Isso é o passo seguinte, depois de olhar o log.

## Decisões (com a evidência que as sustenta)

| # | Decisão | Evidência / razão |
|---|---|---|
| 1 | **Backend = TensorRT** | Já vem no JetPack 4.6; usa a GPU. Escolha do Rafael. Custo aceito: gerar a engine no próprio Jetson e instalar o `pycuda`. |
| 2 | **O repo é clonado no Jetson** | Confirmado pelo Rafael. Permite importar `ai.shared.*` de verdade, em vez de copiar arquivos — cópia manual é exatamente como nasce divergência silenciosa. |
| 3 | **Lógica pura em `src/ai/car/`, adaptadores em `hardware/`** | O `pytest` já tem `pythonpath = ["src"]`, então `src/ai/car/` é testável hoje sem configurar nada. `hardware/` não é importável no PC: `coin_d6.py` faz `sys.exit()` se faltar pygame. |
| 4 | **Um `CoinD6Parser` canônico, extraído** | Hoje há **duas** cópias e elas **já divergiram**: `coin_d6.py` devolve `(ângulo, dist, intensidade)`, `controle_teste.py` devolve `(ângulo, dist)`. A de 2 elementos é a que casa com `scan_to_sectors_m`. Criar uma terceira cópia seria repetir o erro. |
| 5 | **Fechar o scan por wrap de ângulo, não por contagem** | O parser fecha a cada **300 pontos**, não a cada volta. Para o radar tanto faz; para o modelo, um setor não varrido leria "livre" — um buraco na parede, que é justamente o sinal que faz o carro bater. |
| 6 | **Sidecar JSON ao lado do `.onnx`** | A engine TensorRT **não** carrega os `metadata_props` do ONNX — o `fov_deg` se perde no `trtexec`. Um `.json` escrito pelo `export_onnx.py` é lido no Jetson com a stdlib, sem exigir o pacote `onnx` lá. |
| 7 | **`max_range = 1.0` no carro** | Como a normalização divide por `max_range`, é *matematicamente idêntico* a multiplicar as leituras por 12 (a escala 1:12), sem multiplicador mágico e sem tocar no `shared/`. |
| 8 | **ESC desarmado, fixo no código** | O carro não anda nesta fase. Não é flag de linha de comando: tem de exigir editar o arquivo. |
| 9 | **O log é o entregável, não a tela** | Escolha do Rafael sobre um HUD gráfico. Menos código, e responde melhor: o arco de oclusão e o sentido do ângulo saem com número no PC, não no olho. |

## Arquitetura

### Arquivos

| Arquivo | Mudança |
|---|---|
| `src/ai/car/__init__.py` | **novo** |
| `src/ai/car/coin_d6.py` | **novo** — o parser canônico, extraído de `controle_teste.py` (2-tuplas), sem side-effect de import |
| `src/ai/car/scan_assembly.py` | **novo** — puro: acumula pacotes e fecha o scan no wrap de ângulo |
| `src/ai/car/control_map.py` | **novo** — puro: `steer` → µs, clamps, watchdog |
| `src/ai/car/config.py` | **novo** — puro: lê o sidecar JSON do modelo |
| `src/ai/car/image_crop.py` | **novo** — puro: recorte central para compensar a lente de 130° |
| `src/ai/car/run_log.py` | **novo** — grava o log da corrida (nome NÃO é `logging.py`: não sombrear a stdlib) |
| `src/ai/car/loop.py` | **novo** — `DriveLoop` com o hardware injetado; é aqui que o envelope de segurança fica testável no PC |
| `hardware/jetson_runtime.py` | **novo** — fino: GStreamer, serial, TensorRT, PCA9685 + o laço |
| `src/ai/export_onnx.py` | passa a escrever o sidecar `.json` |
| `scripts/analyze_car_log.py` | **novo** — roda no PC: mede o log |
| `tests/` | testes das partes puras |

Tudo em `src/ai/car/` é **Python 3.6-safe**, pela mesma razão que `src/ai/shared/` é.

### Fluxo

```
camera CSI 1280x720 --> corte central (130 -> ~62 graus) --> 640x360 --+
                                                                       +--> preprocess() --+
LiDAR COIN-D6 --> parser --> scan 360 --> scan_to_sectors_m -----------+                   |
                                          apply_fov_mask(fov do JSON)                      +--> engine TRT --> steer
                                          normalize_sectors_m(max_range=1.0) --------------+                    |
                                                                                                                v
                                                          servo ch15 = 1500 + steer*200 us (clamp 1300..1700)
                                                          ESC   ch12 = 1500 us (neutro, DESARMADO)
```

As funções de `ai.shared.*` são **importadas, nunca reescritas** — qualquer divergência ali vira
erro silencioso de inferência.

### Contratos dos módulos puros

- `scan_assembly.ScanAssembler.feed(points) -> lista de scans` — acumula e devolve um scan por
  volta completa, detectando o wrap. Volta parcial no fim não é emitida.
- `control_map.steer_to_us(steer) -> int` — `1500 + steer*200`, clampado em [1300, 1700],
  tolerante a `steer` fora de [-1, 1] e a `NaN` (devolve o centro).
- `control_map.Watchdog(timeout_s)` — diz se o frame estourou o prazo.
- `config.load_model_config(path) -> dict` — `fov_deg`, `n_sectors`, `max_range_m`; erro claro se
  o arquivo faltar (não silencioso, não default mágico).
- `image_crop.center_crop(frame, frac) -> frame` — recorta a região **central** do quadro usando
  a **mesma fração nas duas dimensões**, preservando o aspecto 16:9, e roda **antes** do resize
  para 640×360 (para perder o mínimo de resolução) e antes do `preprocess()`. `frac = 1.0` é
  no-op, para conseguir gravar um log sem recorte nenhum durante a calibração.

## Segurança

- `ESC_ARMADO = False` **fixo no código**, não é flag. O ESC vai a neutro e fica.
- Servo clampado em [1300, 1700] µs — mesmo se o modelo devolver lixo.
- Watchdog de 0.25 s (reusa a ideia já validada no `controle_teste.py`).
- `Ctrl+C` e qualquer exceção saem em **estado seguro**: servo ao centro, ESC neutro.

## O log

Um diretório por execução, `runs/car_<timestamp>/`, com formato fixo:

```
meta.json          config do modelo, fracao de corte, versao, ESC armado (sempre false aqui)
frames.jsonl       1 linha JSON por frame: t, steer, throttle, brake, servo_us, fps, dt
sectors.npy        (N, 72) float32 — o vetor EXATO entregue ao modelo, ja normalizado
scans.jsonl        1 linha por volta do LiDAR: t + a lista de (angulo, distancia) crus
frames/000000.jpg  1 frame a cada N (padrao 10), no tamanho ja recortado
```

Formatos deliberadamente banais: `jsonl` e `npy` se leem no PC sem dependência nenhuma, e o
`sectors.npy` casa com o formato do `lidar.npy` do dataset, então dá para comparar a entrada real
com a de treino direto. O `scripts/analyze_car_log.py` lê isso no PC e devolve:

1. **Arco de oclusão** (medição de bancada #1) — quais ângulos devolvem distância curta constante.
2. **Zero e sentido do ângulo** (medição #2) — comparando com um objeto em posição conhecida.
3. **FPS real** — o sim roda a 20 Hz; bem menos que isso e o carro reage tarde.
4. **Ocupação dos setores** comparada com o `dataset_track_v1`, para ver o quanto a entrada real
   se parece com a de treino.

## Verificação

1. `pytest -q` — testes novos das partes puras + os 125 atuais verdes.
2. No Jetson: `trtexec --onnx=driving_track_180.onnx --fp16` gera a engine.
3. Runtime com rodas no ar: o servo esterça de forma coerente ao apontar o carro para a parede.
4. `analyze_car_log.py` sobre o log da corrida devolve os 4 números acima.

## Fora de escopo (explícito)

- **ESC armado / carro andando.** Passo seguinte, depois de olhar o log.
- **HUD gráfico.** Decisão do Rafael; pode voltar depois, sabendo o que vale desenhar.
- **Parada de emergência.** Vem junto com armar o ESC — sem ESC, não há o que parar.
- **Controle longitudinal aprendido.** Segue fora de escopo desde a 6a: o `dataset_track_v1` tem
  **0 frames com freio** (verificado nos 14.400), a cabeça de brake é inerte.

## Pendências conhecidas (registradas, não esquecidas)

**A lente da câmera é de 130°, o sim foi treinado com 62.2°.** Não há lente sobressalente, então
fica assim por ora — **decisão do Rafael: resolver depois**. É o maior descasamento sim↔real em
aberto: 130° é mais que o dobro do campo, com distorção de barril nas bordas. Mitigação parcial
já embutida: o runtime recorta a região central da imagem, com a fração como **parâmetro de linha
de comando**, default **0.48**. Esse 0.48 é a estimativa *se* a lente for próxima de equidistante
(fisheye); se for retilínea o valor certo fica perto de 0.28. **É um chute a calibrar olhando os
frames gravados, não um número medido** — por isso é parâmetro, não constante.

**O risco de velocidade da 6a segue aberto** e não é tocado aqui (o carro não anda): o modelo
esterça numa velocidade só (p50 1.73 m/s no sim = **0.144 m/s** real). Ver §4.3 do
`docs/ESTADO_FASE6.md`.
