# Estado da IA — Carro Autônomo 1:12 (Digital Twin no CARLA)

> Documento de referência do workstream de IA. Objetivo: orientar pessoas e IAs
> sobre **o que já foi feito, como rodar, e o que vem a seguir**.
> Branch de trabalho: `feat/ai-fase3-dual-input`.

---

## 1. O projeto em uma página

**TCC:** usar *digital twins* (gêmeo digital no CARLA) para treinar um carro RC
em escala **1:12** a dirigir sozinho num circuito.

- **Abordagem:** Behavioral Cloning (BC) — um *expert* dirige no CARLA, gravamos
  (câmera + LiDAR + comandos) e treinamos uma rede a imitar. No fim, a rede
  substitui o piloto do CARLA.
- **Sensores do modelo:** câmera RGB (agora) + LiDAR (Fase 3).
- **Estratégia:** **Estágio A** = Towns padrão do CARLA (valida a pipeline) →
  **Estágio B** = pista custom 1:12 (o gêmeo digital de verdade).
- **Escopo atual:** fazer o modelo **dirigir bem em simulação**. Deploy no
  **Jetson Nano** (ONNX→TensorRT) e a pista física são um plano **futuro**.
- **Treino:** PyTorch 2.6 + CUDA numa **RTX 3070**.

Onde ficam os dados/pesos (fora do OneDrive, para não corromper sync):
`D:\tcc_data\{dataset_v1, dataset_v2, dataset_v3, runs\*.pt}`.

---

## 2. Estado atual (o que funciona)

| Fase | O que é | Status |
|------|---------|--------|
| **0** | Malha fechada com o expert (harness) | ✅ Feito e validado |
| **1** | Pipeline de coleta de dados | ✅ Feito e validado |
| **2** | Modelo só-câmera (direção) em malha fechada | ✅ Dirige (ressalva: cruzamentos) |
| **3** | Dual-input (câmera+LiDAR, acelera/freia) | ✅ **Dirige e freia**; ablação do LiDAR inconclusiva no Town (§5.7) |
| **4** | Pista custom (gêmeo digital) | ✅ **3/3 pistas limpas**; ablação **decisiva** — o LiDAR provou valor (§5.7) |
| 5 | Refino final do modelo (touch-ups) | ⏸️ **ADIADA** (decisão do Rafael) — o modelo atual já serve para o carrinho |
| **6a** | **Modelo 180° + export ONNX** (não precisa do carro) | ⬜ **PRÓXIMA** — design escrito, §7 |
| 6b | Jetson: bancada + runtime (TensorRT + PCA9685) | ⬜ Depois da 6a |
| 7 | Pista física / gap sim-to-real | ⬜ Deferido |

**Marco da Fase 2 (`cam_v2.pt`):** modelo só-câmera dirige **~171 s** centrado a
~0.08 m no Town01. Open-loop `val_MAE 0.044`, `var_ratio 1.00`.

**Marco da Fase 3 (`driving_v3.pt`):** modelo **dual-input (câmera+LiDAR)** que
**dirige E freia**. Open-loop `MAE_steer 0.031`, `MAE_brake 0.067`, `var_ratio ~0.96`.
Num *tour* por 12 spawns de Town01, **10/12 dirigiram 25 s limpos** (desvio de faixa
~0.1 m); as únicas batidas em pista aberta foram em **cruzamento** (§5.3). No teste de
frenagem (carro parado à frente), parou sem bater em **~75–87%** dos trials.

**Marco da Fase 4 (`driving_track_v1.pt`):** modelo dual treinado nas **3 pistas do
estande** (Pure Pursuit como expert). Open-loop `MAE_steer 0.039`, `MAE_throttle 0.017`,
`MAE_brake 0.006`, `var_ratio 1.01`. Em malha fechada (120 s por pista): **3/3 pistas
limpas**, `offlane=0`, `collisions=0`, desvio médio 0.41–0.58 m. **A ablação do LiDAR
finalmente separou** (§5.7): sem LiDAR, **0/3 limpas**.

**A história do LiDAR (§5.7 → §5.7.1):** na **Fase 3** (Towns) a ablação deu
**inconclusiva** — a câmera sozinha bastava para frear atrás de carro grande. A previsão
era que o LiDAR provaria valor nos **obstáculos pequenos** da pista custom. Na **Fase 4** ele
provou — mas pela **parede/corredor** (lane-keeping), não pelos obstáculos. Previsão certa
no destino, errada na causa.

**A ressalva (Fases 2–3):** ainda erra em **cruzamentos complexos** (§5.3) — limite
conhecido do BC mono-câmera, **inexistente na pista fechada 1:12** (Estágio B).

**Qualidade:** 97 testes unitários (`pytest -q`), TDD nas partes puras.

---

## 3. Mapa do código (`src/ai/`)

Rodam no **PC** (Python 3.12). Convenção: `python src/ai/<script>.py` da raiz do
repo; o `src` é injetado no `sys.path` por cada script.

**Compartilhado sim/carro (`shared/`, mantido compatível com Python 3.6):**
- `shared/image_pipeline.py` — `preprocess(bgr)` → `(3,66,200)` float32, **BGR**,
  CHW, [-1,1] (crop 130/30 → resize 200×66). Fonte única sim/Jetson.
- `shared/lidar_pipeline.py` — `scan_to_sectors_m` (72 setores, **metros**),
  `scan_to_sectors` (normalizado) e **`normalize_sectors_m`** (metros→[0,1], fonte
  única usada no treino E na inferência; `near=0/free=1`).

**Percepção / utilidades:**
- `sim_lidar.py` — nuvem do CARLA → setores em metros (filtro de chão z∈[-1.7,2.0],
  descarta retornos do próprio carro < 0.5 m).
- `metrics.py` — `mae`, `variance_ratio` (detector de "só anda reto") e
  `RouteMetrics` (desvio, p95, off-lane, suavidade do steering, **colisões**).
- `paths.py` — injeta o pacote `agents/` do CARLA (para o `BasicAgent`).

**Dados / treino:**
- `dataset_writer.py` — `EpisodeWriter` grava `frames/NNNNNN.jpg` + `lidar.npy` +
  `labels.csv`; `write_meta`. Formato em §4.
- `dataset_index.py` — `list_episodes`, `split_episodes` (**por episódio**, nunca
  por frame), `build_index`, `sample_weights` (peso 3× nas curvas) e
  **`sample_weights_dual`** (sobre-amostra curva ∪ frenagem).
- `dataset.py` — `SteeringDataset` (Fase 2, `(img, steer)`) e **`DrivingDataset`**
  (Fase 3: `(img, lidar[72] normalizado, alvo[steer,throttle,brake])`).
- `model.py` — `CameraSteeringNet` (Fase 2) e **`DrivingNet`** (Fase 3): braço de
  câmera PilotNet (idêntico, para *warm-start*) + braço Conv1D de LiDAR (padding
  circular) → `[steer(tanh), throttle(sigmoid), brake(sigmoid)]`. ~0.5 M params.
- `warmstart.py` — **`load_camera_backbone`**: copia o braço de câmera (`cnn.*`) do
  `cam_v2.pt` para o `DrivingNet` (transfere a direção provada). Copia **só** `cnn.*`
  mesmo quando a origem já é um `DrivingNet` (ex.: `driving_v3.pt` na Fase 4) — o braço
  de LiDAR nasce zerado **de propósito**: herdá-lo do Town importaria o viés de
  "ignorar o LiDAR" (§5.7) justo no experimento que mede o LiDAR.
- `recovery_schedule.py` — **`RecoveryScheduler`** (puro, sem CARLA): quando cutucar o ego
  e quais frames contam como recuperação. `reset()` por episódio é obrigatório (§5.2).
- `track_ref.py` — centerline própria + desvio lateral da pista custom (a pista não tem
  OpenDRIVE, então `map.get_waypoint()` não funciona lá).
- `model_policy.py` — `ModelSteeringPolicy` (Fase 2, `throttle` fixo) e
  **`DrivingPolicy`** (Fase 3): câmera+LiDAR → `(steer, throttle, brake)`. Tem
  **deadzone de freio** (§5.6) e `ablate_lidar` (para a ablação; LiDAR = "livre").
- `report.py` — `dataset_report` / `print_report` (histograma de steer, % freio).
- `noise.py` — `SteeringNoiseInjector` (injeção de ruído client-side; **não** usado
  na coleta atual — reservado para recuperação/DAgger futuros).

**Expert (`expert/`):**
- `expert/basic_agent_expert.py` — `ExpertPolicy` (CARLA `BasicAgent`, PID lateral
  retunado). **Legado**: o expert de coleta hoje é o autopilot do TM (ver §5.1);
  o BasicAgent fica para um modo de recuperação client-side futuro.

**CLIs principais:**
- `collect.py` — coleta (autopilot). `--recovery` liga os nudges (agora **só com o
  carro em movimento**, a cada 5 s); **`--traffic N`** spawna carros → eventos de
  frenagem.
- `train.py` — treino Fase 2 (câmera) e **`--dual`** (Fase 3: `DrivingNet`, loss por
  eixo, sampler curva+freio, **`--init-from cam_v2.pt`** para warm-start).
- `eval_openloop.py` — MAE + var_ratio; auto-detecta a arquitetura do checkpoint
  (câmera → MAE de steer; `DrivingNet` → **MAE por eixo** steer/throttle/brake).
- `collect_track.py` — **coleta na pista custom (Fase 4)**: Pure Pursuit sobre a centerline
  do `track_ref`, gravando no nosso formato. `--pistas pista1,pista2,pista3` roda várias
  pistas num dataset só; `--recovery` usa o `RecoveryScheduler`.
- `eval_track.py` — **malha fechada na pista custom**: desvio medido contra a centerline
  própria; `--ablate-lidar` para a ablação; `--obstacles N` espalha os `tcc_*`.
- `eval_closedloop.py` — **harness de malha fechada**. Política injetável: expert,
  `--autopilot`, ou `--model CKPT` (auto-seleciona `DrivingPolicy` se o checkpoint
  for `DrivingNet`). **`--traffic N`** spawna trânsito; **`--ablate-lidar`** neutraliza
  o LiDAR. Mede desvio, saída de faixa e **colisão**.

**Ferramentas de avaliação (`scripts/`, rodam no CARLA):**
- `scripts/drive_tour.py` — dirige o modelo por vários pontos de spawn (tour),
  seguindo com a câmera; separa falhas de **cruzamento** vs **pista**. `--fast` =
  sem realtime; `--traffic N` = com trânsito.
- `scripts/braking_test.py` — **teste de frenagem/ablação**: carro parado à frente
  numa reta, mede se para antes de bater. Rode com e sem `--ablate-lidar`.

**Base do simulador (reusado, `src/core/`, `settings/`):** cliente CARLA,
modo síncrono 20 Hz, spawn do ego + sensores, `track_builder` (pista custom).
Refactors feitos: câmera configurável por JSON (`baseSettings.json`: 640×360,
FOV 62.2 = IMX219 real); `requirements.txt` em UTF-8; ego spawna com faróis
apagados e, nos caminhos de IA, o TM **ignora semáforos/placas** (§5.4).

---

## 4. Formato do dataset

```
D:\tcc_data\dataset_vN\
  meta.json                 # camera, lidar_sectors, expert, recovery, versão
  ep_0001\
    frames\000000.jpg       # BGR, 640×360 (como capturado)
    lidar.npy               # (N, 72) float32, METROS (livre = alcance máx)
    labels.csv              # 1 linha por frame, MESMO índice do jpg/lidar
  ep_0002\ ...
```
`labels.csv`: `frame, steer, throttle, brake, v, x, y, yaw, noise_active`.
Tudo alinhado pelo índice (imagem, LiDAR e comando do **mesmo tick síncrono**).

`labels.csv` ganhou `throttle`/`brake` reais (já eram gravados) — o `DrivingDataset`
usa os três eixos como alvo.

Datasets existentes:
- `dataset_v1` — 24k frames, condução limpa (autopilot). **88% reto**. Sem freio.
- `dataset_v2` — 24k frames, **com recuperação**. **64% reto**. Sem tráfego → sem
  freio (dataset da **direção**, usado na Fase 2).
- `dataset_v3` — **48k frames, com tráfego (`--traffic 30`) + recuperação**. **76.5%
  reto**, steer balanceado, **19.5% de frenagem**, mean speed 5.8 m/s. É o dataset da
  Fase 3 (direção + freio).

- `dataset_track_v1` — **Fase 4**: 12 episódios × 1200 frames = **14.400 frames** nas 3
  pistas do estande (4 por pista), expert **Pure Pursuit**. **20.3% reto** (pista fechada
  é bem mais curva que o Town), **30.0% recuperação** (129 teleportes), **0% freio** — o
  Pure Pursuit não freia, decisão de escopo da Fase 4. Mean speed 1.73 m/s.

Checkpoints (`D:\tcc_data\runs\`): `cam_v1/cam_v2.pt` (Fase 2, só-câmera),
`driving_v3.pt` (Fase 3, dual-input), **`driving_track_v1.pt` (Fase 4, as 3 pistas)**.
`driving_v2conv.pt` = dual treinado só no `v2` (usado no debug da direção).

---

## 5. Decisões-chave e por quê

### 5.1 Expert = autopilot do Traffic Manager (não o BasicAgent)
Medimos: o `BasicAgent` dirige mal (serpenteia na reta, corta curva/pega guia —
PID lateral sub-amortecido). O **autopilot do TM** é ~9× mais centrado na reta e
~2× na curva. Então a coleta usa o autopilot, lendo `vehicle.get_control()` como
rótulo. (`eval_closedloop.py --autopilot` = motorista de referência para assistir.)

### 5.2 Dados de recuperação (o conserto que fez a Fase 2 funcionar)
BC puro sofre **covariate shift**: o autopilot dirige sempre centralizado, então o
modelo nunca vê "estou torto". Em execução, erros se acumulam → estado nunca visto
→ bate. **Sintoma medido:** modelo treinado só com dados limpos batia em **12,7 s**.

**Conserto (`collect.py --recovery`):** o carro é empurrado para fora do centro
(deslocamento lateral + ângulo, via `set_transform`, só em reta) e gravamos o
**autopilot voltando ao centro** — ensinando "torto → volta". **Efeito:** o modelo
passou a dirigir **171 s** (≈13×) centrado a 0,08 m.

**Bug achado na Fase 4 (e o motivo do `recovery_schedule.py`):** no `collect_track.py`, o
relógio do último teleporte era zerado **uma vez por pista**, mas o `step` reinicia a cada
episódio — então, do 2º episódio em diante, `step - last_teleport` ficava negativo o
episódio inteiro: **zero cutucão** e, pior, **todos os frames rotulados `noise_active=True`**.
Medido no dataset descartado: 9 dos 12 episódios com **0 teleportes** (maior salto entre
frames = 0.09 m) mas "95% recovery" no relatório. O agendador virou
`ai/recovery_schedule.py` (puro, com `reset()` por episódio e 5 testes). Depois do conserto:
**todos** os episódios com 10–12 teleportes e 30.0% de recuperação. **Lição:** conferir o
dado pelo **deslocamento entre frames**, não pela coluna `noise_active` — a coluna pode mentir.

Ajustes na Fase 3: a recuperação agora só teleporta **com o carro em movimento**
(≥2 m/s) e a cada **5 s** — antes teleportava o ego mesmo **parado freando** atrás do
tráfego, corrompendo o exemplo de frenagem. **Efeito colateral conhecido:** com
bastante dado de recuperação o modelo faz um leve *wiggle* (vai-e-vem) em reta —
inofensivo, correção adiada (ver memória `steering-wiggle-from-recovery`).

### 5.3 Limitação restante: ambiguidade de cruzamento
O modelo ainda erra em **junções complexas**. Motivo: no treino, o autopilot num
cruzamento às vezes vira à esquerda, às vezes reto, às vezes à direita — a **mesma
imagem** com ações diferentes. Uma câmera frontal sozinha não desfaz isso (BC
multi-modal). **Não é falta de dados nem de treino.** Fixes possíveis: (a) comando
de navegação na entrada (*Conditional Imitation Learning*); (b) **irrelevante no
Estágio B** — a pista 1:12 é um circuito fechado, sem cruzamentos de várias vias.

### 5.4 Semáforos e faróis
A pista física 1:12 não tem semáforo; parar no vermelho só ensinaria o modelo a
parar sem motivo. Então, **só para o ego** nos caminhos de IA, o TM ignora
semáforos/placas (`ignore_traffic_lights=True` em `spawn_actor_vehicle`). O ego
também spawna com os **faróis do veículo apagados** (consistência da câmera).

### 5.5 Outras
- LiDAR guardado em **metros** (re-normalizável no treino).
- Split **por episódio** (frames vizinhos são quase idênticos → split por frame
  vazaria a validação).
- Modelo **pequeno + exportável (ONNX)** de propósito, pensando no Jetson.

### 5.6 Deadzone de freio (o conserto que fez o dual dirigir)
O head de freio nunca regride exato a 0. Um `brake` residual (~0.02) aplicado
**junto** com o throttle **segura o carro parado na partida** (medido: `mean_speed=0`;
zerando o freio, acelera a 3.3 m/s). Conserto no `DrivingPolicy`: se `brake < 0.1`
zera o freio; se `brake ≥ 0.1` é freada de verdade e o **throttle vai a 0** (throttle
e brake mutuamente exclusivos).

### 5.7 Teoria do LiDAR: quando ele agrega (e por que a ablação deu inconclusiva)
A Fase 3 provou que o modelo **freia** — mas a **ablação com/sem LiDAR não mostrou
diferença** (6/8 vs 7/8 pararam; = ruído). Ou seja, **a câmera sozinha está fazendo a
frenagem** atrás de carro grande. Dois motivos:
1. **A câmera vê o carro cedo e longe.** Um veículo inteiro parado é fácil na imagem.
2. **O LiDAR só vê até 12 m.** Nos trials em que o ego freou cedo, o carro ficou
   **fora do alcance** do LiDAR o tempo todo → ele reportava "livre" → não tinha o que
   contribuir. Quando a câmera basta, o BC aprende a usar só ela.

**Onde o LiDAR deve provar valor:** obstáculos **pequenos/baixos** — `tcc_cone`,
`tcc_mureta`, `tcc_pessoa2d` da pista custom. Um cone é minúsculo na imagem (pior de
longe) mas dá retorno claro no LiDAR. **Conclusão: a ablação decisiva é na Fase 4**
(pista custom), não no Town com carros grandes. O pipeline dual está pronto para isso.

### 5.7.1 RESOLVIDO na Fase 4 — o LiDAR provou valor (na parede, não no cone)
A ablação na pista custom (`driving_track_v1.pt`, 120 s por pista) **separou de forma
limpa**:

| pista | com LiDAR | LiDAR ablado |
|---|---|---|
| pista1 | dev 0.41 m, **0 colisões**, 1.9 m/s | dev 0.62 m, **1 colisão @ 79.3 s**, 1.7 m/s |
| pista2 | dev 0.41 m, **0 colisões**, 1.6 m/s | dev 0.77 m, **1377 colisões @ 39.6 s**, 0.6 m/s |
| pista3 | dev 0.58 m, **0 colisões**, 1.6 m/s | dev 0.85 m, **1721 colisões @ 24.2 s**, 0.3 m/s |
| **resumo** | **3/3 limpas** | **0/3 limpas** |

As contagens de 1377/1721 com `mean_speed` caindo a 0.3–0.6 m/s são o ego **encostado
na parede raspando** (um evento por tick), não 1700 batidas distintas: bateu e ficou preso.

**Por que aqui separou e no Town não:** o valor do LiDAR veio da **parede/corredor**, não
dos obstáculos pequenos que a §5.7 previa. A pista 1:12 é um corredor estreito de paredes
contínuas (~34/72 setores com retorno o tempo todo), então o LiDAR informa o *lane-keeping*
a cada tick. No Town, o carro-alvo ficava fora do alcance de 12 m e a câmera bastava.

**Ressalva metodológica (importante para a escrita do TCC):** `--ablate-lidar` alimenta a
rede com "tudo livre", entrada que ela **nunca viu no treino** — logo o resultado prova que
*este modelo dual depende do LiDAR*, e **não** que uma rede só-câmera treinada do zero
fracassaria na pista. O controle honesto para a afirmação mais forte é treinar um modelo
só-câmera nos mesmos dados e comparar (§7).

**`offlane=0` nas corridas abladas não é contradição:** as paredes ficam a ~1–2 m do centro,
dentro do limiar de saída de pista (metade da largura = 3.18 m). **Nesta pista o sinal de
falha é `collisions`, não `offlane`.**

---

## 6. Como rodar (cookbook)

Pré-requisitos: venv com deps (`torch` CUDA já instalado). Rodar da raiz do repo.
Use caminhos com `\` no PowerShell (funciona); em Git Bash use `/`.

```powershell
# 1) Coletar dados COM recuperação (o que alimenta o treino)  ~5 min
python src\ai\collect.py --out D:\tcc_data\dataset_v2 --episodes 20 --seconds 60 --recovery --slow 20

# Relatório de um dataset (sem CARLA)
python src\ai\collect.py --report D:\tcc_data\dataset_v2

# 2) Treinar (GPU, ~8 min). Salva o melhor val_MAE.
python src\ai\train.py --data D:\tcc_data\dataset_v2 --out D:\tcc_data\runs\cam_v2.pt --epochs 20

# 3) Avaliar open-loop (sem CARLA): MAE, var_ratio, MAE reta/curva
python src\ai\eval_openloop.py --ckpt D:\tcc_data\runs\cam_v2.pt --data D:\tcc_data\dataset_v2 --split val

# 4) Malha fechada — ASSISTIR o modelo dirigindo (janela do CARLA)
python src\ai\eval_closedloop.py --model D:\tcc_data\runs\cam_v2.pt --realtime --routes 1 --seconds 120

#    Motorista de referência (autopilot), para comparar
python src\ai\eval_closedloop.py --autopilot --realtime --routes 1 --seconds 120

# Testes unitários
python -m pytest -q
```

**Fase 3 (dual-input câmera+LiDAR):**
```powershell
# 1) Coletar COM tráfego + recuperação (gera frenagem)  ~10 min
python src\ai\collect.py --out D:\tcc_data\dataset_v3 --episodes 20 --seconds 120 --traffic 30 --recovery --slow 20

# 2) Treinar o dual com warm-start do modelo de câmera  (GPU)
python -u src\ai\train.py --dual --data D:\tcc_data\dataset_v3 --out D:\tcc_data\runs\driving_v3.pt --init-from D:\tcc_data\runs\cam_v2.pt --epochs 40

# 3) Open-loop (MAE por eixo; auto-detecta DrivingNet)
python src\ai\eval_openloop.py --ckpt D:\tcc_data\runs\driving_v3.pt --data D:\tcc_data\dataset_v3 --split val

# 4) ASSISTIR o modelo dirigindo por vários pontos do mapa (tour)
python scripts\drive_tour.py --model D:\tcc_data\runs\driving_v3.pt --spawns 8 --seconds 25
#    ...COM carros na rua (trânsito):
python scripts\drive_tour.py --model D:\tcc_data\runs\driving_v3.pt --spawns 8 --seconds 25 --traffic 30
#    (--fast = fast-forward na tela; sem --fast = tempo real)

# 5) Ablação de frenagem (prova do LiDAR): rode as DUAS e compare
python scripts\braking_test.py --model D:\tcc_data\runs\driving_v3.pt --trials 8 --gap 18 --fast
python scripts\braking_test.py --model D:\tcc_data\runs\driving_v3.pt --trials 8 --gap 18 --fast --ablate-lidar
```

**Fase 4 (pista custom 1:12 — as 3 pistas do estande):**
```powershell
# 1) Coletar nas 3 pistas com Pure Pursuit + recuperacao  (~3 min, nao e realtime)
.venv\Scripts\python.exe -u src\ai\collect_track.py --out D:/tcc_data/dataset_track_v1 `
    --pistas pista1,pista2,pista3 --episodes-por-pista 4 --seconds 60 --recovery
.venv\Scripts\python.exe src\ai\collect_track.py --report D:/tcc_data/dataset_track_v1

# 2) Treinar o dual com warm-start (copia SO o braco de camera do driving_v3)  ~13 min
.venv\Scripts\python.exe -u src\ai\train.py --dual --data D:/tcc_data/dataset_track_v1 `
    --out D:/tcc_data/runs/driving_track_v1.pt --init-from D:/tcc_data/runs/driving_v3.pt --epochs 40

# 3) Malha fechada nas 3 pistas (--realtime para assistir)
.venv\Scripts\python.exe -u src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt `
    --pistas pista1,pista2,pista3 --seconds 120

# 4) A ablacao do LiDAR (§5.7.1): rode as DUAS e compare `collisions`
.venv\Scripts\python.exe -u src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt --pistas pista1,pista2,pista3 --seconds 120
.venv\Scripts\python.exe -u src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt --pistas pista1,pista2,pista3 --seconds 120 --ablate-lidar
```

No fim do closed-loop o harness imprime, por rota:
`offlane=<saídas de faixa>  collisions=<eventos>  ... FIRST COLLISION @ <s>`.
**Sucesso = `collisions=0`** numa corrida longa.

---

## 7. Próximos passos

- **Cruzamento (§5.3): DECIDIDO — aceito.** Não vamos resolver a ambiguidade de
  cruzamento no Town; a pista custom 1:12 é fechada e não tem cruzamentos.
- **Fase 3: FEITA.** Modelo dual dirige e freia; ablação do LiDAR ficou para a Fase 4
  (§5.7). Pipeline dual-input completo e testado.

- **Fase 4: FEITA.** Ego + sensores spawnam na pista custom, centerline própria
  (`track_ref`), expert Pure Pursuit, `driving_track_v1.pt` dirige as 3 pistas do
  estande em malha fechada (**3/3 limpas**) e a **ablação do LiDAR separou** (§5.7.1).

### Fase 5 — refino final do modelo (*touch-ups*) — ⏸️ ADIADA

> **DECISÃO (Rafael, 2026-07-28): esta fase foi ADIADA.** O modelo atual já é bom o
> suficiente para testar no carrinho; a prioridade passou a ser o **Jetson (Fase 6)**.
> A lista abaixo fica registrada, **na ordem recomendada**, para quando for retomada.

**Ordem recomendada — medir antes de ajustar.** Os itens 5 e 6 são só *medição*: não
mexem no modelo e dizem quais dos outros de fato valem esforço. Fazer o barato primeiro
evita afinar às cegas algo que talvez nem seja o problema:
1. **Obstáculos (item 6)** — o único que roda **sem tocar em código** (`--obstacles N`
   já existe). Espera-se que bata: o modelo nunca viu obstáculo no treino. Isso não é
   regressão, é a medida de quanto o `tcc_*` está fora da distribuição, e é o que
   justifica (ou dispensa) o item 7. É também a previsão original da §5.7, nunca testada.
2. **Partidas fora do waypoint 0 (item 5)** — mudança pequena (`--start-idx`), e é o
   resultado mais fácil de contestar hoje: "3/3 limpas" é sempre do mesmo ponto de
   partida. Transforma o teste em generalização de verdade.
3. **pista3 (item 3)** — com os números acima na mão, dá para saber se ela é pior em
   todas as condições ou só na geometria dela.
4. **Wiggle (item 2) e freio (item 7)** — por último: exigem recoletar ou mexer no loss.
   Não vale pagar esse custo antes de saber se algo mais básico falha.

**Premissa: o `driving_track_v1` já está bom** (3/3 pistas limpas em 420 s acumulados
por pista). Esta fase é para deixá-lo redondo, **não** para reescrevê-lo. **O modelo
continua dual (câmera+LiDAR)** — decisão do Rafael, e o LiDAR é o sensor do carro real.

1. **Controle só-câmera — evidência, não melhoria.** Treinar uma rede **só-câmera** no
   `dataset_track_v1` e rodar o mesmo `eval_track`. É uma rede **descartável, de
   comparação**: o modelo entregue continua sendo o dual. Serve para poder afirmar "o
   LiDAR é **necessário** nesta pista" em vez de só "este modelo dual depende dele"
   (ressalva da §5.7.1) — é a resposta para a banca perguntar "por que LiDAR se a câmera
   já dirige?". **Requer um ajuste:** o `eval_track.py` instancia `DrivingPolicy` fixo,
   precisaria aceitar `ModelSteeringPolicy` (throttle fixo) para um checkpoint só-câmera.
2. **Wiggle do steering em reta** (memória `steering-wiggle-from-recovery`): vem do
   volume de dado de recuperação. Candidatos: aumentar `--recovery-every`, pesar menos
   os frames de recuperação no sampler, ou penalizar *jerk* de steering no loss.
3. **A pista3 é medidamente a pior:** desvio médio **0.58–0.60 m** e p95 **1.52–1.58 m**,
   contra 0.40–0.42 m e p95 ~1.08 m nas pistas 1 e 2 (consistente nas corridas de 120 s
   e 300 s, então é sistemático, não ruído). Investigar se é geometria (curva mais
   fechada) ou falta de dado — e, se for dado, coletar mais episódios só nela.
4. **Velocidade é escolha, não defeito:** o expert roda a `target_speed=2.0` m/s
   (`settings/pistaTCC.json`) e o modelo **acompanha** (1.7–2.1 m/s). Para volta mais
   rápida, subir o `target_speed` e **recoletar** — o modelo imita a velocidade do expert.
5. **Robustez de partida:** hoje o ego sempre spawna no **waypoint 0**. Testar partidas
   de vários pontos da centerline mede generalização de verdade.
6. **Obstáculos `tcc_*`** (a previsão original da §5.7): `eval_track --obstacles N` já
   espalha cones/muretas. **Ressalva:** o Pure Pursuit não desvia deles, então o dataset
   não tem exemplo de contorno — serve como teste de **robustez/frenagem**, não de desvio.
7. **Freio inerte:** o `dataset_track_v1` tem **0% de frenagem**, então o head de brake
   está praticamente morto (MAE 0.006 porque o alvo é sempre ~0). Para controle
   longitudinal real na pista seria preciso dado de frenagem — expert que freie, ou
   obstáculos que forcem parada.

### Fase 6 — Jetson Nano: colocar o modelo no carrinho (A PRÓXIMA)

**Objetivo:** rodar o modelo no **Jetson Nano** dirigindo o RC 1:12 (PCA9685 I2C, **não**
GPIO 33/32). Dividida em **6a** (simulador + export, não precisa do carro) e **6b** (bancada +
runtime). O código de export ainda não existe, mas a fase **não é mais campo aberto**: os
bloqueadores foram investigados e há design escrito (ver abaixo).

**O que já está pronto de propósito para isso:**
- `shared/image_pipeline.py` e `shared/lidar_pipeline.py` são mantidos **compatíveis com
  Python 3.6** justamente para rodar no Jetson. São a **fonte única** sim↔carro: o mesmo
  `preprocess()` e o mesmo `normalize_sectors_m()` do treino. **Reusar, nunca reescrever** —
  qualquer divergência aqui vira erro silencioso de inferência.
- O `DrivingNet` é pequeno (~0.5 M params) e foi desenhado exportável.
- A câmera do sim já usa **FOV 62.2** = IMX219 real.

**Bloqueadores originais — TODOS RESOLVIDOS ou encaminhados (análise de 2026-08-19,
cruzando `hardware/controle_teste.py` com os dados do `dataset_track_v1`):**

1. ✅ **ESCALA 12× — resolvido sem gambiarra.** Como a normalização divide por `max_range`,
   usar **`max_range = 12.0/12 = 1.0 m` no carro** é *matematicamente idêntico* a multiplicar
   as leituras por 12 — e não exige tocar no `shared/`. Parede real a 0.265 m → 0.265
   normalizado, igual ao treino. (Velocidade segue valendo: 2 m/s sim = **0.167 m/s real**.)
2. ✅ **O carrinho TEM LiDAR: o COIN-D6 (WitMotion), 2D 360°.** E o `CoinD6Parser` do
   `hardware/controle_teste.py` já devolve `(angle_deg, dist_m)` — **exatamente** a assinatura
   de `scan_to_sectors_m`. O gap 3D×2D **não é fatal**: `points_to_sectors_m` já achata a nuvem
   (mínimo por setor), e para parede vertical a distância horizontal independe da elevação do
   feixe. O modelo nunca viu 3D — viu 72 números.
3. 🔴 **NOVO e o mais importante: oclusão pela própria carroceria.** Para ver a parede (baixa),
   o sensor precisa ficar baixo; nessa altura o carro bloqueia a traseira. Mascarar a traseira
   **só na inferência** muda 25% dos valores (erro médio **0.177** na escala 0–1) = a condição
   da ablação. **Conserto: retreinar com FOV limitado** — ver Fase 6a abaixo. Montar o LiDAR no
   **para-choque dianteiro** (no centro a carroceria bloqueia nos dois sentidos do eixo).
4. ✅ **8 vs 72 setores:** ficam **72**. O `scan_to_sectors_m` é parametrizado; o HUD do
   hardware usa 8 só para exibição.
5. ✅ **Freio:** o controle longitudinal aprendido está **fora de escopo** (dataset com 0% de
   frenagem — nem o throttle do modelo pararia o carro). Velocidade fixa + parada de emergência
   **em código**. Medido: trava no cone frontal ±10° a **0.25 m real → 0 falsos positivos em
   14.400 frames** (o pior 0.1% da condução normal fica a 0.287 m).
6. 🟡 **Montagem da câmera:** segue valendo conferir altura/pitch reais contra o
   `baseSettings.json` — o crop 130/30 do `preprocess()` assume o enquadramento do sim.

**ONNX validado (2026-08-19) — o Jetson consegue rodar:**
export em **opset 11** OK (1.9 MB), `onnx.checker` OK, **paridade PyTorch↔ONNX 5.96e-08**.
O padding circular do braço de LiDAR **não** virou `Pad(wrap)` exótico: o exportador decompõe em
`Slice`+`Concat`. Ops geradas — `Conv, Elu, Gemm, Flatten, Unsqueeze, Slice, Concat, Constant,
Tanh, Sigmoid` — são todas nativas do TensorRT. O TensorRT 8.x do JetPack 4.6 cobre até opset
13–14, então opset 11 passa com folga (`trtexec --onnx=… --fp16`, rodado **no próprio Jetson**).

#### Fase 6a — modelo 180° + export ONNX (A PRÓXIMA, não precisa do carro)

**Spec:** `docs/superpowers/specs/2026-08-19-ai-fase6a-fov180-onnx-design.md`.
Máscara de FOV parametrizada (`fov_deg`, default 180°) em `shared/lidar_pipeline.py`, usada
**identicamente** no sim e no carro; **72 setores mantidos** (o padding circular só é honesto com
o vetor de 360°); **`fov_deg` gravado no checkpoint** para não haver descasamento silencioso;
retreino a partir do `dataset_track_v1` **sem recoletar** (~20 min de GPU); validação com
`eval_track` (**critério: seguir 3/3 limpas**); e `export_onnx.py` com paridade obrigatória.

#### Fase 6b — carro (depois)

Medições de bancada **antes** do runtime: (1) arco real de oclusão → vira o `fov_deg` do retreino;
(2) zero e sentido do ângulo do COIN-D6; (3) velocidade real a 1600 µs; (4) altura da parede e do
cone × plano do LiDAR. Depois: runtime câmera + LiDAR → `shared/*` → engine TensorRT → PCA9685
(servo ch15 `1500 + steer*200` µs; ESC ch12 em PWM fixo), reusando `watchdog`/`ESC_ARMADO` do
`hardware/controle_teste.py`. Medir **FPS real** (o sim roda a 20 Hz).
Só então: pista física e medição do gap sim-to-real (Fase 7).

---

## 8. Notas operacionais / gotchas

- **Dataset fora do OneDrive** (`D:\tcc_data`) — o repo está no OneDrive; datasets
  grandes lá corrompem/estouram sync. `.gitignore` já ignora `data/`, `*.pt`, `runs/`.
- **CARLA zumbi:** `CarlaUE4.exe` é um stub que gera `CarlaUE4-Win64-Shipping.exe`;
  o harness já mata a árvore no fim. Se sobrar (porta 2000 presa):
  `Get-Process CarlaUE4,CarlaUE4-Win64-Shipping | Stop-Process -Force`.
- **`--report` com caminho `D:\...` no Git Bash** falha (o Git Bash mastiga o
  caminho). Use `D:/...` ou rode no PowerShell.
- **Saída bufferizada** ao rodar em background: use `python -u ...` para ver o
  progresso ao vivo.
- **Modelo só-câmera** (`ModelSteeringPolicy`) dirige com `throttle` fixo; o **dual**
  (`DrivingPolicy`, Fase 3) tem controle longitudinal **aprendido** (acelera/freia).
- **CARLA back-to-back:** lançar um servidor logo após matar o anterior às vezes dá
  **timeout de boot** (o engine ainda estava liberando a porta 2000). Se acontecer:
  matar zumbis (`taskkill /F /IM CarlaUE4-Win64-Shipping.exe`) e relançar.
- **`| grep` mascara o exit code** de um script Python que falhou (o código de saída
  vira o do `grep`). Para ver falha real em background, rode sem pipe e filtre na leitura.
- **Determinismo:** o CARLA não é 100% determinístico entre execuções — a mesma
  política pode pegar rotas/cruzamentos diferentes. Julgue por corridas longas, não
  por um único run curto.
