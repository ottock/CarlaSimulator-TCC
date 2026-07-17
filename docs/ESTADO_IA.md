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
| 4 | Pista custom (gêmeo digital) | ⬜ **Próxima** — onde o LiDAR deve provar valor |
| 5–6 | Hardware (Jetson / pista real) | ⬜ Deferido |

**Marco da Fase 2 (`cam_v2.pt`):** modelo só-câmera dirige **~171 s** centrado a
~0.08 m no Town01. Open-loop `val_MAE 0.044`, `var_ratio 1.00`.

**Marco da Fase 3 (`driving_v3.pt`):** modelo **dual-input (câmera+LiDAR)** que
**dirige E freia**. Open-loop `MAE_steer 0.031`, `MAE_brake 0.067`, `var_ratio ~0.96`.
Num *tour* por 12 spawns de Town01, **10/12 dirigiram 25 s limpos** (desvio de faixa
~0.1 m); as únicas batidas em pista aberta foram em **cruzamento** (§5.3). No teste de
frenagem (carro parado à frente), parou sem bater em **~75–87%** dos trials.

**Descoberta honesta sobre o LiDAR (§5.7):** a ablação com/sem LiDAR **não** mostrou
dependência — a **câmera sozinha basta** para frear atrás de carro grande. O valor do
LiDAR deve aparecer nos **obstáculos pequenos/baixos** da pista custom (Fase 4), que
são difíceis para a câmera e fáceis para o LiDAR.

**A ressalva (Fases 2–3):** ainda erra em **cruzamentos complexos** (§5.3) — limite
conhecido do BC mono-câmera, **inexistente na pista fechada 1:12** (Estágio B).

**Qualidade:** 83 testes unitários (`pytest -q`), TDD nas partes puras.

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
  `cam_v2.pt` para o `DrivingNet` (transfere a direção provada).
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

Checkpoints (`D:\tcc_data\runs\`): `cam_v1/cam_v2.pt` (Fase 2, só-câmera),
`driving_v3.pt` (Fase 3, dual-input). `driving_v2conv.pt` = dual treinado só no `v2`
(usado no debug da direção).

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

No fim do closed-loop o harness imprime, por rota:
`offlane=<saídas de faixa>  collisions=<eventos>  ... FIRST COLLISION @ <s>`.
**Sucesso = `collisions=0`** numa corrida longa.

---

## 7. Próximos passos

- **Cruzamento (§5.3): DECIDIDO — aceito.** Não vamos resolver a ambiguidade de
  cruzamento no Town; a pista custom 1:12 é fechada e não tem cruzamentos.
- **Fase 3: FEITA.** Modelo dual dirige e freia; ablação do LiDAR ficou para a Fase 4
  (§5.7). Pipeline dual-input completo e testado.

1. **Fase 4 — pista custom (gêmeo digital), a próxima:**
   - Corrigir o fluxo para **spawnar ego + sensores NA pista custom** (hoje o
     `track_builder` só monta e exibe — ver o design em `docs/superpowers/specs/`).
   - Derivar uma **centerline própria** da geometria do `track_builder` (a pista não
     tem malha OpenDRIVE, então `map.get_waypoint()` **não** funciona lá — a métrica
     de desvio e o teleporte de recuperação precisam vir da centerline própria).
   - Expert **Pure Pursuit** sobre essa centerline; coletar → treinar (fine-tuning do
     `driving_v3`) → **ablação do LiDAR nos obstáculos `tcc_*`** (onde ele deve
     finalmente ganhar da câmera, §5.7).
2. **Hardware (deferido):** Jetson Nano, ONNX→TensorRT, atuação (o hardware atual da
   equipe usa **PCA9685 I2C**, não GPIO 33/32), pista real, medição do gap sim-to-real.

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
