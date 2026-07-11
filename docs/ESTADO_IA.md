# Estado da IA — Carro Autônomo 1:12 (Digital Twin no CARLA)

> Documento de referência do workstream de IA. Objetivo: orientar pessoas e IAs
> sobre **o que já foi feito, como rodar, e o que vem a seguir**.
> Branch de trabalho: `feat/ai-fase0-malha-fechada`.

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
`D:\tcc_data\{dataset_v1, dataset_v2, runs\*.pt}`.

---

## 2. Estado atual (o que funciona)

| Fase | O que é | Status |
|------|---------|--------|
| **0** | Malha fechada com o expert (harness) | ✅ Feito e validado |
| **1** | Pipeline de coleta de dados | ✅ Feito e validado |
| **2** | Modelo só-câmera (direção) em malha fechada | ✅ **Dirige** (com 1 ressalva: cruzamentos) |
| 3 | Dual-input (LiDAR + acelera/freia) | ⬜ Próxima |
| 4 | Pista custom (gêmeo digital) | ⬜ |
| 5–6 | Hardware (Jetson / pista real) | ⬜ Deferido |

**Resultado da Fase 2 (o marco atual):** o modelo (`cam_v2.pt`) **dirige ~171 s
em malha fechada centrado a ~0.08 m da faixa** no Town01, pilotado 100% pela rede
(câmera → direção). Métricas open-loop: `val_MAE 0.044`, `var_ratio 1.00`.

**A ressalva:** ele ainda erra em **cruzamentos complexos** (ver §5.3). Isso é um
limite conhecido do BC só-câmera e **não existe na pista fechada 1:12** (Estágio B).

**Qualidade:** 62 testes unitários (`pytest -q`), TDD nas partes puras.

---

## 3. Mapa do código (`src/ai/`)

Rodam no **PC** (Python 3.12). Convenção: `python src/ai/<script>.py` da raiz do
repo; o `src` é injetado no `sys.path` por cada script.

**Compartilhado sim/carro (`shared/`, mantido compatível com Python 3.6):**
- `shared/image_pipeline.py` — `preprocess(bgr)` → `(3,66,200)` float32, **BGR**,
  CHW, [-1,1] (crop 130/30 → resize 200×66). Fonte única sim/Jetson.
- `shared/lidar_pipeline.py` — `scan_to_sectors_m` (72 setores, **metros**) e
  `scan_to_sectors` (normalizado = metros/alcance).

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
  por frame), `build_index`, `sample_weights` (peso 3× nas curvas).
- `dataset.py` — `SteeringDataset` (torch): frame → `(img, steer)`.
- `model.py` — `CameraSteeringNet` (PilotNet/DAVE-2, ~0.3 M params, img→steer,
  tanh). Pequeno de propósito (Jetson/ONNX depois).
- `model_policy.py` — `ModelSteeringPolicy(obs) -> (steer, throttle, brake)`.
  Câmera-only: o modelo dá o `steer`; `throttle` fixo (`--model-throttle`).
- `report.py` — `dataset_report` / `print_report` (histograma de steer, % freio).
- `noise.py` — `SteeringNoiseInjector` (injeção de ruído client-side; **não** usado
  na coleta atual — reservado para recuperação/DAgger futuros).

**Expert (`expert/`):**
- `expert/basic_agent_expert.py` — `ExpertPolicy` (CARLA `BasicAgent`, PID lateral
  retunado). **Legado**: o expert de coleta hoje é o autopilot do TM (ver §5.1);
  o BasicAgent fica para um modo de recuperação client-side futuro.

**CLIs principais:**
- `collect.py` — coleta (autopilot). `--recovery` liga os nudges de recuperação.
- `train.py` — treino (SmoothL1, Adam, cosine, `WeightedRandomSampler`,
  early-stop, salva o menor `val_MAE`).
- `eval_openloop.py` — MAE + var_ratio + MAE por regime (reta vs curva).
- `eval_closedloop.py` — **harness de malha fechada**. Política injetável:
  expert (default), `--autopilot` (motorista de referência), ou `--model CKPT`
  (a rede). Mede desvio, saída de faixa e **colisão**.

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

Datasets existentes:
- `dataset_v1` — 24k frames, condução limpa (autopilot). **88% reto**.
- `dataset_v2` — 24k frames, **com recuperação** (`--recovery`). **64% reto**
  (muito mais exemplos de correção). **É o dataset a usar.**

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

**Conserto (`collect.py --recovery`):** a cada ~2,5 s o carro é empurrado para fora
do centro (deslocamento lateral + ângulo, via `set_transform`, só em reta) e
gravamos o **autopilot voltando ao centro** — ensinando "torto → volta". **Efeito:**
o modelo passou a dirigir **171 s** (≈13×) centrado a 0,08 m.

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

No fim do closed-loop o harness imprime, por rota:
`offlane=<saídas de faixa>  collisions=<eventos>  ... FIRST COLLISION @ <s>`.
**Sucesso = `collisions=0`** numa corrida longa.

---

## 7. Próximos passos

1. **Decidir o cruzamento** (§5.3): implementar comando de navegação (Conditional
   IL) para fechar Town01 inteira **ou** aceitar a Fase 2 e migrar para o Estágio B,
   onde o problema não existe.
2. **Fase 3 — dual-input:** os dados já têm `lidar.npy`. Falta: braço Conv1D de
   LiDAR no `model.py` + saída longitudinal (throttle/brake, hoje é fixo); coletar
   tráfego para eventos de frenagem; ablação com/sem LiDAR.
3. **Fase 4 — pista custom (gêmeo digital):** expert Pure Pursuit sobre os
   waypoints do `track_builder`; fine-tuning; é o modelo que iria para o carro.
4. **Hardware (deferido):** Jetson Nano, ONNX→TensorRT, atuação (conferir GPIO 33/32
   vs PCA9685 — o hardware atual da equipe usa PCA9685), pista real, medição do gap
   sim-to-real.

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
- **Modelo só-câmera** dirige com `throttle` fixo (`--model-throttle`); o controle
  longitudinal (acelera/freia aprendido) vem na Fase 3.
- **Determinismo:** o CARLA não é 100% determinístico entre execuções — a mesma
  política pode pegar rotas/cruzamentos diferentes. Julgue por corridas longas, não
  por um único run curto.
