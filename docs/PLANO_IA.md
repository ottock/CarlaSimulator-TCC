# Plano de Desenvolvimento da IA — Behavioral Cloning Dual-Input (Câmera + LiDAR)

> **TCC:** Utilização de Digital Twins para o Treinamento de Veículos Autônomos
> **Equipe:** Rafael Assanti · Otto Camargo Kuchkarian · Rodrigo Fernandes Faltz · André Felipe Silva Xavier — Orientador: Prof. Carlos Menezes
> **Documento vivo:** atualize os marcadores de status (✅ concluído · 🔄 em andamento · ⬜ não iniciado) conforme o trabalho avança.
>
> **Revisões:** v1.0 (09/07/2026) · v1.1 (09/07/2026) — LiDAR corrigido para COIN-D6; estratégia em 2 estágios (Towns padrão primeiro, pista custom depois); atuação via Jetson.GPIO conforme `Testes de hardware/Controle_do_carro.py` (fonte da verdade de hardware). · **v1.2 (09/07/2026)** — reordenação das fases: **o primeiro passo é o CARLA em malha fechada** (esqueleto do laço com o expert); **toda a conexão de hardware foi consolidada na fase final**, depois de tudo validado em simulação.

---

## 0. Decisões-chave (resumo executivo)

| # | Decisão | Escolha | Justificativa curta |
|---|---------|---------|---------------------|
| D1 | Arquitetura | **DAVE-2 modificada dual-input** (CNN câmera + Conv1D LiDAR) | TransFuser é inviável no Nano 2GB (ver §3) |
| D2 | LiDAR real | **COIN-D6** (360°, 12 m, serial 230400) — parser já validado no repo | Correção da equipe (09/07/2026); `Coin_d6.py` e o parser em `Controle_Teste.py` são reaproveitados |
| D3 | Representação LiDAR | **72 setores de 5°, mínimo por setor, normalizado** | Robusto a ruído/dropout; idêntico em sim e real (§4) |
| D4 | Estratégia de pista | **2 estágios: (A) Towns padrão do CARLA primeiro → (B) pista custom / gêmeo digital depois** de o pipeline estar validado | Decisão da equipe (09/07/2026); no estágio A o autopilot nativo funciona |
| D5 | Expert na coleta | **Estágio A: `BasicAgent` do CARLA** (client-side, permite injeção de ruído) · **Estágio B: Pure Pursuit** sobre waypoints do `track_builder` | Autopilot/TM não roda na pista de props (sem OpenDRIVE) |
| D6 | Escala | Estágio A: escala natural dos Towns · Estágio B: **real ×12** (`escala: "real"`) | Geometria relativa preservada no gêmeo; sem blueprint de buggy 1:12 |
| D7 | Saída do modelo | **2 neurônios: `steer` + eixo longitudinal assinado `a`** → 3 valores derivados (throttle/brake) | O atuador físico É um único canal (ESC); resolve desbalanceamento de freio (§6) |
| D8 | Treino | **PyTorch 2.x no PC com GPU NVIDIA da equipe** | Sem limite de sessão; export ONNX (§7) |
| D9 | Inferência no Nano | **ONNX → TensorRT FP16** (JetPack 4.6.x já traz TRT ≥8.0) | Sem PyTorch no carro; menor memória e latência (§8) |
| D10 | Atuação no carro | **Jetson.GPIO PWM direto: servo pino 33, ESC pino 32** (50 Hz, duty cycle) — fonte da verdade: `Controle_do_carro.py` + `Controle_ESC.py` | Último setup validado pela equipe; scripts PCA9685 viram referência legada |
| D11 | Frequência de controle | **20 Hz** no sim (`fixed_delta_seconds: 0.05`) e no carro | Mesma cadência dos dois lados fecha parte do gap (§8) |
| D12 | Deadband do ESC | **Resolvido na camada de atuação, invisível ao modelo** (§7.5) | O modelo vive só no espaço normalizado [-1,1] |

---

## ⚠️ Regras de segurança (INEGOCIÁVEIS — valem em todas as fases)

- 🛞 **RODAS FORA DO CHÃO** em qualquer teste que envolva ESC/motor. Sempre. Sem exceção.
- 🔴 **KILL SWITCH NA MÃO** sempre que o carro estiver energizado sob controle do Jetson.
- ⚡ **NUNCA 5V em GPIO do Jetson** (lógica 3.3V). Sinal de receptor a 5V exige divisor de tensão (diagrama em `PWM_reader.py`). O servo/ESC recebem o *sinal* do Jetson (3.3V é suficiente); a *alimentação* deles vem do BEC do ESC, nunca do Jetson.
- 🔋 **NUNCA a LiPo 7.4V direto no barrel jack** do Jetson — step-down 5V/5A é obrigatório.
- 🔌 **GND comum** entre Jetson, ESC e servo.
- 🐕 **Watchdog em software**: qualquer travamento do laço ⇒ ESC em neutro (duty 7.5% = 1500 µs) e servo centralizado — padrão já validado em `Testes de hardware/Controle_Teste.py`, deve ser portado para o runtime da IA.
- 🧱 **Teto de throttle em software**: duty 8.5% (1700 µs) até segunda ordem (o limitador do rádio original não existe mais).

---

## 1. Auditoria do código existente

### 1.1 O que já existe e a IA reutiliza

| Componente | Arquivo | Reuso pela IA |
|------------|---------|---------------|
| Cliente CARLA com retry, modo síncrono, lifecycle | `src/core/carlaClient/world_manager.py` | Base da coleta (modo síncrono já correto para dataset) |
| Spawn de ego + autopilot via Traffic Manager | `world_manager.py` (`spawn_actor_vehicle`) | Ponto de partida do modo coleta no Estágio A |
| Wrapper de câmera RGB | `src/core/carlaClient/camera_manager.py` | Reusa a estrutura de callback; **precisa refatorar** (§1.3) |
| Wrapper de LiDAR + detecção de obstáculos | `src/core/carlaClient/lidar_manager.py` | Reusa parsing da nuvem; a config vem do JSON (bom) |
| **Parser COIN-D6 validado** | `Testes de hardware/Coin_d6.py` + classe `CoinD6Parser` em `Controle_Teste.py` | 🔑 Vira `jetson/sensors/coin_d6.py` quase sem mudanças (buffer AA55, dist mm LE, scan fecha com ≥300 pts) |
| Diagnóstico do LiDAR (bytes/s, scans/s, pts/scan) | `Controle_Teste.py` (print 1×/s) | Mede as specs reais do COIN-D6 para configurar o CARLA (§4.3) |
| **Atuação validada via Jetson.GPIO** | `Controle_do_carro.py` (servo, pino 33) + `Controle_ESC.py` (ESC, pino 32) | 🔑 Base de `jetson/actuators/gpio_pwm_actuator.py` (duty cycle a 50 Hz) |
| Calibração do ESC em µs (neutro 1500 / início movimento 1600 / teto 1700 / freio 1400) | `Controle_PWM_Vel.py`, `Controle_Teste.py` | A calibração é propriedade do ESC — vale independente do driver PWM |
| Padrão de segurança: watchdog, rampas, hold-to-move, estado seguro no `finally` | `Controle_Teste.py` | Mesmo padrão no laço de inferência |
| Pipeline GStreamer da CSI IMX219 | `Testes de hardware/Camera_Teste.py` | Base de `jetson/sensors/csi_camera.py` |
| Pista custom + waypoints (Estágio B) | `src/core/carlaClient/track_builder.py` | `_montar()` já devolve `waypoints` (x, y, heading) — espinha dorsal do Pure Pursuit e da métrica de desvio lateral |
| Config JSON validada · Logger · Visualização | `src/utils/config.py`, `src/core/logger/`, `src/presentation/visualization.py` | Reuso direto |

> Scripts PCA9685 (`Controle_PWM_Steering.py`, `Controle_PWM_Vel.py`, `Teste.py`, laço de `Controle_Teste.py`) ficam como **referência legada** — a lógica de µs, rampas e segurança migra; a escrita I2C não.

### 1.2 O que NÃO existe (a IA precisa criar)

- ⬜ Modo coleta no `main.py` (ego + autopilot + gravação sincronizada) — o `data_collector.py` antigo (hoje só um `.pyc` órfão em `src/ai/__pycache__/`) gravava apenas câmera + controles, **sem LiDAR**.
- ⬜ Módulo de pré-processamento **compartilhado** sim/carro (restrição: sintaxe Python 3.6).
- ⬜ Modelo, dataset, treino, export ONNX, avaliação open-loop e closed-loop.
- ⬜ Runtime de inferência no Jetson (captura + inferência + atuação + segurança) — as peças existem separadas nos testes; falta a integração com o modelo.
- ⬜ (Estágio B) Expert Pure Pursuit + densificação da centerline da pista custom.

### 1.3 O que precisa de refatoração ANTES da IA

| Item | Problema | Ação |
|------|----------|------|
| `camera_manager.py` | Resolução (800×600), FOV (90°) e pose **hardcoded** — não batem com a IMX219 | Ler `camera` do settings.json (como o LiDAR já faz); valores em §7.2 |
| `lidar_manager.py` (config) | Perfil atual (32 canais, FOV -25..+15, 100 m) é um Velodyne, não um COIN-D6 | Novo perfil planar no JSON (§4.3) |
| `utils/config.py` | Schema não cobre `camera`, `track`, nem os novos blocos `expert`/`collect` | Estender `_SCHEMA` |
| `requirements.txt` | Está em UTF-16 e fixa `logging==0.4.9.6` — pacote PyPI da era Python 2 que **sombreia a stdlib** (`.venv/Lib/site-packages/logging/`) | Regravar em UTF-8, **remover** `logging`, adicionar `torch`, `opencv-python`, `pandas` (só no PC) |
| Agents do CARLA | `BasicAgent`/`BehaviorAgent` vivem em `PythonAPI/carla/agents` do pacote CARLA, fora do wheel pip | Copiar o pacote `agents/` para `src/` (ou adicionar ao `sys.path`) — tarefa da Fase 0 |
| `main.py` modo pista (Estágio B) | Com `track.enabled`, retorna antes de spawnar o ego — pista e veículo são mutuamente exclusivos hoje | Corrigir quando o Estágio B começar (Fase 4) |
| Waypoints esparsos (Estágio B) | `_montar()` devolve 1 ponto por peça | `densificar_centerline()` interpolando retas e arcos (~0.5 m ×12) — Fase 4 |

---

## 2. Estrutura de arquivos proposta

```
src/ai/                          # roda no PC (Python 3.12) — treino e sim
├── __init__.py
├── shared/                      # ⚠️ TAMBÉM roda no Jetson (Python 3.6!)
│   ├── __init__.py              #    proibido: walrus, dataclasses, match; f-string ok
│   ├── image_pipeline.py        # crop + resize + normalização — ÚNICA fonte da verdade
│   └── lidar_pipeline.py        # scan polar → vetor de 72 setores
├── expert/
│   ├── basic_agent_expert.py    # Estágio A: BasicAgent + injeção de ruído
│   ├── centerline.py            # Estágio B: densificar_centerline() do track_builder
│   └── pure_pursuit.py          # Estágio B: expert da pista custom
├── collect.py                   # CLI: roda episódios e grava dataset
├── dataset.py                   # torch Dataset + balanceamento + augmentation
├── model.py                     # DualInputPilot
├── train.py                     # CLI de treino
├── export_onnx.py               # checkpoint → .onnx (opset 11)
├── eval_openloop.py             # MAE/relatórios no test set
└── eval_closedloop.py           # modelo dirige no CARLA; mede métricas de §10

jetson/                          # roda no carro (Python 3.6) — NÃO importa torch
├── drive.py                     # laço principal 20 Hz
├── sensors/
│   ├── csi_camera.py            # GStreamer (de Camera_Teste.py), thread própria
│   └── coin_d6.py               # parser do COIN-D6 (de Coin_d6.py), thread + buffer de setores
├── actuators/
│   └── gpio_pwm_actuator.py     # Jetson.GPIO pinos 33/32, duty 50 Hz, deadband map, teto 1700 µs
├── safety.py                    # watchdog, kill por teclado, limites, estado seguro
└── trt_runner.py                # carrega engine TensorRT e faz o forward

data/                            # ⚠️ FORA do OneDrive (ex.: D:\tcc_data) — ver risco R9
└── dataset_v1/
    ├── meta.json                # fov, config lidar (medida real!), estágio/escala, versão do pipeline
    ├── ep_0001/
    │   ├── frames/000000.jpg    # 640×360 (pré-proc final acontece no treino)
    │   ├── lidar.npy            # (N, 72) float32, METROS (sem normalizar)
    │   └── labels.csv           # frame, steer, a_long, throttle, brake, v, x, y, yaw
    └── ep_0002/ ...
docs/PLANO_IA.md                 # este arquivo
```

Regra de ouro: **`src/ai/shared/` é importado pelos dois mundos.** Teste de unidade compara byte a byte a saída do pipeline no PC e no Jetson (fixtures salvas em `tests/fixtures/`).

---

## 3. Decisão de arquitetura

### 3.1 Comparação

| Critério | DAVE-2 dual-input (recomendado) | TransFuser | Alternativa: ResNet18 + MLP |
|---|---|---|---|
| Parâmetros | ~0.8 M | ~66 M (2 backbones + transformers) | ~11 M |
| Latência no Nano 2GB (FP16 TRT) | **~3–8 ms** ✅ | Estimado > 300 ms, **se couber** ❌ | ~30–60 ms ⚠️ |
| Memória de inferência | < 100 MB | > 1.5 GB só de pesos+ativações — **não cabe** em 2 GB compartilhados com SO+câmera ❌ | ~400 MB ⚠️ |
| Dados de treino | 10⁵ frames (nosso alvo) | Treinado com centenas de milhares de frames de datasets do Leaderboard; faminto por dados ❌ | 10⁵ ok |
| Complexidade p/ equipe de 4 sem base ML | Baixa — arquitetura de 2014, dezenas de tutoriais | Alta — fusão por atenção, auxiliary losses, waypoint prediction | Média |
| Robustez sim-to-real | Boa com randomização; modelo pequeno decora menos textura | Alta em sim, mas irrelevante se não roda no alvo | Boa |
| Compatível com ONNX→TRT 8.x do JetPack 4.6 | ✅ trivial | ⚠️ atenção/LayerNorm em opset alto, TRT 8.0 sofre | ✅ |

### 3.2 Veredito

**TransFuser é inviável no Jetson Nano 2GB — descartado.** Não é margem apertada: é uma ordem de grandeza de memória e latência. O Nano 2GB tem ~1.2–1.5 GB realmente livres após SO + pipeline de câmera; TransFuser assume GPU de desktop.

**Recomendação: DAVE-2 modificada dual-input.** O braço de visão é o PilotNet clássico (5 convs); o LiDAR entra por um braço Conv1D **circular** (o vetor de setores é topologicamente um anel — setor 0 é vizinho do 71); os embeddings concatenam num MLP com duas cabeças. Se, na Fase 3, o modelo saturar (não aprender desvio de obstáculo), o upgrade controlado é trocar o braço de visão por MobileNetV2-0.5 pré-treinada — ainda cabe folgado no Nano — **sem** mudar o resto do pipeline.

### 3.3 Arquitetura concreta (`src/ai/model.py`)

```python
import torch
import torch.nn as nn

class DualInputPilot(nn.Module):
    """Entrada: img (B,3,66,200) normalizada [-1,1]; lidar (B,1,72) em [0,1].
    Saída: (B,2) -> [steer, a_long], ambos em [-1,1] via tanh."""
    def __init__(self, lidar_bins: int = 72, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(                      # DAVE-2 / PilotNet
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Flatten(),                              # -> 64*1*18 = 1152
        )
        self.lidar = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, padding_mode="circular"), nn.ELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ELU(),
            nn.Flatten(),                              # -> 32*18 = 576
        )
        self.head = nn.Sequential(
            nn.Linear(1152 + 576, 256), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ELU(),
            nn.Linear(64, 2), nn.Tanh(),
        )

    def forward(self, img, lidar):
        z = torch.cat([self.cnn(img), self.lidar(lidar)], dim=1)
        return self.head(z)        # [:,0]=steer  [:,1]=a_long
```

---

## 4. Representação do LiDAR

### 4.1 Opções e tradeoff

| Representação | Tamanho | Prós | Contras |
|---|---|---|---|
| **72 setores de 5°, distância mínima (recomendado)** | 72 floats | Robusto a dropout/ruído; invariante à densidade de pontos (a maior diferença sim×real!); barato; mapeia 1:1 no COIN-D6 e no CARLA | Perde detalhe fino (~irrelevante: pista tem cones/muretas, não geometria fina) |
| 360 bins de 1° | 360 floats | Mais resolução | O COIN-D6 entrega ~300–500 amostras/volta irregulares → bins vazios viram ruído estrutural que o CARLA não reproduz |
| BEV 2D (grade de ocupação) | ~64×64 | Espacialmente rico | CNN 2D extra no Nano; sobra pouco ganho para 1 plano de varredura |
| Range image | 1×N | Padrão em LiDAR 3D | Sem eixo vertical no COIN-D6, degenera no caso dos bins |

**Por que "mínimo por setor"**: para desvio de obstáculo, o que importa é *a coisa mais próxima em cada direção*. Mínimo é conservador e estável entre sensores com densidades diferentes.

### 4.2 Encoding compartilhado (`src/ai/shared/lidar_pipeline.py` — Python 3.6!)

```python
import numpy as np

N_SETORES = 72
ALCANCE_MAX_M = 12.0    # COIN-D6; no sim Estágio B usar 144 m e dividir por 12 (escala)

def scan_para_setores(angulos_deg, dist_m, n=N_SETORES, alcance=ALCANCE_MAX_M):
    """Vetor fixo de setores. IGUAL no sim e no carro.
    angulos_deg: 0 = frente do carro, crescendo anti-horário (visto de cima).
    dist_m <= 0 ou > alcance => sem leitura => setor fica 'livre' (=alcance)."""
    setores = np.full(n, alcance, dtype=np.float32)
    a = np.asarray(angulos_deg, dtype=np.float32) % 360.0
    d = np.asarray(dist_m, dtype=np.float32)
    ok = (d > 0.05) & (d <= alcance)
    idx = (a[ok] / (360.0 / n)).astype(np.int32) % n
    np.minimum.at(setores, idx, d[ok])
    return setores / alcance                      # [0,1]; 1.0 = livre
```

No carro, a thread do COIN-D6 (parser de `Coin_d6.py`) mantém um **buffer rolante por setor** (última leitura conhecida + timestamp); o laço de 20 Hz lê o buffer. No sim, os pontos de ticks consecutivos (= 1 volta completa) alimentam a mesma função — **semântica idêntica**: "varredura mais recente", atualizada mais devagar que o controle.

⚠️ **Convenção de ângulo:** `lidar2.py` aplicava correção `(angle+180)%360` — o zero do COIN-D6 não é a frente do sensor. Medir o offset de montagem no chassi (apontar um obstáculo à frente e ler o setor) e gravar em `meta.json`; o runtime aplica o mesmo offset.

### 4.3 Casamento sim ↔ real (obrigatório — senão o gap aumenta de graça)

O COIN-D6 não tem datasheet confiável à mão — **as specs saem do diagnóstico já implementado** (`Controle_Teste.py` imprime `bytes/s`, `scans_fechados`, `pts_exibidos` 1×/s). Tarefa da Fase 0: rodar 60 s e preencher a coluna "medido".

| Parâmetro | COIN-D6 (real) | CARLA (`settings.json`, perfil novo) |
|---|---|---|
| Planos | 1 (2D) | `channels: 1`, `upper_fov: 0`, `lower_fov: 0` |
| Rotação | **medir** (placeholder: ~8 Hz) | `rotation_frequency:` = valor medido |
| Amostras/s | **medir** = scans/s × pts/scan (placeholder: ~4000) | `points_per_second:` = valor medido |
| Alcance | 12 m (filtro `> 12000 mm` no código) | Estágio A: `range: 12` · Estágio B: `range: 144` (×12) e dividir por 12 antes do encoding |
| Ruído | dropouts reais (dist 0/1 descartada no parser) | `noise_stddev: 0.01–0.1` + `dropoff_general_rate: 0.10` |
| Montagem | Altura real do sensor no chassi (medir) | `carla.Location(z=...)` no lugar do z=2.0 hardcoded (Estágio A: altura de teto de carro; Estágio B: altura real ×12) |

---

## 5. Estratégia de coleta de dados no CARLA

### 5.1 Dois estágios, dois experts

**Estágio A — Towns padrão (Town01/Town02/Town03):** valida o pipeline inteiro (coleta → treino → closed-loop) no ambiente onde o CARLA dá tudo de graça: malha OpenDRIVE, autopilot, tráfego, semáforos.
Expert do ego: **`BasicAgent`** (do pacote `agents` do CARLA) em vez do autopilot do Traffic Manager, por um motivo decisivo: o BasicAgent calcula o controle **no cliente**, então podemos (a) ler o rótulo limpo e (b) aplicar um comando perturbado — a injeção de ruído de §5.2 fica trivial. O TM continua controlando os veículos de tráfego. Fallback v0 (para o primeiro teste de gravação): autopilot do TM lendo `vehicle.get_control()` — já suportado por `spawn_actor_vehicle`.

**Estágio B — pista custom / gêmeo digital (×12):** quando o pipeline estiver validado (Fases 0–4 aprovadas), troca-se o expert por **Pure Pursuit** sobre a centerline densificada do `track_builder` — o autopilot não funciona lá (a pista de props não tem OpenDRIVE). O modelo que vai para o carro físico é treinado neste estágio (fine-tuning a partir dos pesos do Estágio A).

```python
# src/ai/expert/basic_agent_expert.py (esboço — Estágio A)
from agents.navigation.basic_agent import BasicAgent

agent = BasicAgent(vehicle, target_speed=25)          # km/h
agent.set_destination(destino.location)
...
controle_limpo = agent.run_step()                     # RÓTULO gravado
controle_aplicado = perturbar(controle_limpo, ruido)  # o que o carro executa
vehicle.apply_control(controle_aplicado)
```

```python
# src/ai/expert/pure_pursuit.py (esboço — Estágio B)
def expert_step(veh, centerline, v_alvo_ms=8.0, lookahead_m=6.0):
    tf = veh.get_transform()
    alvo = ponto_a_frente(centerline, tf.location, lookahead_m)
    dx, dy = para_frame_local(alvo, tf)
    curvatura = 2.0 * dy / (lookahead_m ** 2)
    steer = float(np.clip(curvatura * GANHO_STEER, -1.0, 1.0))
    erro_v = v_alvo_ms - norm(veh.get_velocity())
    a_long = float(np.clip(KP_V * erro_v, -1.0, 1.0))
    d_frente = min(setores_frontais())                # lidar sim (±30°)
    if d_frente < D_FREIO:
        a_long = -1.0
    elif d_frente < D_DESVIO:
        steer = combinar_com_desvio(steer, setores)
    return steer, a_long
```

### 5.2 Recuperação de desvio: injeção de ruído (não DAgger — por enquanto)

O erro clássico do behavioral cloning: o expert nunca sai da linha central, então o modelo nunca vê "estou torto, como volto?". Solução barata com expert client-side:

- Em **30% do tempo de coleta**, soma-se ao steering *aplicado* uma perturbação senoidal + ruído (`±0.3`, períodos de 2–4 s).
- O **rótulo gravado é sempre a saída limpa do expert** (a correção), nunca o comando perturbado.
- Frames com o carro fora da faixa (Estágio A: distância ao waypoint da faixa > meia-largura via `map.get_waypoint()`; Estágio B: desvio à centerline) são descartados no pós-processamento.

**DAgger fica como extensão opcional** (Fase 3b): o modelo dirige, o expert rotula os estados visitados, re-treina. Só vale a pena se a injeção de ruído não bastar.

### 5.3 Quanto e o quê coletar

**Estágio A (Towns) — alvo ~130 mil frames ≈ 1h50 a 20 Hz:**

| Bloco | Conteúdo | Episódios | Frames (~) |
|---|---|---|---|
| A1 — Condução limpa | Rotas aleatórias em Town01 e Town02 × 5 condições de sol/nuvem (`sun_altitude_angle` ∈ {15, 30, 50, 70, 90}, `cloudiness` ∈ {0, 40, 80}), sem tráfego | 25 rotas de ~2 min | 60 k |
| A2 — Tráfego e frenagem | Mesmas rotas com 20–30 veículos + 10–20 pedestres: seguir líder, parar em semáforo/cruzamento — eventos de freio "de graça" | 20 | 40 k |
| A3 — Recuperação | Blocos A1/A2 com injeção de ruído ligada | 15 | 30 k |

**Estágio B (pista custom ×12) — alvo ~80 mil frames (fine-tuning):**

| Bloco | Conteúdo | Episódios | Frames (~) |
|---|---|---|---|
| B1 — Voltas limpas | oval + quadrado + octogono × 2 sentidos × 5 iluminações | 20 × 3 voltas | 35 k |
| B2 — Obstáculos | `obstacles.count` ∈ {2, 4, 6}, seeds diferentes | 20 | 25 k |
| B3 — Recuperação + casos difíceis | ruído ligado; partida deslocada/±30°; obstáculo após curva cega; 10 episódios "reta + mureta no fim" (frenagem limpa) | 15 | 20 k |

- **Frenagem**: A2/B2 garantem eventos, mas serão ~5–8% dos frames — **rebalancear no treino** (§7.4), não coletar mais.
- **"90% reto"**: medir o histograma de `steer` no relatório do dataset (`collect.py --report`) e **downsample de retas no sampler** (§7.4), nunca deletando dados do disco.
- Coleta é barata: CARLA headless via `-RenderOffScreen` roda a noite toda por script. 130 k frames ≈ 5–6 GB em JPG 640×360.

### 5.4 O que cada amostra grava (formato em §2)

`frame, steer_expert, a_long_expert, throttle, brake, v_ms, x, y, yaw, ruido_ativo` + `frames/NNNNNN.jpg` + linha correspondente em `lidar.npy` (72 floats, metros; ÷12 no Estágio B). Gravação **no tick síncrono** — imagem, LiDAR e controle do mesmo `world.tick()` (sem dessincronia).

---

## 6. Design das saídas e função de perda

### 6.1 Por que 2 saídas e não 3

O atuador longitudinal físico é **um único canal do ESC**: acima do neutro acelera, abaixo freia. Throttle e brake nunca coexistem. Um **eixo assinado `a ∈ [-1,1]`** (positivo = acelerar, negativo = frear) é isomórfico ao hardware, elimina o caso inconsistente "throttle e brake juntos" e reduz o desbalanceamento (frenagem vira a cauda negativa de uma distribuição contínua, não uma classe rara).

Os **3 valores pedidos pelo TCC são derivados por construção**:
`throttle = max(a, 0)` · `brake = max(-a, 0)` · `steer = steer`.
No CARLA: `carla.VehicleControl(throttle=max(a,0), brake=max(-a,0), steer=steer)`. No carro: §7.5.

### 6.2 Perda

```python
loss = 1.0 * F.smooth_l1_loss(pred[:, 0], alvo_steer) \
     + 0.5 * F.smooth_l1_loss(pred[:, 1], alvo_a_long)
```

- **Smooth L1** (Huber): MSE pune demais outliers de rotulagem; L1 puro tem gradiente ruim perto de 0.
- **Peso 1.0 : 0.5** — steering é o que mantém o carro na pista; longitudinal erra com consequência menor (ponto de partida; ajustar se o freio ficar fraco na Fase 3).
- Desbalanceamento de frenagem tratado por **amostragem ponderada**, não pela perda (§7.4) — mais estável que pesar a loss.

---

## 7. Pipeline de treinamento e mitigação do gap sim-to-real

### 7.1 Pré-processamento de imagem — idêntico nos dois mundos

`src/ai/shared/image_pipeline.py` (Python 3.6):

```python
import cv2
import numpy as np

# Entrada: BGR uint8 640x360 (sim: resize do frame CARLA; carro: saída do GStreamer)
CORTE_TOPO = 130      # remove céu/horizonte (calibrar com dados reais na Fase 6!)
CORTE_BASE = 30       # remove para-choque/chassi

def preprocessar(img_bgr_640x360):
    img = img_bgr_640x360[CORTE_TOPO:360 - CORTE_BASE, :, :]   # 200 x 640
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0                 # [-1, 1]
    return np.transpose(img, (2, 0, 1))                        # CHW, BGR
```

**BGR nos dois lados** (OpenCV é BGR nativo; o CARLA entrega BGRA → drop do alfa já é BGR). Nunca converter para RGB em só um dos lados — é o tipo de bug silencioso que fabrica gap.

### 7.2 Casamento da câmera sim ↔ IMX219

| Parâmetro | Carro (IMX219) | CARLA (novo bloco `camera` no settings.json) |
|---|---|---|
| Modo de captura | **1640×1232 @ 30 fps** (binned 2×2, **FOV total**) — ⚠️ o modo 1280×720 CORTA o sensor e muda o FOV! | — |
| FOV horizontal | 62.2° | `fov: 62.2` (o atributo do CARLA é FOV horizontal) |
| Aspect no modelo | Redimensionar captura → 640×360 no GStreamer (`nvvidconv`) | `image_size_x: 640, image_size_y: 360` |
| Posição | Medir no chassi (ex.: 8 cm acima do solo, 5 cm à frente do eixo) | Estágio A: pose de para-brisa comum · Estágio B: posição real × 12 (ex.: `z=0.96, x=0.60`), `pitch` igual ao da montagem |
| Flip | `flip-method=2` se montada invertida | — (garantir imagem "de pé" nos dois lados) |

### 7.3 Randomização de domínio (no treino, não na coleta extra)

Augmentation em `dataset.py` (aplicada on-the-fly):

| Augmentation | Faixa | Observação |
|---|---|---|
| Brilho/contraste/gamma | ±30% / ±25% / 0.7–1.4 | Cobre exposição automática da IMX219 |
| Ruído gaussiano + blur leve | σ≤8 / kernel 3 | Sensor real é mais ruidoso que o render |
| Sombras sintéticas (polígonos escuros) | 0–2 por imagem | Sombras reais do laboratório |
| Color jitter de matiz | ±8° | Diferença de cor entre material real e textura do prop |
| **Flip horizontal** | p=0.5 | Negar `steer`; **espelhar o vetor LiDAR**: `setores_flip = setores[::-1]` rolado 1 (setor 0 permanece frontal) |
| LiDAR: dropout de setores | 0–8 setores → "livre" | Reflexos/absorção no mundo real |
| LiDAR: ruído multiplicativo | ±3% | Ordem de grandeza típica de LiDAR de triangulação |

Mais os elementos estruturais: clima variado na coleta (§5.3), obstáculos/tráfego em posições aleatórias, **pose da câmera com jitter ±2 cm/±2° entre episódios** (tolerância a erro de montagem).

**Experimento do TCC:** treinar o modelo B sem randomização e comparar o gap dos dois (§10.3) — a randomização deixa de ser fé e vira resultado.

### 7.4 Dataset, split e balanceamento

- **Split por episódio, jamais por frame** (frames do mesmo episódio são quase idênticos → val vazaria). Estágio A: `train` = episódios de Town01; `val` = 20% dos episódios (held-out); **`test` = todos os episódios de Town02** (cidade nunca vista). Estágio B: `test` = layout octogono, excluído inteiramente do treino.
- **Balanceamento** via `WeightedRandomSampler`:
  - `|steer| < 0.05` (reta) → peso 1
  - `|steer| ≥ 0.05` (curva) → peso 3
  - `a_long < -0.1` (freio) → peso 4 (acumula com o de curva)
- Relatório automático pós-coleta: histograma de steer, % frames de freio, distância percorrida, frames descartados fora da faixa.

### 7.5 A ponte de atuação e o deadband (a resposta explícita)

**O modelo nunca vê µs nem duty cycle.** Ele vive no espaço normalizado [-1,1] — o mesmo do CARLA. A descontinuidade 1500–1600 µs do ESC é resolvida por uma função determinística na camada de atuação do Jetson, que **remapeia o eixo contínuo do modelo diretamente para a faixa ativa do ESC, pulando a banda morta**. Hardware conforme `Controle_do_carro.py`/`Controle_ESC.py`: **Jetson.GPIO, servo no pino 33, ESC no pino 32, 50 Hz** (duty% = µs / 20000 × 100 → 7.5% = 1500 µs).

```python
# jetson/actuators/gpio_pwm_actuator.py
import Jetson.GPIO as GPIO

SERVO_PIN, ESC_PIN = 33, 32          # fonte da verdade: Controle_do_carro.py / Controle_ESC.py
PWM_HZ = 50                          # periodo 20000 us

STEER_L_US, STEER_C_US, STEER_R_US = 1300, 1500, 1700   # faixa conservadora calibrada;
                                                          # (o sweep 1000-2000 us de
                                                          #  Controle_do_carro.py e teste
                                                          #  de bancada, nao limite de uso)
NEUTRO_US       = 1500               # duty 7.5
FIM_DEADBAND_US = 1600               # duty 8.0 -- inicio do movimento real (medido)
TETO_US         = 1700               # duty 8.5 -- teto de seguranca de bancada
FREIO_US        = 1400               # duty 7.0 -- freio (Controle_ESC.py); nunca re
A_ZERO          = 0.05               # zona morta de DECISAO do modelo (histerese)

GPIO.setmode(GPIO.BOARD)
GPIO.setup([SERVO_PIN, ESC_PIN], GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, PWM_HZ)
esc = GPIO.PWM(ESC_PIN, PWM_HZ)
servo.start(7.5)
esc.start(7.5)                       # arming: segurar neutro ~2 s antes de acelerar

def us_para_duty(us):
    return us * PWM_HZ / 10000.0     # 1500 us -> 7.5

def acao_para_us(steer, a):
    steer_us = int(STEER_C_US + steer * (STEER_R_US - STEER_C_US))
    if a > A_ZERO:      # acelerar: [A_ZERO..1] -> [1600..1700] (deadband PULADA)
        frac = (a - A_ZERO) / (1.0 - A_ZERO)
        thr_us = int(FIM_DEADBAND_US + frac * (TETO_US - FIM_DEADBAND_US))
    elif a < -A_ZERO:   # frear: comando unico de freio (nunca re)
        thr_us = FREIO_US
    else:
        thr_us = NEUTRO_US
    return steer_us, thr_us

def aplicar(steer, a):
    steer_us, thr_us = acao_para_us(steer, a)
    servo.ChangeDutyCycle(us_para_duty(steer_us))
    esc.ChangeDutyCycle(us_para_duty(thr_us))
```

Consequências e cuidados:
- Em sim, `a=0.1` produz aceleração fraca; no carro, produz o movimento mínimo real (≥1600 µs). A **função de transferência é monotônica e sem descontinuidade do ponto de vista do modelo** — a banda morta não existe no espaço de ação dele.
- O gap residual (curva de torque real ≠ modelo do CARLA) é medido na Fase 6 comparando velocidade real × `a` e, se preciso, corrigido com uma LUT calibrada em bancada (rodas no ar).
- Freio é **um estado discreto** (1400 µs) — coerente com o ESC RC (abaixo do neutro = freio; nunca usar o duplo-clique de ré).
- ⚠️ **`GPIO.PWM` do Jetson.GPIO é PWM por software** — sujeito a jitter quando a CPU carrega (inferência!). Os pinos 32/33 são justamente os dois com **PWM por hardware** no Nano: habilitar via `sudo /opt/nvidia/jetson-io/jetson-io.py` (pinmux) e **validar o jitter na Fase 5** com `PWM_reader.py`/osciloscópio antes de armar o ESC. Se o jitter passar de ±20 µs sob carga, o plano B é voltar ao PCA9685 (código legado pronto).
- Rampas de suavização (`STEER_STEP_US`, `THROTTLE_STEP_US` de `Controle_Teste.py`) permanecem na camada de atuação como proteção mecânica.

### 7.6 Treino — ponto de partida

```bash
python src/ai/train.py --data D:/tcc_data/dataset_v1 \
    --epochs 40 --batch 128 --lr 1e-4 --weight-decay 1e-5 \
    --dropout 0.3 --early-stop 6
```

| Hiperparâmetro | Valor inicial |
|---|---|
| Otimizador | Adam, lr 1e-4, cosine decay |
| Batch | 128 (cabe em qualquer GPU ≥ 4 GB p/ este modelo) |
| Épocas | 40 com early stopping (paciência 6) na MAE de steer no val |
| Critério de checkpoint | menor `val_steer_MAE` |
| Fine-tuning Estágio B | partir dos pesos do Estágio A, lr 3e-5, 15 épocas |

Frameworks: **PyTorch 2.x + Python 3.12 no PC** (o mesmo venv do CARLA serve). Export: **ONNX opset 11** (`export_onnx.py`) — opset baixo de propósito, para compatibilidade garantida com o TensorRT 8.0/8.2 do JetPack 4.6.x.

---

## 8. Deploy e laço de inferência no Jetson

### 8.1 Conversão do modelo

```bash
# No PC:
python src/ai/export_onnx.py --ckpt runs/best.pt --out pilot.onnx   # opset 11, batch 1

# No Jetson (TensorRT já vem no JetPack — nada para instalar):
/usr/src/tensorrt/bin/trtexec --onnx=pilot.onnx --saveEngine=pilot_fp16.trt \
    --fp16 --workspace=256
```

Runtime: `jetson/trt_runner.py` usa os bindings Python do TensorRT (inclusos no JetPack, compatíveis Py3.6) + `pycuda`. Dependências no carro: `numpy`, `cv2` (do JetPack, com GStreamer), `pyserial`, `Jetson.GPIO`, `tensorrt`, `pycuda`. **Zero PyTorch no carro.**
Plano B (se TRT emperrar): wheel NVIDIA de PyTorch 1.10 para JetPack 4.6/Py3.6 — funciona, mas come ~700 MB de RAM; usar só como fallback de depuração.

### 8.2 Orçamento de tempo real — alvo 20 Hz (50 ms/frame)

| Etapa | Budget | Nota |
|---|---|---|
| Frame da câmera (thread GStreamer, sempre o mais recente) | 0 ms no laço | appsink `drop=true max-buffers=1` |
| Vetor LiDAR (thread do COIN-D6, buffer de setores) | 0 ms no laço | COIN-D6 fecha scans mais devagar que 20 Hz — ok, obstáculos a ≥1 m com carro a ≤1 m/s |
| Pré-proc imagem (crop+resize+norm) | ~3 ms | `image_pipeline.py` |
| Inferência TRT FP16 | ~5–10 ms | modelo de 0.8 M params |
| Mapeamento p/ duty + `ChangeDutyCycle` (2 canais) | ~1 ms | Jetson.GPIO |
| **Total** | **~15 ms** | folga de 35 ms → 20 Hz confortável; até 30 Hz possível |

**Mínimo viável**: 15 Hz para um 1:12 a ≤1 m/s (percorre ≤6.7 cm entre decisões). Abaixo de 10 Hz, não armar o ESC.

### 8.3 Política de falhas (implementada em `jetson/safety.py`)

| Evento | Detecção | Ação |
|---|---|---|
| Frame de câmera velho | timestamp > 200 ms | ESC neutro, servo centro, log |
| Scan LiDAR velho | timestamp > 500 ms | reduz teto de `a` para 50%; > 1 s ⇒ neutro |
| Laço travou (modelo/OS stall) | duração do frame > 250 ms (mesmo watchdog do `Controle_Teste.py`) | neutro imediato + centro |
| Saída do modelo NaN/fora de [-1,1] | checagem por frame | neutro + encerrar |
| Tecla de pânico (espaço/q) e Ctrl+C | pygame/terminal | estado seguro no `finally` (neutro → `GPIO.cleanup()`), como já validado |
| **Kill switch físico** | humano | **sempre na mão — regra inegociável** |

### 8.4 Memória no Nano 2GB

- Rodar `drive.py` **headless** (sem janela; HUD opcional via `--debug` só em bancada).
- Desabilitar GUI do Ubuntu nos testes (`sudo systemctl set-default multi-user.target`) — libera ~350 MB.
- zram habilitado (padrão do JetPack) + swapfile de 2 GB no cartão para picos de build de engine.
- Construir o engine TRT **uma vez** e versionar o `.trt` fora do git (é específico da GPU do Nano).

---

## 9. Fases incrementais (cada uma com critério de aprovação)

> **Princípio de execução (v1.2):** o **primeiro passo é o CARLA rodando em malha fechada** — montamos o *esqueleto* do laço de controle (sentir → decidir → atuar → `world.tick()` → medir) já na Fase 0, com o **expert** como primeira política plugada. O modelo entra depois no mesmo ponto de plugagem, sem mudar a estrutura do laço; assim tudo é testável em malha fechada desde o dia 1. **A conexão física com o hardware é a ÚLTIMA coisa** (Fases 5–6) e só acontece depois de todo o sistema validado em simulação.
>
> Regra geral: não iniciar uma fase sem a anterior aprovada. **Fases 0–4 são 100% simulação — nenhum risco físico.** Segurança física entra só na Fase 5.
> Estágio A = Fases 0–3 (Towns). Estágio B = Fase 4 (pista custom / gêmeo digital). **Hardware = Fases 5–6.** Experimentos = Fase 7.

### Fase 0 — Fundações + esqueleto do closed-loop ⬜ · *o primeiro passo: CARLA em malha fechada*
> Antes de qualquer modelo, o laço de controle roda ponta a ponta no CARLA com o expert dirigindo. Esse harness (`eval_closedloop.py`) recebe uma política `policy(obs) -> (steer, a)` e é reutilizado sem mudança quando o modelo substituir o expert (Fase 2+). Os sensores já são lidos e transformados aqui (embora o `BasicAgent` dirija por waypoints, não por eles) — assim o caminho de dados inteiro é exercitado desde o começo.
- [ ] `camera_manager.py` configurável via JSON (fov 62.2, 640×360, pose)
- [ ] Perfil LiDAR "coin_d6" no settings (§4.3): planar, range 12 m
- [ ] **Medir specs reais do COIN-D6** (rodar diag de `Controle_Teste.py` 60 s → scans/s e pts/scan) e preencher §4.3 + `meta.json`
- [ ] Pacote `agents/` do CARLA acessível no projeto (copiar de `CARLA_0.9.16/PythonAPI/carla/agents` ou `sys.path`)
- [ ] `shared/image_pipeline.py` + `shared/lidar_pipeline.py` com testes de unidade (fixtures)
- [ ] `requirements.txt` regravado (UTF-8, sem `logging`)
- [ ] **Harness de malha fechada `eval_closedloop.py`**: laço síncrono `obs = ler_sensores() → (steer, a) = policy(obs) → vehicle.apply_control(...) → world.tick()`, com watchdog de saída de faixa e métricas de §10.2. **Primeira política = `expert_policy` (BasicAgent).**
- **✅ Aprovada quando:** **o expert dirige o ego em malha fechada em Town01** por ≥ 3 rotas de ~2 min sem sair da faixa, com o harness imprimindo as métricas de §10.2; câmera/LiDAR nos specs novos aparecem na visualização; testes do `shared/` passam. *(É literalmente "o CARLA funcionando em malha fechada" — o marco pedido.)*

### Fase 1 — Pipeline de coleta (Towns) ⬜
- [ ] `collect.py`: episódios com `BasicAgent`, gravação síncrona (imagem + LiDAR + rótulos do mesmo tick)
- [ ] Injeção de ruído com rótulo limpo (§5.2) + descarte de frames fora da faixa
- [ ] `collect.py --report`: histograma de steer, % freio, vídeo com steer sobreposto (checagem de sincronia)
- **✅ Aprovada quando:** blocos A1+A3 gravados (≥ 80 k frames), relatório sem dessincronia visível e histograma de steer com massa nas curvas.

### Fase 2 — Modelo câmera (steering) em malha fechada (Towns) ⬜
- [ ] `dataset.py`, `model.py` (só braço da câmera), `train.py`, `eval_openloop.py`
- [ ] Plugar o modelo treinado no **harness da Fase 0** (o modelo vira a `policy`; controla o `steer`, `a_long` fixo do expert) — métricas via `map.get_waypoint()`
- **✅ Aprovada quando:** *open-loop* `val_steer_MAE < 0.07` **E** desvio-padrão das predições ≥ 60% do dos rótulos (pega o "modelo que só anda reto" cedo — risco R4); **E** *closed-loop* completa **3 rotas de ~2 min em Town01 sem sair da faixa** (desvio lateral médio < 0.5 m) e ≥ 1 rota em Town02 (cidade nunca vista).

### Fase 3 — Dual-input completo (LiDAR + longitudinal, Towns) ⬜
- [ ] Coleta bloco A2 (tráfego/frenagem); dataset Estágio A completo (~130 k frames)
- [ ] Modelo completo (2 saídas), sampler balanceado, augmentations de LiDAR
- [ ] (Opcional 3b) rodada DAgger se a frenagem/desvio ficar abaixo do alvo
- **✅ Aprovada quando:** closed-loop com tráfego: **para atrás de veículo líder / pedestre em ≥ 8/10 cenários novos (seeds não vistos)** sem colisão, mantendo os critérios da Fase 2 em pista livre. Ablation registrada: modelo sem LiDAR falha mais nos mesmos cenários (prova que o LiDAR é usado).

### Fase 4 — Gêmeo digital: pista custom ×12 (closed-loop CARLA) ⬜
> Só aqui, com o pipeline inteiro já validado nos Towns, migramos para a pista custom (o gêmeo digital). Este é o modelo que irá para o carro.
- [ ] `main.py`: pista custom **com** ego + sensores (remover o `return` precoce do modo track)
- [ ] `densificar_centerline()` + `pure_pursuit.py`; calibração até o expert dar 5 voltas limpas em cada preset
- [ ] Coleta blocos B1–B3 (~80 k frames); fine-tuning a partir dos pesos da Fase 3
- **✅ Aprovada quando:** closed-loop na pista custom: **3 voltas no oval e no quadrado sem sair**, octogono (nunca visto) ≥ 1 volta; obstáculos: para/desvia em ≥ 8/10 seeds novos. **Fim de toda a etapa de simulação.**

---

> ## 🔌 FASE FINAL — HARDWARE (Fases 5–6)
> **A conexão física com o carro começa aqui e em nenhum momento antes.** Todo o sistema já foi validado em malha fechada no simulador (Fases 0–4). Duas etapas obrigatórias por segurança: **primeiro bancada com rodas no ar, só então pista real**. Reler as **Regras de Segurança** no topo do documento antes de energizar qualquer coisa.

### Fase 5 — Hardware, bancada Jetson (RODAS NO AR 🛞) ⬜
- [ ] `jetson/sensors/coin_d6.py` (porte do parser validado de `Coin_d6.py`) + visualização de sanidade
- [ ] `jetson/actuators/gpio_pwm_actuator.py` (pinos 33/32, §7.5) portado de `Controle_do_carro.py`/`Controle_ESC.py`
- [ ] **PWM por hardware nos pinos 32/33** via `jetson-io` + medição de jitter sob carga de inferência (`PWM_reader.py`/osciloscópio); jitter > ±20 µs ⇒ plano B PCA9685 (risco R6)
- [ ] Export ONNX → engine TRT; `trt_runner.py` reproduz saídas do PyTorch (tolerância 1e-2)
- [ ] `drive.py` integrado: câmera + LiDAR + inferência + GPIO, **`--dry-run` primeiro** (ESC desarmado, só telemetria)
- [ ] Medir: Hz do laço, p99 de latência, RAM, temperatura em 10 min contínuos
- **✅ Aprovada quando:** ≥ 20 Hz sustentado por 10 min, p99 < 50 ms, sem OOM; **com rodas no ar e kill switch na mão**, servo/motor respondem plausivelmente quando se mostra um obstáculo ao LiDAR e se aponta a câmera para uma cena de pista (freia ao aproximar objeto frontal).

### Fase 6 — Hardware, pista real tethered (🔴 kill switch, teto 1700 µs) ⬜
- Pré-requisito de compra: **2ª bateria LiPo + step-down 5V/5A** (risco R5). Até lá, testes com o Jetson em fonte de bancada e cabo acompanhando o carro em trechos curtos.
- [ ] Calibração fina: offset do zero do servo, LUT de velocidade × `a` (§7.5), corte de horizonte (`CORTE_TOPO`) com imagens reais, offset angular do COIN-D6 (§4.2)
- [ ] Gravar dataset real pequeno (condução manual pelo teclado, mesmos logs) → avaliação open-loop real (§10.2)
- **✅ Aprovada quando:** **1 volta completa autônoma na pista real sem intervenção**, com velocidade na faixa 1600–1650 µs.

---

### Fase 7 — Experimentos do TCC ⬜
- [ ] Protocolo completo de medição do gap (§10.3), N ≥ 10 corridas por condição
- [ ] Ablations: com vs. sem randomização de domínio; com vs. sem LiDAR; modelo Estágio A vs. Estágio B na pista real (valor do gêmeo digital!)
- [ ] Consolidação de tabelas/figuras para a monografia
- **✅ Aprovada quando:** tabelas do §10.3 preenchidas com intervalos (média ± desvio) e discussão escrita.

---

## 10. Métricas de avaliação

### 10.1 Open-loop (baratas, rodam a cada treino)
- `MAE_steer`, `MAE_a_long` em val e test (test = cidade/layout nunca visto).
- Razão de variância `σ(pred)/σ(alvo)` — detector de colapso para a média.
- MAE por regime: reta / curva / evento de freio (a média global esconde o que importa).

### 10.2 Closed-loop no CARLA (`eval_closedloop.py`, mesmo harness para expert e modelo)
Por corrida (fixar seeds; N ≥ 10 por condição):
- **Taxa de rota/volta completa** (%); **saídas de faixa/pista** por corrida (Towns: via `map.get_waypoint()`; pista custom: via centerline);
- **Desvio lateral** médio e p95 (m; ÷12 no Estágio B para reportar em escala real);
- **Colisões** por corrida; **sucesso de parada/desvio** (%);
- Tempo de rota (proxy de suavidade).

### 10.3 Quantificação do gap sim-to-real (o fenômeno de pesquisa)

Mesmo layout físico e virtual (o gêmeo digital do Estágio B), mesmas posições de obstáculos (medidas com trena e reproduzidas no `track_builder` com posições fixas — adicionar modo `obstacles.fixed` no JSON):

| Métrica | Sim | Real | Gap |
|---|---|---|---|
| M1 — MAE open-loop do modelo sobre dados gravados | test set sim (pista custom) | dataset real da Fase 6 (rótulo = comando humano) | `MAE_real − MAE_sim` |
| M2 — taxa de volta completa | N=10 | N=10 | `Δ` pontos percentuais |
| M3 — intervenções humanas /volta | 0 por definição | contadas | valor absoluto |
| M4 — desvio lateral médio | via centerline | vídeo zenital (celular no alto + marcações na pista; anotação manual por amostragem) | `Δ` cm |
| M5 — sucesso de parada por obstáculo | %/10 | %/10 | `Δ` |

**Gap relativo** por métrica: `(perf_sim − perf_real) / perf_sim`. Os experimentos centrais do TCC:
1. Matriz **{com, sem} randomização × {sim, real}** — se a randomização reduzir o gap relativo, o resultado está demonstrado quantitativamente.
2. **Modelo Estágio A (só Towns) vs. Estágio B (gêmeo digital)** avaliados na pista real — mede diretamente o valor do gêmeo digital, que é o título do TCC.

Complemento (análise de entrada): comparar distribuições sim × real de brilho/contraste das imagens pré-processadas e de estatísticas dos vetores LiDAR — localiza *onde* está o gap (percepção vs. dinâmica).

---

## 11. Registro de riscos

| # | Risco | Prob. | Impacto | Mitigação |
|---|-------|-------|---------|-----------|
| R1 | **OOM no Nano 2GB** durante inferência | Média | Alto | Modelo 0.8 M params; TRT FP16 sem PyTorch; headless (`multi-user.target`); zram+swap; teste de resistência de 10 min na Fase 5 |
| R2 | **Python 3.6 bloqueia libs modernas** | Alta (certa) | Alto | Fronteira dura: treino no PC (3.12), carro só com numpy/cv2/TRT/pyserial/Jetson.GPIO; `shared/` escrito em sintaxe 3.6; validação: `python3.6 -m py_compile` dos arquivos de `shared/` e `jetson/` |
| R3 | **Specs do COIN-D6 sem datasheet** (rotação, pts/s, precisão) | Média | Médio | Parser já validado ✅; medir empiricamente com o diag existente (Fase 0) e configurar o CARLA com os valores medidos; encoding por setores é tolerante a densidade |
| R4 | **Modelo aprende a "ir reto"** | Alta | Alto | Sampler ponderado (§7.4); injeção de ruído (§5.2); *gate* objetivo na Fase 2 (razão de variância ≥ 0.6) |
| R5 | **Uma única LiPo** — sem andar destetherado | Certa | Médio | Compra da 2ª bateria + step-down 5V/5A **antes da Fase 6** (item de orçamento, responsável definido já); enquanto isso, bancada com fonte + trechos tethered |
| R6 | **Jitter do PWM por software** (Jetson.GPIO) sob carga de inferência | Média | Alto | Pinos 32/33 têm PWM por hardware — habilitar via `jetson-io` (Fase 5); medir jitter com `PWM_reader.py`; plano B: PCA9685 (código legado pronto, era o setup anterior) |
| R7 | Dados insuficientes / de má qualidade | Média | Alto | Coleta programática barata (roda à noite); relatório automático do dataset; começar treino com 80 k frames na Fase 2 e crescer |
| R8 | **Deadband/ESC variam com carga da bateria** | Média | Médio | Recalibrar 1500/1600/1700 µs com bateria cheia e a 50%; LUT de velocidade (§7.5); sempre reportar tensão da LiPo nos logs de corrida |
| R9 | **Dataset dentro do OneDrive** (sync trava, corrompe, estoura cota) | Alta | Médio | `data/` fora do OneDrive (ex.: `D:\tcc_data`); no repo só `meta.json` de exemplo; `.gitignore` para `*.npy`, `*.jpg` de dataset, `*.trt`, `*.onnx` |
| R10 | **Modelo do Estágio A não transfere para o carro** (Towns ≠ pista 1:12) | Alta | Baixo (por design) | É esperado — o Estágio A valida o *pipeline*, não o modelo final; o modelo embarcado vem do Estágio B (gêmeo digital); a diferença entre os dois vira o experimento 2 de §10.3 |
| R11 | Física ×12 não escala (dinâmica ≠ geometria) | Média | Médio | Velocidades baixas nos dois mundos (regime quase-cinemático); o TCC **mede** esse gap (M1–M5) em vez de escondê-lo — vira resultado |
| R12 | Fidelidade visual do gêmeo (cores/materiais dos props ≠ pista real) | Média | Médio | Randomização de cor/brilho (§7.3); foto real vs. render lado a lado na Fase 6; ajustar texturas no Blender se a diferença for gritante |
| R13 | Curva de aprendizado de ML da equipe | Média | Médio | Este plano fixa decisões e valores iniciais; fases pequenas com critérios objetivos; pair programming nas Fases 2–4 |

---

## 12. Comandos de referência (colar e usar)

```bash
# ── PC: coleta (Estágio A — Towns) ───────────────────────────
python src/main.py settings/baseSettings.json --mode collect --episodes 10 --noise 0.3 \
    --out D:/tcc_data/dataset_v1
python src/ai/collect.py --report D:/tcc_data/dataset_v1      # histogramas + sync check

# ── PC: coleta (Estágio B — pista custom, na Fase 4) ─────────
python src/main.py settings/pistaTCC.json --mode collect --episodes 10 --noise 0.3 \
    --out D:/tcc_data/dataset_v2

# ── PC: treino e avaliação ───────────────────────────────────
python src/ai/train.py --data D:/tcc_data/dataset_v1 --epochs 40 --batch 128 --lr 1e-4
python src/ai/eval_openloop.py --ckpt runs/best.pt --data D:/tcc_data/dataset_v1 --split test
python src/ai/eval_closedloop.py --ckpt runs/best.pt --settings settings/baseSettings.json --routes 3

# ── PC: export ───────────────────────────────────────────────
python src/ai/export_onnx.py --ckpt runs/best.pt --out pilot.onnx

# ── Jetson: build do engine e condução ───────────────────────
/usr/src/tensorrt/bin/trtexec --onnx=pilot.onnx --saveEngine=pilot_fp16.trt --fp16 --workspace=256
python3 jetson/drive.py --engine pilot_fp16.trt --dry-run     # ESC desarmado, só telemetria
python3 jetson/drive.py --engine pilot_fp16.trt --arm         # 🛞 RODAS NO AR + 🔴 KILL SWITCH
```

---

## 13. Próximos passos imediatos

1. ⬜ Commitar este plano (`docs/PLANO_IA.md`) e revisar em equipe (30 min): validar D1–D12.
2. ⬜ Abrir as tarefas da **Fase 0** (uma issue por checkbox) — foco no **esqueleto de malha fechada** (o CARLA dirigindo com o expert é o primeiro entregável).
3. ⬜ Encomendar **2ª LiPo + step-down 5V/5A** (destrava a Fase 6 lá na frente — lead time longo, pedir já mesmo sendo a última fase).
4. ⬜ Medir e registrar em `meta.json`: pose real da câmera e do COIN-D6 no chassi (cm), offset angular do LiDAR e specs medidas (scans/s, pts/scan).
5. ⬜ Habilitar PWM por hardware nos pinos 32/33 (`jetson-io`) — pré-requisito barato que evita o risco R6 lá na frente.
