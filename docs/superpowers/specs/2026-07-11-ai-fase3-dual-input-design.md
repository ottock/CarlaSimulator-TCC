# Fase 3 — Dual-input (câmera + LiDAR) com saída longitudinal

**Data:** 2026-07-11
**Branch:** `feat/ai-fase3-dual-input`
**Pré-requisito:** Fases 0–2 concluídas (ver [docs/ESTADO_IA.md](../../ESTADO_IA.md)). O modelo
só-câmera `cam_v2.pt` dirige em malha fechada ~171 s centrado a ~0.08 m em Town01.

## Objetivo

Substituir o modelo só-câmera por um modelo **dual-input** que enxerga câmera **e** LiDAR e
controla os três comandos nativos do CARLA — `steer`, `throttle`, `brake` — para **frear atrás
de veículos** sem perder a qualidade de direção em pista livre.

## Escopo (decisões tomadas neste brainstorm)

| # | Decisão | Escolha |
|---|---------|---------|
| 1 | Ambiente de treino/validação | **Towns com tráfego dinâmico** (plano original), não a pista custom |
| 2 | Estratégia de treino | **Warm-start**: reaproveita o braço de câmera do `cam_v2.pt` |
| 3 | Obstáculos de frenagem | **Só veículos** (sem pedestres nesta fase) |
| 4 | Arquitetura de fusão | **Modelo único fundido** (Abordagem A): câmera + LiDAR → cabeça única de 3 saídas |
| 5 | Codificação do LiDAR na rede | **Conv1D com padding circular** sobre os 72 setores |

**Fora de escopo:** pedestres/walkers; pista custom (Fase 4); export ONNX/TensorRT e deploy no
Jetson (plano de hardware). O design mantém o modelo pequeno e com dois inputs bem definidos
para esse export futuro ser direto.

## Achado que reduz o trabalho

O pipeline de dados **já foi construído para dual-input** nas Fases 1–2:

- [dataset_writer.py](../../../src/ai/dataset_writer.py) já grava `lidar.npy` (N×72, metros) e as
  colunas `throttle`/`brake`/`v` no `labels.csv`, tudo alinhado por tick síncrono.
- [dataset_index.py](../../../src/ai/dataset_index.py) `build_index()` já devolve por frame:
  `image`, `lidar` (path do `.npy`), `row` (índice na array = nº do frame), `steer`, `throttle`,
  `brake`.
- [world_manager.py](../../../src/core/carlaClient/world_manager.py) já tem
  `spawn_random_vehicles(world, actor_list, count, traffic_manager)`.

Ou seja: **formato de dados e indexação não mudam**. A Fase 3 é modelo + treino + avaliação, mais
um pequeno acréscimo de tráfego na coleta.

## Arquitetura e componentes

### 1. Modelo — `DrivingNet` (nova classe em `src/ai/model.py`)

`CameraSteeringNet` **permanece intacta** (compatibilidade Fase 2 e ablation de referência).
Nova classe:

```
img   (B, 3, 66, 200) ─► CNN PilotNet (5 conv, IDÊNTICA à CameraSteeringNet) ─► flatten ─┐
lidar (B, 72)         ─► LidarArm (Conv1D×2–3, padding circular) ─► flatten ─────────────┴─► concat
                                                                                              │
                                                          FC (concat → 100 → 50) ────────────┘
                                                                       │
                                            ┌──────────────────────────┼──────────────────────────┐
                                        steer=Tanh (−1..1)      throttle=Sigmoid (0..1)     brake=Sigmoid (0..1)
```

- **Braço de câmera** = **mesma** `nn.Sequential` de conv da `CameraSteeringNet` (mesmos nomes de
  submódulo), para que o warm-start seja uma cópia direta de `state_dict` das chaves `cnn.*`.
- **Braço de LiDAR**: entrada `(B, 1, 72)`; `Conv1d` com **padding circular** (o setor 71 é
  vizinho do setor 0). Ex.: `Conv1d(1,16,5)→ELU→Conv1d(16,32,3)→ELU→Flatten`. Pequeno de
  propósito.
- **Cabeça**: `LazyLinear(100)→ELU→Dropout→Linear(100,50)→ELU`, depois **três cabeças lineares
  separadas** (steer/throttle/brake) com as ativações acima. Saídas separadas deixam a ablation e
  a leitura de métricas por eixo triviais.
- Contagem de parâmetros mantida na casa de ~0.3–0.5 M (exportável).

### 2. Codificação e normalização do LiDAR

- Fonte: os 72 setores em **metros** já gravados (`lidar.npy`), gerados por
  `sim_lidar.points_to_sectors_m` (min-distância por setor, com filtro de chão). Espaço livre =
  `max_range` = 12.0 m.
- Normalização para a rede: `x = clip(dist_m / 12.0, 0, 1)`. Perto = 0, livre = 1.
- **Mesma normalização** no treino e na inferência (uma só fonte da verdade, em `ai.shared` ou
  numa função utilitária compartilhada por dataset e policy).

### 3. Coleta com tráfego — acréscimo em `src/ai/collect.py`

Único código novo na coleta: **gerar eventos de frenagem**.

- Nova flag `--traffic N` (default 0 = comportamento atual). Quando `N>0`, após spawnar o ego,
  chamar `spawn_random_vehicles(world, actor_list, N, traffic_manager)`.
- O **autopilot do TM freia sozinho** atrás do veículo da frente (respeita
  `global_distance_to_leading_vehicle`) → os `throttle`/`brake` lidos de `vehicle.get_control()`
  são os **rótulos longitudinais do expert**. Nada de novo no writer.
- Mantém `--recovery` (dados de recuperação lateral da direção) ligável junto.
- `--slow` continua útil para dar tempo de reação. Registrar `traffic: N` no `meta.json`.
- Produz **`dataset_v3`** (com tráfego). Datasets antigos (v1/v2, sem tráfego) seguem válidos
  para a parte de direção.

**Risco/observação:** frames de frenagem (`brake>0`) são minoria mesmo com tráfego. Mitigação no
treino (sampler, abaixo). O `--report` deve mostrar a fração de frames com `brake>0` para
confirmar que há sinal suficiente antes de treinar.

### 4. Dataset + treino

- **`DrivingDataset`** (`src/ai/dataset.py`, ao lado de `SteeringDataset`): retorna
  `(img (3,66,200), lidar (72,) normalizado, alvo (3,) = [steer, throttle, brake])`. Carrega a
  linha `row` de `lidar.npy` (mmap por episódio para não reler o arquivo inteiro por frame).
- **Warm-start** (`src/ai/train.py`): carregar `cam_v2.pt`, copiar as chaves `cnn.*` para o braço
  de câmera do `DrivingNet`; braço de LiDAR e cabeça iniciam do zero. Flag `--init-from
  D:/tcc_data/runs/cam_v2.pt`.
- **Loss**: soma ponderada de `SmoothL1` por eixo — `w_steer·L(steer) + w_thr·L(throttle) +
  w_brake·L(brake)`. Defaults `w_steer=1.0, w_thr=0.5, w_brake=1.0` (frenagem importa e é rara;
  throttle é mais tolerante). Pesos expostos como flags.
- **Sampler**: estender a lógica de `sample_weights` para **combinar** dois desbalanceamentos —
  curvas (`|steer|` alto) **e** frenagem (`brake` alto). Frame com qualquer um dos dois é
  sobre-amostrado. Mantém split **por episódio**.
- Checkpoint salvo pelo menor **val loss combinado**; registrar MAE por eixo + var_ratio de steer.
  Salvar como `driving_v1.pt` em `D:/tcc_data/runs/`.

### 5. Política de inferência — `DrivingPolicy` (novo em `src/ai/model_policy.py`)

- Consome `image` **e** `lidar` do `obs`; converte a nuvem CARLA em 72 setores com
  `sim_lidar.points_to_sectors_m` (**mesma** função da coleta) → normaliza → modelo →
  `(steer, throttle, brake)`.
- `ModelSteeringPolicy` (Fase 2, throttle fixo) **permanece** para comparação.
- **Piso de throttle (mitigação):** se o modelo pedir throttle muito baixo com `brake≈0` em pista
  livre, o carro pode arrastar. Se isso aparecer na validação, aplicar um piso simples
  (`throttle = max(pred, cruise_min)` quando `brake < ε`). Fica como parâmetro, desligado por
  padrão; só liga se o sintoma aparecer.

### 6. Avaliação — `src/ai/eval_openloop.py` e `src/ai/eval_closedloop.py`

- **Open-loop**: MAE por eixo (steer/throttle/brake) + var_ratio de steer sobre o split de val.
- **Closed-loop com tráfego**: cenário parametrizado por seed — colocar um veículo lento/parado à
  frente do ego na mesma faixa e medir. **Aprovado quando o ego para atrás sem colidir em ≥8/10
  seeds**, mantendo lane-keeping em pista livre. Reusar o sensor de colisão já existente
  (`_attach_collision_sensor`).
- **Ablation** (exigência do plano): rodar **o mesmo modelo** com o vetor de LiDAR **"livre"
  (ones = pista limpa)** nos mesmos seeds → deve colidir/parar pior. Prova que o LiDAR está sendo
  usado (e não a câmera sozinha). Registrar a tabela com/sem LiDAR. (Na convenção `near=0/free=1`,
  o sinal neutro de "sem obstáculo" é `ones`; um vetor zerado significaria "obstáculo em tudo".)

### 7. Testes (pytest, sem servidor CARLA)

- `DrivingNet`: shapes de entrada/saída; forward com LiDAR extremo (vetor de zeros) não quebra.
- Padding circular do braço de LiDAR: setor 0 e 71 tratados como vizinhos (teste de invariância a
  deslocamento simples).
- `DrivingDataset`: alinhamento img↔lidar↔label (frame `k` puxa a linha `k` do `.npy`); shapes e
  dtype; normalização em [0,1].
- Warm-start: após copiar, as chaves `cnn.*` do `DrivingNet` são iguais às do `cam_v2.pt`
  (usar um checkpoint sintético no teste, sem depender de `D:`).
- `DrivingPolicy`: dado um `obs` com imagem+nuvem sintéticas, retorna 3 floats nos ranges certos.
- Sampler combinado: frames de curva e de frenagem recebem peso maior.

## Fluxo de dados (ponta a ponta)

1. **Coleta** (`collect.py --traffic N --recovery`): a cada tick síncrono grava
   `frame.jpg` + linha de `lidar.npy` (72 setores, m) + `labels.csv` (`steer,throttle,brake,v,…`)
   do **mesmo tick**. Autopilot do TM = expert (lateral e longitudinal).
2. **Treino** (`train.py --init-from cam_v2.pt`): índice → `DrivingDataset` → warm-start →
   loss ponderado + sampler combinado → `driving_v1.pt`.
3. **Avaliação**: `eval_openloop.py` (MAE por eixo) → `eval_closedloop.py` (cenários com veículo à
   frente + ablation sem LiDAR).

## Arquivos a criar/modificar

**Modificar:**
- `src/ai/model.py` — adicionar `DrivingNet` (mantém `CameraSteeringNet`).
- `src/ai/dataset.py` — adicionar `DrivingDataset` (mantém `SteeringDataset`).
- `src/ai/dataset_index.py` — estender `sample_weights` para combinar curva+frenagem (ou nova
  função `sample_weights_dual`).
- `src/ai/collect.py` — flag `--traffic N` + `spawn_random_vehicles`; registrar no `meta.json`.
- `src/ai/train.py` — warm-start (`--init-from`), loss multi-eixo, sampler combinado, checkpoint
  dual.
- `src/ai/model_policy.py` — adicionar `DrivingPolicy` (mantém `ModelSteeringPolicy`).
- `src/ai/eval_openloop.py` — métricas por eixo.
- `src/ai/eval_closedloop.py` — cenário com veículo à frente + suporte a ablation (LiDAR livre/ones).
- `docs/ESTADO_IA.md` — atualizar estado ao fim da fase.

**Criar:**
- `tests/` — testes das novas peças (acima).
- (opcional) uma função utilitária compartilhada de normalização de LiDAR se ainda não houver
  ponto único.

## Critério de aprovação da Fase 3

1. Open-loop: MAE de steer mantém patamar da Fase 2; MAE de throttle/brake razoável; var_ratio de
   steer ≥ 0.6.
2. Closed-loop **com tráfego**: para atrás de veículo em **≥8/10 seeds novos sem colisão**,
   mantendo lane-keeping em pista livre.
3. **Ablation registrada**: o mesmo modelo **sem LiDAR** falha mais nos mesmos seeds.

Só então a Fase 3 é considerada concluída e seguimos para a Fase 4 (pista custom).
