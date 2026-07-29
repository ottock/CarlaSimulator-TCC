# Estado da Fase 4 — Handoff (retomar aqui)

> **Leia isto primeiro.** Contexto + o que já foi feito + **próximo passo exato**.
> Companheiro do `docs/ESTADO_IA.md` (Fases 0–3 em detalhe).
> Branch: **`feat/ai-fase4-pista-custom`**.

---

## 1. Projeto em uma página

TCC "Digital Twins pra treinar carros autônomos": RC **1:12** que dirige um circuito, treinado por
**Behavioral Cloning** no CARLA (câmera RGB 640×360 + LiDAR 72 setores → `steer/throttle/brake`),
futuro no Jetson. Expert dirige no sim, gravamos, a rede imita. Dados/pesos **fora do OneDrive**:
`D:\tcc_data`. Treino PyTorch 2.6+cu124 (RTX 3070). **Nunca** usar a branch `AItrain`.

Estágio A = Towns (validou a pipeline, Fases 0–3) → **Estágio B = pista custom 1:12 (Fase 4, AQUI)**.

---

## 2. ONDE ESTAMOS AGORA (resumo de 30s)

- **Fases 0–3: FEITAS (na `main`).** Modelo dual-input `driving_v3.pt` dirige e freia nos Towns.
  Ablação do LiDAR foi **inconclusiva nos Towns** (câmera basta pra carro grande).
- **Fase 4: FEITA e VALIDADA.** Coleta → treino → loop fechado → ablação, tudo rodado:
  - `dataset_track_v1`: **14.400 frames** nas 3 pistas, 30% recuperação, 20.3% reto.
  - `driving_track_v1.pt`: open-loop `MAE s/t/b = 0.039/0.017/0.006`, `var_ratio 1.01`.
  - Malha fechada (120 s/pista): **3/3 pistas limpas**, `offlane=0`, `collisions=0`.
  - **Ablação do LiDAR: 3/3 limpas COM vs 0/3 SEM.** A §5.7 finalmente separou — e o valor
    veio da **parede/corredor** (lane-keeping), não dos obstáculos pequenos previstos.
- **Bug corrigido no caminho:** a recuperação só acontecia no 1º episódio de cada pista
  (relógio do teleporte não reiniciava por episódio) e os outros 9 vinham **rotulados como
  recuperação sem nenhum teleporte**. Virou `ai/recovery_schedule.py` + 5 testes; dataset
  recoletado. Detalhe na §5.2 do `ESTADO_IA.md`.
- **CARLA:** roda normal. Lição: **manter o CARLA fora do OneDrive**; ele pode sumir de novo.

**FASE 4 FECHADA** (decisão do Rafael, 2026-07-28). Validada, commitada e documentada.
**Próximo:** a **Fase 5 — refino final do modelo** (*touch-ups*), lista completa na §7 do
`ESTADO_IA.md`. O modelo atual já está bom; a Fase 5 é para deixá-lo redondo.

**O modelo continua dual (câmera+LiDAR)** — o LiDAR fica, é o sensor do carro real. O
"controle só-câmera" da lista é uma rede **descartável de comparação**, não uma troca do
modelo entregue (ver §4 abaixo).

---

## 3. Fase 4 — o que é

**Objetivo:** treinar **UMA rede nas 3 pistas do estande (pista1, pista2, pista3)** — não só a
pista1 — até dirigi-las e **validar em loop fechado**. Escopo: lane-following + loop fechado +
ablação do LiDAR. (Frenagem por obstáculo fica de fora: o Pure Pursuit não reage a obstáculo.)

**A ponte (integração dos 2 lados):**
- **Do colega** (`core/carlaClient/`): `track_builder.py` (peças `tcc_*`, pistas `pista1/2/3`,
  `gerar_waypoints`), `professor.py` (expert **Pure Pursuit**).
- **Nosso** (Fase 3): `EpisodeWriter`, `train_dual`, `DrivingPolicy`, `points_to_sectors_m`.
- **Como casa:** a coleta da pista grava no **nosso formato** (JPG + `lidar.npy` 72 setores +
  `labels.csv`) usando o Pure Pursuit + nossos sensores → **`train_dual` roda sem mudança**. No loop
  fechado o desvio é medido contra a **centerline própria** (`track_ref`), pois a pista não tem
  OpenDRIVE (`map.get_waypoint()` não funciona lá).

### Decisões travadas
- **Pistas: pista1 + pista2 + pista3 (todas), uma rede combinada.** No CLI: `--pistas pista1,pista2,pista3`
  (o `collect_track`/`eval_track` sobrescrevem o `preset` por pista; o `track.preset` do
  `pistaTCC.json` só afeta o `main.py`/visualização).
- **Recuperação baseada na centerline** na coleta (foi O conserto do covariate shift na Fase 2).
- Escala **real**, pista **com paredes** (chão preto/paredes brancas → LiDAR vê o corredor).

### Código da Fase 4 — FEITO e COMMITADO (92 testes verdes)
| Item | Arquivo | Commit |
|---|---|---|
| Fix gap da pista1 (fechar_gap × fator) + testes de geometria | `core/carlaClient/track_builder.py`, `tests/test_track_geometry.py` | `e1c8a9b` |
| `track_ref` (centerline + desvio, puro/testável) + testes | `src/ai/track_ref.py`, `tests/test_track_ref.py` | `2fcf9f2` |
| `spawn_transform` + `collect_track.py` + `eval_track.py` | `core/carlaClient/world_manager.py`, `src/ai/{collect_track,eval_track}.py` | `1ea8390` |

### Smoke validada no CARLA (pista2, 30s) ✅
28/28 peças colocadas (build custom OK); ego spawnou no waypoint 0; 600 frames gravados no nosso
formato, 0 dropped; **LiDAR vê as paredes (~34/72 setores com retorno)**; `DrivingDataset` carrega
direto; frame limpo (chão preto/paredes brancas, o crop 130/30 preserva a fronteira parede/chão);
recuperação só em movimento (30% noise-active). **A integração funciona ponta-a-ponta.**

---

## 4. Fase 4 fechada — o que fica para a Fase 5

Critério de aprovação da Fase 4 **foi atingido**: 3/3 pistas sem sair da pista e sem colisão
(confirmado em corridas de 120 s **e** de 300 s por pista — 420 s acumulados, dev 0.40–0.60 m),
e a ablação separou (3/3 vs 0/3).

A lista de *touch-ups* da **Fase 5** está na **§7 do `ESTADO_IA.md`**: wiggle do steering,
a pista3 (medidamente a pior: dev 0.58–0.60 m vs 0.40–0.42 m), partidas fora do waypoint 0,
obstáculos `tcc_*`, freio inerte (dataset tem 0% de frenagem) e velocidade (o expert roda a
`target_speed=2.0` m/s e o modelo acompanha — subir exige recoletar). Abaixo, o item que é
**evidência para o texto**, não melhoria de modelo:

**O que falta para a escrita do TCC.** `--ablate-lidar` alimenta a rede com "tudo livre" — uma
entrada que ela nunca viu no treino. Isso prova que **este modelo dual depende do LiDAR**, mas
não que **uma rede só-câmera fracassaria** na pista (ela poderia aprender a se virar só com a
imagem, como aconteceu no Town). Para poder afirmar o mais forte:

```powershell
# treinar SO-camera no mesmo dataset (train.py sem --dual) e rodar o mesmo eval_track
.venv\Scripts\python.exe -u src\ai\train.py --data D:/tcc_data/dataset_track_v1 `
    --out D:/tcc_data/runs/cam_track_v1.pt --init-from D:/tcc_data/runs/cam_v2.pt --epochs 40
```
Se a só-câmera também bater nas paredes → **o LiDAR é necessário na pista** (afirmação forte).
Se ela dirigir limpo → a conclusão honesta é "o dual aprendeu a depender do LiDAR", mais fraca.
**Atenção:** o `eval_track` hoje instancia `DrivingPolicy`; rodar um checkpoint só-câmera exige
suportar `ModelSteeringPolicy` lá (throttle fixo), então é um pequeno ajuste de código, não só
um comando.

---

## 5. CARLA — gotchas (o que nos travou hoje)
- **CARLA fora do OneDrive.** O motor sumiu (OneDrive evacuando/extração) → `CreateProcess()
  returned 2`. Já restaurado. Idealmente mover pra `D:\CARLA_0.9.16` + junction
  (`mklink /J repo\CARLA_0.9.16 D:\CARLA_0.9.16`).
- **A pasta `CARLA_0.9.16` NÃO é git** — trocar de branch não muda nada nela.
- **`CarlaUE4.exe` é um stub** que gera `CarlaUE4-Win64-Shipping.exe`. Town10HD_Opt é pesado (boot
  lento). Se der timeout de boot: subir manual (`Start-Process ...CarlaUE4.exe ... -WorkingDirectory
  CARLA_0.9.16`), esperar 1–2 min, rodar com **`--no-launch`**.
- **Rodar sem `| grep`** (bufferiza/mascara a saída de scripts em background).

---

## 6. Comandos (receita de reprodução — já rodados, resultados na §2)

Da raiz do repo, `.venv\Scripts\python.exe`. Se o CARLA já estiver rodando, acrescente `--no-launch`.
Tempos medidos: coleta ~3 min, treino ~13 min (RTX 3070), cada `eval_track` ~4 min.

```powershell
# 0) sanity
.venv\Scripts\python.exe -m pytest tests/ -q            # 97 passed

# 1) COLETA das 3 pistas (o proximo passo)  — nao e realtime, rapido
.venv\Scripts\python.exe -u src\ai\collect_track.py --out D:/tcc_data/dataset_track_v1 ^
    --pistas pista1,pista2,pista3 --episodes-por-pista 4 --seconds 60 --recovery
#   conferir: --report D:/tcc_data/dataset_track_v1  (histograma de steer, % recovery)

# 2) TREINO (dual, warm-start do driving_v3)
.venv\Scripts\python.exe -u src\ai\train.py --dual --data D:/tcc_data/dataset_track_v1 ^
    --out D:/tcc_data/runs/driving_track_v1.pt --init-from D:/tcc_data/runs/driving_v3.pt --epochs 40

# 3) OPEN-LOOP (MAE por eixo)
.venv\Scripts\python.exe src\ai\eval_openloop.py --ckpt D:/tcc_data/runs/driving_track_v1.pt --data D:/tcc_data/dataset_track_v1 --split val

# 4) LOOP FECHADO por pista (assistir)
.venv\Scripts\python.exe src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt ^
    --pistas pista1,pista2,pista3 --seconds 120 --realtime

# 5) ABLACAO do LiDAR (a §5.7 na pista com paredes): rodar as DUAS e comparar colisoes/desvio
.venv\Scripts\python.exe src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt --pistas pista1,pista2,pista3 --seconds 120
.venv\Scripts\python.exe src\ai\eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt --pistas pista1,pista2,pista3 --seconds 120 --ablate-lidar
```

Depois de validar: `pytest` verde, atualizar `docs/ESTADO_IA.md` (§Fase 4) + memória, e abrir PR.

---

## 7. Git
- Branch **`feat/ai-fase4-pista-custom`** (a partir da `main`, que tem Fases 0–3 + merges do colega:
  PR#2 professor/pista, PR#3 nossa Fase 3, PR#4 as 3 pistas do estande).
- Commits nossos: `e1c8a9b`, `2fcf9f2`, `1ea8390` (ver tabela §3). Working tree limpo (só este doc).
- 92 testes verdes.

---

## 8. Como abrir a nova conversa
Cole algo assim:
> "Retomando o TCC (carro autônomo no CARLA). Lê `docs/ESTADO_FASE4.md` (handoff/estado) e
> `docs/ESTADO_IA.md` (Fases 0–3). Estou na branch `feat/ai-fase4-pista-custom`; a ponte da Fase 4
> está commitada e a smoke passou. **Próximo passo:** rodar a coleta completa das **3 pistas**
> (`collect_track --pistas pista1,pista2,pista3`) → treino `driving_track_v1` (warm-start do
> `driving_v3`) → loop fechado (`eval_track`) + ablação do LiDAR. O CARLA já está restaurado. Bora."
