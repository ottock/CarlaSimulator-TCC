# Estado atual — Handoff (retomar aqui)

> **Leia isto primeiro.** Contexto + o que já foi feito + **próximo passo exato**.
> Referência completa: `docs/ESTADO_IA.md`. Spec da fase que acabou de fechar:
> `docs/superpowers/specs/2026-08-19-ai-fase6a-fov180-onnx-design.md`.
> Branch: **`feat/ai-fase4-pista-custom`** · Atualizado: 2026-08-23

---

## 1. Projeto em uma página

TCC "Digital Twins para treinar carros autônomos": um RC **1:12** que dirige um circuito,
treinado por **Behavioral Cloning** no CARLA (câmera RGB 640×360 + LiDAR 72 setores →
`steer/throttle/brake`) e depois embarcado num **Jetson Nano**. Um *expert* dirige no sim, a
gente grava, a rede imita.

Dados/pesos **fora do OneDrive**: `D:\tcc_data`. Treino PyTorch 2.6+cu124 (RTX 3070).
**Nunca** usar a branch `AItrain`. Estágio A = Towns (Fases 0–3) → Estágio B = pista custom
(Fase 4) → **carro real (Fase 6)**.

---

## 2. ONDE ESTAMOS (resumo de 30 s)

- **Fases 0–4: FEITAS.** `driving_track_v1.pt` dirige as 3 pistas do estande em malha
  fechada (3/3 limpas) e a ablação do LiDAR separou (3/3 com vs **0/3** sem).
- **Fase 5 (refinos): ADIADA** por decisão do Rafael.
- **Fase 6a: FEITA** (2026-08-23). Modelo **180°** validado + **ONNX** pronto. Detalhes no §3.
- **Fase 6b (carro): É A PRÓXIMA.** Precisa do hardware. §4.
- **125 testes verdes.**

**Próximo passo:** as **medições de bancada** do §4 — antes de escrever qualquer runtime.

---

## 3. O que a Fase 6a entregou

**Problema que ela resolveu:** no carro real a **própria carroceria oclui a traseira** do
LiDAR (o sensor tem de ficar baixo para ver a parede, que é baixa). Mascarar a traseira só
na inferência mudaria 25% do vetor (erro médio 0.177) — a condição exata da ablação. Então
retreinamos com o campo de visão limitado.

**Artefatos:**

| Arquivo | O que é |
|---|---|
| `D:/tcc_data/runs/driving_track_180.pt` | Modelo dual treinado com `--fov-deg 180` |
| `D:/tcc_data/runs/driving_track_180.onnx` | ONNX opset 11, 1.9 MB, pronto para o `trtexec` |

**Resultados medidos:**

```
malha fechada (120 s/pista)   pista1 dev 0.40  pista2 dev 0.38  pista3 dev 0.63
                              offlane=0  collisions=0  ->  3/3 LIMPAS  (criterio atendido)
open-loop                     MAE_steer 0.043  (v1 360 graus: 0.039, +10%)
ablacao do LiDAR no 180       1/3 limpas  (ver a ressalva na 5.7.2 do ESTADO_IA)
ONNX                          checker OK   paridade 5.13e-07 no split de validacao
ops geradas                   Conv Elu Gemm Flatten Unsqueeze Slice Concat Constant Tanh Sigmoid
```

**Como o FOV viaja (não depende de ninguém lembrar de uma flag):**
`train.py --fov-deg` grava `fov_deg` no checkpoint → `DrivingPolicy` **lê do checkpoint** →
`export_onnx.py` copia para os `metadata_props` do `.onnx`. O `eval_track` loga
`LiDAR FOV 180` em cada corrida. Checkpoints antigos não têm a chave e seguem rodando 360°.

**A função compartilhada** é `apply_fov_mask(sectors_m, fov_deg, max_range)` em
`src/ai/shared/lidar_pipeline.py` — pura, Python 3.6-safe, **a mesma no sim e no carro**.
O `lidar.npy` guarda sempre os 360°: **mudar o FOV é retreinar, não recoletar.**

---

## 4. Fase 6b — o carro (A PRÓXIMA)

### 4.1 Medições de bancada — ANTES de escrever runtime

1. **Arco real de oclusão.** Carro em área aberta, logar o scan cru e ver quais ângulos
   devolvem distância curta constante (é a carroceria). **Se não for ~180°, é só retreinar**
   com o `--fov-deg` medido — a infra já está pronta e custa ~13 min de GPU.
2. **Zero e sentido do ângulo do COIN-D6.** Objeto em ângulo conhecido. Se estiver
   espelhado, o mundo chega invertido na rede.
3. **Velocidade real a 1600 µs.** ⚠️ **O alvo é 0.144 m/s** — ver o alerta em 4.3.
4. **Altura da parede e do cone × plano do LiDAR.** Se o feixe passar por cima, não há
   referência nenhuma.

### 4.2 Runtime

câmera + LiDAR → `shared/*` (as MESMAS funções: `preprocess`, `scan_to_sectors_m`,
`apply_fov_mask`, `normalize_sectors_m`) → engine TensorRT → PCA9685.

Do `hardware/controle_teste.py` (já validado no Jetson):
- servo **ch15**: `1500 + steer*200` µs (1300 esq / 1500 centro / 1700 dir)
- ESC **ch12**: PWM fixo (neutro 1500, **mínimo para andar 1600**, máx 1700)
- reusar `watchdog` 0.25 s e `ESC_ARMADO`

**Escala 12×:** usar **`max_range = 1.0`** no carro. Como a normalização divide por
`max_range`, isso é *matematicamente idêntico* a multiplicar as leituras por 12 — sem
multiplicador mágico e sem tocar no `shared/`.

**O `fov_deg` para mascarar o scan real** sai dos `metadata_props` do `.onnx` — não
hardcodar.

Medir **FPS real** (o sim roda a 20 Hz; se o Jetson entregar menos, o carro reage tarde).

### 4.3 ⚠️ O risco aberto mais concreto

**O modelo esterça numa velocidade só.** Nos 14.400 frames do `dataset_track_v1` a
velocidade é quase uma **delta**: p5 **1.69**, p50 **1.73**, p95 **1.80** m/s. Na escala
1:12 → **0.144 m/s** no carro. (Atenção: docs antigos citavam 0.167 m/s, que é o
`target_speed` do expert e **não** o atingido.)

Steering e velocidade não são independentes. Se o ESC no mínimo andar bem mais que
0.144 m/s — o que é bem provável num 1:12 — o carro **corta as curvas**, e isso é falha
**lateral**, que está no escopo do modelo, não longitudinal. Se a bancada confirmar,
o conserto honesto é **recoletar com `target_speed` maior e retreinar**, não ajustar ganho
no runtime.

### 4.4 Fora de escopo (decidido, não esquecido)

**Controle longitudinal aprendido.** Verificado nos 14.400 frames: **`brake` é 0.000 em
todos**. O Pure Pursuit nunca freou, então a cabeça de brake aprendeu a constante zero — é
inerte, não fraca. Na 6b a velocidade é **fixa**, com parada de emergência **em código**:
trava no cone frontal ±10° a **0.25 m real** dá **0 falsos positivos** nos 14.400 frames
(o pior 0.1% da condução normal fica a 0.287 m).

**Montagem:** LiDAR no **para-choque dianteiro** — no centro a carroceria bloqueia nos
dois sentidos do eixo.

---

## 5. Gotchas

- **CARLA fora do OneDrive.** O motor já sumiu uma vez (`CreateProcess() returned 2`); a
  pasta `CARLA_0.9.16` **não é git**, então trocar de branch não muda nada nela.
- **`CarlaUE4.exe` é um stub** que gera `CarlaUE4-Win64-Shipping.exe`. Town10HD_Opt é
  pesado; se der timeout de boot, subir manual e rodar com `--no-launch`.
- **Rodar sem `| grep`** — bufferiza e mascara o exit code de scripts em background.
- **`python -u`** para ver progresso ao vivo.
- **`requirements.txt` está em UTF-16 LE sem BOM** — editar preservando o encoding.

---

## 6. Git

- Branch **`feat/ai-fase4-pista-custom`**.
- **125 testes verdes** (`pytest -q`).
- Comandos da 6a (retreino, eval, ablação, export): §6 do `docs/ESTADO_IA.md`.
