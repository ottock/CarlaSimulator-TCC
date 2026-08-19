# Estado atual — Handoff (retomar aqui)

> **Leia isto primeiro.** Contexto + o que já foi feito + **próximo passo exato**.
> Referência completa: `docs/ESTADO_IA.md`. Design da fase atual:
> `docs/superpowers/specs/2026-08-19-ai-fase6a-fov180-onnx-design.md`.
> Branch: **`feat/ai-fase4-pista-custom`** · Atualizado: 2026-08-19
> *(Este arquivo era o `ESTADO_FASE4.md`; renomeado para haver um único doc de estado atual.)*

---

## 1. Projeto em uma página

TCC "Digital Twins para treinar carros autônomos": um RC **1:12** que dirige um circuito,
treinado por **Behavioral Cloning** no CARLA (câmera RGB 640×360 + LiDAR 72 setores →
`steer/throttle/brake`) e depois embarcado num **Jetson Nano**. Um *expert* dirige no sim, a gente
grava, a rede imita.

Dados/pesos **fora do OneDrive**: `D:\tcc_data`. Treino PyTorch 2.6+cu124 (RTX 3070).
**Nunca** usar a branch `AItrain`. Estágio A = Towns (Fases 0–3) → Estágio B = pista custom
(Fase 4) → **carro real (Fase 6, AQUI)**.

---

## 2. ONDE ESTAMOS (resumo de 30 s)

- **Fases 0–4: FEITAS.** `driving_track_v1.pt` dirige as **3 pistas do estande** em malha
  fechada: **3/3 limpas** (`offlane=0`, `collisions=0`), e a **ablação do LiDAR separou**
  (3/3 com LiDAR vs **0/3** sem).
- **Fase 5 (refinos): ADIADA** por decisão do Rafael — o modelo já serve para o carro.
- **Fase 6 (Jetson): EM DESIGN.** Nesta sessão analisamos o hardware real
  (`hardware/controle_teste.py`) contra os dados do modelo, resolvemos os bloqueadores e
  **dividimos em 6a e 6b**. O design da 6a está escrito e aprovado para implementação.
- **97 testes verdes**, working tree limpo.

**Próximo passo:** implementar a **Fase 6a** (§4) — não precisa do carro.

---

## 3. O que descobrimos nesta sessão (a análise que mudou o desenho)

Cruzando `hardware/controle_teste.py` (já validado no Jetson) com o `dataset_track_v1`:

**✅ O LiDAR real já fala a nossa língua.** O carro usa um **COIN-D6 (WitMotion), 2D 360°**, e o
`CoinD6Parser` devolve `(angle_deg, dist_m)` — **exatamente** a assinatura de
`shared/lidar_pipeline.scan_to_sectors_m`. Isso não é sorte: esse módulo foi escrito na Fase 0 já
citando o COIN-D6 e mantido em Python 3.6 para rodar no Jetson.

**✅ O gap 3D×2D não é fatal.** O sim usa ray-cast de 32 canais, mas `points_to_sectors_m`
**achata a nuvem** (mínimo por setor). O modelo nunca viu 3D — viu **72 números**. Para parede
vertical a distância horizontal independe da elevação do feixe, então o 2D produz o mesmo vetor.

**✅ Escala 12× resolvida sem gambiarra.** Como a normalização divide por `max_range`, usar
**`max_range = 12/12 = 1.0 m` no carro** é *matematicamente idêntico* a multiplicar as leituras
por 12 — sem multiplicador mágico e sem tocar no `shared/`.

**✅ ONNX validado (o Jetson consegue).** Export em **opset 11** OK (1.9 MB), `onnx.checker` OK,
**paridade PyTorch↔ONNX 5.96e-08**. O padding circular **não** virou `Pad(wrap)` exótico — o
exportador decompõe em `Slice`+`Concat`. Todas as ops são nativas do TensorRT; o TRT 8.x do
JetPack 4.6 cobre até opset 13–14.

**🔴 O achado que obriga a retreinar: oclusão pela própria carroceria.** Para enxergar a parede
(baixa), o sensor precisa ficar baixo — e nessa altura **o carro bloqueia a traseira**. Mascarar
a traseira **só na inferência** muda **25% dos valores** do vetor (erro médio **0.177** na escala
0–1), que é **exatamente a condição da ablação** (0/3 pistas). Logo: **retreinar com FOV
limitado**.

### Números medidos (embasam tudo acima)

```
dataset_track_v1 (14.400 frames)
  ocupação   frente(180°) 45.1%   |  traseira(180°) 49.6%
  mediana    frente 0.376 m real  |  traseira 0.272 m real
  máscara da traseira: 25% dos valores, erro médio 0.177

cone frontal ±10° em condução NORMAL: pior 0.1% = 0.287 m real
  → trava a 0.25 m real = 0 falsos positivos em 14.400 frames
```

---

## 4. Fase 6a — a próxima (design pronto, não precisa do carro)

**Spec completo:** `docs/superpowers/specs/2026-08-19-ai-fase6a-fov180-onnx-design.md`

**Objetivo:** entregar um modelo dual **180° validado em malha fechada** + o **ONNX pronto**.

| Decisão | Por quê |
|---|---|
| Máscara de FOV parametrizada (`fov_deg`, default **180°**) em `shared/lidar_pipeline.py` | Mesma função no sim e no carro; se a bancada medir 200°, é só retreinar |
| **Manter 72 setores**, mascarar como "livre" — não encolher para 36 | O Conv1D usa **padding circular** (setor 71 é vizinho do 0). Num vetor de 36 as pontas (−90°/+90°) deixariam de ser vizinhas e o padding costuraria extrema-esquerda na extrema-direita |
| **`fov_deg` viaja no checkpoint** | Descasamento silencioso treino×inferência = a condição da ablação. Não pode depender de lembrar uma flag |
| Retreinar do `dataset_track_v1` — **sem recoletar** | O `lidar.npy` guarda os 72 setores em metros; a máscara é transformação de dados (~20 min GPU) |
| Longitudinal **fora de escopo** | Dataset tem **0% de frenagem** → head de brake inerte. Velocidade fixa na 6b |

**Arquivos:** `shared/lidar_pipeline.py` (+`apply_fov_mask`), `dataset.py`, `train.py`,
`model_policy.py`, **`export_onnx.py` (novo)**, `requirements.txt` (+onnx, onnxruntime), testes.

**Critério de aprovação:** `eval_track` nas 3 pistas segue **3/3 limpas** com o modelo 180°,
e a paridade do ONNX fica **< 1e-4** no split de validação.

---

## 5. Fase 6b — carro (depois)

**Medições de bancada ANTES do runtime:**
1. **Arco real de oclusão** — carro em área aberta, logar o scan cru e ver quais ângulos devolvem
   distância curta constante (é a carroceria). Esse número vira o `fov_deg` do retreino.
2. **Zero e sentido do ângulo** do COIN-D6 — se estiver espelhado, o mundo chega invertido.
3. **Velocidade real a 1600 µs** — o modelo aprendeu a 2 m/s no sim = **0.167 m/s real**. Se o ESC
   no mínimo andar bem mais rápido, o carro corta as curvas.
4. **Altura da parede e do cone × plano do LiDAR** — se o feixe passar por cima, não há referência.

**Runtime:** câmera + LiDAR → `shared/*` (as mesmas funções) → engine TensorRT → PCA9685.
Do `hardware/controle_teste.py` (já validado): servo **ch15** `1500 + steer*200` µs
(1300 esq / 1500 centro / 1700 dir), ESC **ch12** em PWM fixo (neutro 1500, **mínimo para andar
1600**, máx 1700), mais `watchdog` 0.25 s e `ESC_ARMADO`. Medir **FPS real** (o sim roda a 20 Hz).

**Montagem:** LiDAR no **para-choque dianteiro** — no centro a carroceria bloqueia nos dois
sentidos do eixo.

---

## 6. Gotchas
- **CARLA fora do OneDrive.** O motor já sumiu uma vez (`CreateProcess() returned 2`); a pasta
  `CARLA_0.9.16` **não é git**, então trocar de branch não muda nada nela.
- **`CarlaUE4.exe` é um stub** que gera `CarlaUE4-Win64-Shipping.exe`. Town10HD_Opt é pesado; se
  der timeout de boot, subir manual e rodar com `--no-launch`.
- **Rodar sem `| grep`** — bufferiza e mascara o exit code de scripts em background.
- **`python -u`** para ver progresso ao vivo.

---

## 7. Git
- Branch **`feat/ai-fase4-pista-custom`** (contém a `main` + Fases 3/4 + os merges do colega).
- Últimos commits: `1e19e21` (adia Fase 5, planeja Fase 6), `7b19531`, `c70dc07` (Fase 4
  concluída), `2e6d0d2` (fix da recuperação por episódio).
- **97 testes verdes.**
