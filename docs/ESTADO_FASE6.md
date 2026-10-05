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

**Próximo passo (DECISÃO do Rafael, 2026-08-23):** ele quer **primeiro ver o modelo rodando
no carro**, antes de fechar as medições de bancada. O caminho seguro para isso é o
**primeiro teste com as rodas no ar** (§4.0) — que entrega o que ele quer ver E as medições
#1 e #2 de brinde.

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

### 4.0 Primeiro teste no carro — rodas no ar (o que o Rafael pediu)

**Objetivo:** ver o modelo esterçando o servo de verdade, sem risco. O `controle_teste.py`
já tem `ESC_ARMADO = False`, então o carro **não anda**; só o servo (ch15) responde.

**Construído (2026-08-23).** `hardware/jetson_runtime.py` é o executável: adaptadores de
câmera CSI, LiDAR serial, engine TensorRT e PCA9685, mais o `main`. A lógica que dá para
errar vive em `src/ai/car/` e é testada no PC: parser do COIN-D6, montagem da volta,
mapa do servo, config e o `DriveLoop` (que prova, com dublês, que o servo centraliza sem
dado e que o ESC nunca sai de neutro). O `scripts/analyze_car_log.py` transforma o log
nas medições #1 e #2.

Rodar:

    /usr/src/tensorrt/bin/trtexec --onnx=driving_track_180.onnx         --saveEngine=driving_track_180.engine --fp16
    python3 hardware/jetson_runtime.py --engine driving_track_180.engine         --config driving_track_180.json --out runs/car_teste1 --seconds 60
    # de volta no PC:
    python scripts/analyze_car_log.py runs/car_teste1

Ordem:
1. Copiar `driving_track_180.onnx` para o Jetson.
2. `trtexec --onnx=driving_track_180.onnx --fp16` **no próprio Jetson** (engines são
   específicas da máquina/versão).
3. Runtime com **ESC desarmado**, lendo o `fov_deg` dos `metadata_props` do `.onnx` e
   usando **`max_range = 1.0`** (a escala 12×).

**O que observar nesse teste:**
- o servo esterça de forma coerente com o que a câmera vê? (aponte o carro para a parede)
- **o ângulo do LiDAR está espelhado?** Se estiver, o carro esterça para o lado errado —
  invisível no papel, óbvio com as rodas no ar, e uma batida com elas no chão.
- **FPS real** do Jetson. O sim roda a 20 Hz; bem menos que isso e o carro reage tarde.
- logar o scan cru já resolve as medições **#1 (arco de oclusão)** e **#2 (zero/sentido)**.

**Só depois disso** armar o ESC — e aí a medição #3 (velocidade) vira a próxima prioridade,
pelo motivo do §4.3.

### 4.1 Medições de bancada — ANTES de soltar o carro

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

---

## Fase 6c — calibração do esterço (2026-10-03)

**A causa raiz do subesterço no carro.** O `steer` normalizado é `ângulo_da_roda /
ângulo_máximo`, e esse máximo não estava em escala: **70°** no Tesla do CARLA contra
**27° medidos** no WLtoys 124017 V2. O mesmo número significava ângulos quase 3×
diferentes — o modelo pedia 42° e o carro entregava 15°.

O que tem de casar é o **raio escalado**, não o ângulo (o Tesla tem 2,875 m de
entre-eixos contra 0,21 × 12 = 2,52 m do carro, 14% de diferença). Igualando o raio
sai **30,2°** → `src/ai/steer_scale.py`, com as medidas do carro num lugar só.

### A verificação reprovou — e o número que explica

| config | volta | v média | parado | \|steer\|=1 |
|---|---|---|---|---|
| 70°, Ld 4,0 (**controle**) | **112%** | 1,73 | 0,8% | 0,0% |
| 30,2°, Ld 4,0 | 46% | 0,71 | 59,2% | 71,7% |
| 30,2°, Ld 8,0 | 45% | 0,69 | 57,3% | 6,1% |
| 30,2°, Ld 4,0, alvo 1,0 m/s | 45% | 0,47 | 44,0% | 59,9% |

O controle fecha a volta limpo **pelo código novo**, então `apply_max_steer_angle`
não quebrou a física: a causa é o limite em si. Com 30,2° o expert encosta na parede
**externa** no meio da primeira curva e fica preso — trava total, acelerador cheio,
parado por mais da metade do episódio. Confirmado nos frames (lente tomada por parede).

**O número:** com 70° o expert usa até `steer` 0,64 = **44,8° de roda = raio 2,89 m**
no sim = **0,241 m** na escala do carro. O carro real só faz **0,412 m**. O traçado
que o Pure Pursuit segue exige **1,7× mais esterço do que o carro tem** — não é
artefato de normalização, e nenhum ajuste de lookahead ou velocidade muda isso
(testados acima).

### Por que nenhum parâmetro resolve

O Pure Pursuit **segue** um caminho, não planeja um. Ele persegue a linha de centro,
não tem noção de corredor viável, e quando satura simplesmente abre e bate.

### O caminho viável existe

A curva é 1/4 de disco com raio interno **zero**, então cortar o ápice é permitido.
Um arco de **R = 4,95 m** (o raio mínimo do carro ×12) tangente às duas retas passa
com **1,72 m de folga** no sim — 14 cm no carro. É também a linha que o carro real é
obrigado a fazer (e a que o Rafael faz quando dirige na mão), então é o que o modelo
precisa aprender.

**Próximo passo:** gerar o caminho do expert com esse arco no lugar do arco da linha
de centro, re-verificar, e só então recoletar e retreinar.

### O curso do servo que faltava (2026-10-04)

O `controle_teste` comandava **±260 µs** de um batente medido de **±300** — 40 µs
de folga para o servo não forçar. Esses 13% de curso custavam caro:

| span | diâmetro do círculo | raio | reserva de esterço na ponta do oval |
|---|---|---|---|
| 260 µs | 125 cm | 0,625 m | **9%** — o expert raspava a parede e travava |
| **300 µs** | **107 cm** | **0,535 m** | **22%** — dirige |

Com isso o `max_steer_deg` do simulador passa de 21,0 para **24,1°** e o
`STEER_SPAN_US` do carro de 260 para **300**.

**Verificação no CARLA, `oval_tcc`, 120 s:**

| configuração | voltas | v | parado | encosta na parede |
|---|---|---|---|---|
| **alvo 2,0 m/s, Ld 4,0, margem 0** | **3,57** | 1,74 | 0,4% | 6,0% |
| alvo 1,0 m/s, Ld 4,0 | 0,56 | 0,27 | 68,5% | 68,8% |
| alvo 2,0 m/s, Ld 6,0 | 0,56 | 0,27 | 84,9% | 85,6% |
| alvo 2,0, Ld 4,0, margem 0,5 m | 1,10 | 0,53 | 69,1% | — |

Dois resultados contra a intuição. **Margem de parede piora**: mais margem é raio
menor, que é mais esterço, que é menos reserva — o gargalo é autoridade de
direção, não espaço lateral. E **mais devagar piora**: a 1,0 m/s o controlador de
acelerador não vence o arrasto das dianteiras em esterço alto.

Pendência conhecida: o expert ainda **toca a parede externa em 6% dos quadros**
(mínimo −0,24 m). Ele completa as voltas, mas o traçado tem degrau de curvatura
na entrada do arco e o Pure Pursuit sai por fora ali. Uma transição em clotoide
resolveria.

### Batente real: 350 µs (2026-10-04)

O batente de ±300 medido em 12/09 estava errado — havia um **fio solto** no servo,
e a roda "parava de responder" por falha de contato. Remedido com o fio consertado,
e confirmado olhando o servo (ainda ganha ângulo entre 300 e 350, sem zumbido e sem
travar):

| span | círculo | raio | reserva na ponta |
|---|---|---|---|
| 260 µs | 125 cm | 0,625 m | 9% — o expert travava |
| 300 µs | 107 cm | 0,535 m | 22% — dirige, raspa 6% |
| **350 µs** | **94 cm** | **0,470 m** | **31%** |
| 400 µs | 94 cm | — | idêntico: já é batente |

**Expert no `oval_tcc`, 120 s:** 3,57 voltas, 1,74 m/s, parado 0,4%,
**encosta na parede 0,0%**, satura 0,6%. O ápice se resolveu com a autoridade de
direção extra — sem margem de parede e sem clotoide.

Esterço do expert: |steer| médio **0,697**, p90 **0,917**, acima de 0,7 em **61%**
dos quadros. No `dataset_track_v1` (sobre o eixo da pista) era 0,308 de média e
0,826 de máximo — um alvo que o carro nunca conseguiu executar.

### Velocidade do simulador: não é a alavanca que parecia

O Pure Pursuit é geométrico (`delta = atan(2L·sin(α)/Ld)`, sem termo de
velocidade), então o par (imagem, esterço) que o modelo aprende é o **mesmo** a
1,74 ou a 4,0 m/s. Muda só a quantidade de quadros por volta (726 contra 353).

Testado: 4,0 m/s dá 7,35 voltas com 0,0% de contato — funciona, mas não compra
transferência. **Mantido 2,0**, que rende o dobro de quadros por volta e um pouco
mais de folga de parede.

O descasamento de velocidade que importa não é a amplitude do esterço; é quanto o
carro anda **às cegas entre duas decisões**: 6,7 cm no carro (1,0 m/s a 15 Hz)
contra 0,73 cm equivalentes no treino, e nesses 6,7 cm o rumo muda 5,7°. A alavanca
para isso é o **FPS do Jetson**, não a velocidade do simulador.

### Modelo do oval — `driving_oval_v1` (2026-10-04)

**Dataset `dataset_oval_v1`:** 12 episódios, 14.400 quadros, 30% recuperação,
1,74 m/s, zero quadros fora da pista.

Dois consertos na coleta, os dois medidos:

**Recuperação por ruído, não por teleporte.** O teleporte trava o carro nesta
pista: o ep0 ficou 54% parado e os seis seguintes 100% parados — 8.400 quadros de
carro imóvel. `set_transform` mantém a velocidade antiga enquanto troca a pose, e
um Tesla de 4,7 m atravessado numa faixa de 6,36 encrava. O `SteeringNoiseInjector`
(já existente desde a Fase 2) perturba o esterço **aplicado** e grava o comando
**limpo** como rótulo: o carro deriva por física, nunca em pose impossível.

**Espelho horizontal no treino.** O oval tem as quatro curvas para o mesmo lado:
14.364 quadros negativos contra 36 positivos. Dirigir o oval ao contrário foi
tentado e quebrou (o carro trava na partida), então a flag foi removida. O espelho
inverte imagem, LiDAR e sinal do esterço em metade das amostras de treino —
simetria exata, sem simulação. Resultado: 50/50, média +0,001.

**Treino:** 40 épocas, parada antecipada na 34. MAE s/t/b = **0,0316**/0,0057/0,0029,
var_ratio 1,02. Na Fase 4 o MAE de esterço era 0,039–0,043.

**Malha fechada, 120 s:** `mean_dev=0,83 m p95=1,71 max=1,82 offlane=0
collisions=0` → **1/1 limpa**. O desvio é medido contra o eixo e o traçado fica a
até 1,93 m dele por construção, então esse número **não é comparável** com os da
Fase 4.

**Ablação do LiDAR: também 1/1 limpa.** Nesta pista o LiDAR não contribui — a
câmera dirige sozinha. Na Fase 4 a ablação dava 0/3 e era o argumento que defendia
a escolha dual. Aqui não dá. Converge com o que o carro real mostrou: o feixe passa
por cima das paredes de isopor, então no asfalto ele também não contribui.

**ONNX:** opset 11, 1,9 MB, paridade 5,07e-07, `onnx.checker` OK. Em `models/`.

### `driving_oval_v2` — recuperação de verdade (2026-10-04)

Rafael perguntou se o dataset tinha caso de "bateu e não volta". Tinha, e a
verificação disso destravou três defeitos encadeados.

**1. Só ruído de esterço não cobre recuperação.** Dispersão de 0,08 m (0,7 cm na
escala do carro) numa faixa que dá ±16 cm. Aumentar a amplitude não ajuda: em 1,0
o carro trava em 55% dos quadros e a dispersão quase não muda. Numa pista apertada
o ruído só altera um pouco o raio; não há para onde vagar.

**2. O empurrão é necessário — errei onde ancorá-lo, por dois lados opostos.**
Ancorado no eixo, arrancava o carro 1,93 m antes de somar o deslocamento.
Ancorado no traçado, no início do episódio o carro ainda está no eixo e um
"empurrão de 0,6" virou um salto medido de 1,4 m com a carroceria 25 cm dentro da
parede. **Da posição atual** o empurrão é exatamente o que diz ser.

**3. Quadros de batida entravam no dataset.** 656 quadros (4,7%) com a carroceria
sobreposta à parede, 138 deles abaixo de 1 m/s com rótulo `steer = −1,000`. Agora
a coleta liga o sensor de colisão do CARLA e descarta o toque e 1 s depois.

Três defesas, cada uma pegou um problema diferente: filtro de velocidade (carro
parado), `StuckDetector` (um encrave contaminava todos os episódios seguintes),
sensor de colisão (contato com a parede).

**Dataset `dataset_oval_v2`:** 14.820 quadros, 16 episódios **todos** a 1,75 m/s,
cobertura fora da linha p95 0,84 m e máx **13 cm na escala do carro**, zero
quadros venenosos.

**Treino:** parada antecipada na época 39. MAE s/t/b = 0,0426/0,0187/0,0039,
var_ratio 1,03.

**Malha fechada (acelerador constante 0,20, como o PWM fixo do carro):**

| modelo | resultado |
|---|---|
| `driving_oval_v1` (só ruído) | **2029 colisões em 30 m** |
| **`driving_oval_v2`** | **1/1 limpa**, `mean_dev=0,89 p95=1,76 max=1,78`, 3,4 m/s |
| `v2` com LiDAR ablado | 1/1 limpa, `mean_dev=0,99 max=1,92` |

A cobertura de recuperação é a diferença entre bater em 30 m e completar a pista.

Dois achados laterais que valem para a escrita:

**O `eval_track` aprovava carro parado.** Reportou "1/1 limpa" para uma corrida com
`mean_speed=0,0` — um carro parado não bate e não sai da pista. O critério agora
exige que ele tenha andado (`ai/eval_criterio.py`), e a corrida que motivou isso
está travada em teste.

**O modelo dirige a 3,4 m/s, o dobro da velocidade de treino.** O Pure Pursuit é
geométrico, então o par (imagem, esterço) não depende de velocidade — e o fechado
confirma. Isso é a favor do carro real, que roda ~7× mais rápido que o treino.

**Acelerador na avaliação é CONSTANTE**, como o PWM fixo do carro: a cabeça de
throttle da rede é ignorada no asfalto, e avaliá-la mediria um carro que não
existe. Com o dataset novo, que descarta os quadros de arrancada, ela aprendeu só
o cruzeiro (0,125) e nem sairia do lugar.
