# Runtime no Jetson (Fase 6b, parte 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rodar o modelo treinado no carro real — câmera + LiDAR → TensorRT → servo — com o ESC travado em neutro, gravando um log que o PC analisa depois.

**Architecture:** Toda a lógica testável vive em `src/ai/car/` (puro, Python 3.6-safe, coberto pelo `pytest` do PC porque `pythonpath = ["src"]`). O `hardware/jetson_runtime.py` fica só com os adaptadores de hardware (GStreamer, serial, TensorRT, PCA9685) e a fiação. O laço recebe suas dependências injetadas, então o envelope de segurança é testado com dublês aqui, sem Jetson.

**Tech Stack:** Python 3.6 (Jetson) / 3.12 (PC), numpy, OpenCV, TensorRT 8.x + pycuda (Jetson), pyserial, Adafruit_PCA9685, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-ai-fase6b-runtime-jetson-design.md`

## Global Constraints

- **Tudo em `src/ai/car/` DEVE ser Python 3.6-safe**: sem f-strings, sem walrus, sem type hints em assinatura, sem `dataclasses`. Use `.format()`. Mesma razão de `src/ai/shared/`: esse código roda no Jetson.
- **Nunca reimplementar `ai.shared.*`**: `preprocess`, `scan_to_sectors_m`, `apply_fov_mask`, `normalize_sectors_m` são importados. Divergência ali vira erro silencioso de inferência.
- **O ESC nunca sai de neutro neste plano.** `ESC_ARMADO = False` fixo no código, não é flag de linha de comando.
- **Servo sempre clampado em [1300, 1700] µs**, mesmo com saída inválida do modelo.
- **Convenção de ângulo do LiDAR**: 0° = frente, valores em graus, distâncias em metros — a assinatura de `scan_to_sectors_m`.
- **`max_range` do carro é derivado, não hardcoded**: `max_range_m do modelo / 12.0` (a escala 1:12).
- Rodar os testes com `.venv/Scripts/python.exe -m pytest -q` a partir da raiz do repo.
- Baseline atual: **125 testes verdes**. Nenhuma task pode deixar a suíte vermelha.

---

### Task 1: Pacote `ai.car` + mapeamento do servo e watchdog

**Files:**
- Create: `src/ai/car/__init__.py`
- Create: `src/ai/car/control_map.py`
- Test: `tests/test_car_control_map.py`

**Interfaces:**
- Consumes: nada.
- Produces: `steer_to_us(steer) -> int`, `FrameWatchdog(timeout_s=0.25)` com `.tick(now) -> bool`, e as constantes `STEER_CENTER_US = 1500`, `STEER_LEFT_US = 1300`, `STEER_RIGHT_US = 1700`, `ESC_NEUTRAL_US = 1500`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_car_control_map.py`:

```python
"""Mapeamento da saida do modelo para o servo do carro (Fase 6b).

O modelo devolve steer em [-1, 1]; o servo fala microssegundos. Este mapa e o
ultimo ponto antes do hardware, entao ele tem de ser seguro mesmo recebendo lixo:
um NaN ou um valor fora de faixa NAO pode virar um comando extremo no servo.
"""
import math

import pytest

from ai.car.control_map import (
    ESC_NEUTRAL_US,
    STEER_CENTER_US,
    STEER_LEFT_US,
    STEER_RIGHT_US,
    FrameWatchdog,
    steer_to_us,
)


def test_zero_steer_is_centre():
    assert steer_to_us(0.0) == 1500


def test_full_right_matches_carla_convention():
    # CARLA: steer +1 = direita. O MASTER do hardware: 1700 us = direita.
    assert steer_to_us(1.0) == STEER_RIGHT_US == 1700


def test_full_left():
    assert steer_to_us(-1.0) == STEER_LEFT_US == 1300


def test_half_right_is_linear():
    assert steer_to_us(0.5) == 1600


def test_out_of_range_is_clamped_not_wrapped():
    # A cabeca de steer usa tanh, entao nao deveria estourar -- mas se estourar,
    # o servo nao pode receber um comando alem do batente mecanico.
    assert steer_to_us(5.0) == STEER_RIGHT_US
    assert steer_to_us(-5.0) == STEER_LEFT_US


def test_nan_is_centre():
    assert steer_to_us(float("nan")) == STEER_CENTER_US


def test_none_is_centre():
    assert steer_to_us(None) == STEER_CENTER_US


def test_esc_neutral_constant():
    assert ESC_NEUTRAL_US == 1500


def test_watchdog_does_not_fire_on_the_first_tick():
    wd = FrameWatchdog(timeout_s=0.25)
    assert wd.tick(100.0) is False


def test_watchdog_quiet_when_frames_are_fast():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.05) is False


def test_watchdog_fires_when_a_frame_stalls():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.5) is True


def test_watchdog_recovers_after_a_stall():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    wd.tick(100.5)
    assert wd.tick(100.55) is False
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_control_map.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/__init__.py` vazio (arquivo em branco).

Criar `src/ai/car/control_map.py`:

```python
"""Model output -> car actuator commands (Fase 6b).

Last stop before the hardware, so it must be safe with garbage input: a NaN or an
out-of-range value must never become an extreme servo command. Python 3.6-safe --
this runs on the Jetson.

Numbers come from ``hardware/controle_teste.py``, already validated on the car.
"""

STEER_CENTER_US = 1500
STEER_LEFT_US = 1300
STEER_RIGHT_US = 1700
STEER_SPAN_US = 200
ESC_NEUTRAL_US = 1500


def steer_to_us(steer):
    """Map ``steer`` in [-1, 1] to servo microseconds, clamped to the mechanical stops.

    CARLA's convention (+1 = right) matches the car's (1700 us = right).
    Anything unusable -- ``None``, NaN, a non-number -- returns centre, which is
    the safe command.
    """
    try:
        s = float(steer)
    except (TypeError, ValueError):
        return STEER_CENTER_US
    if s != s:  # NaN
        return STEER_CENTER_US
    us = int(round(STEER_CENTER_US + s * STEER_SPAN_US))
    return max(STEER_LEFT_US, min(STEER_RIGHT_US, us))


class FrameWatchdog:
    """Flags a stalled loop: a frame that took longer than ``timeout_s``.

    The clock is passed in (``tick(now)``) instead of read internally, so the
    behaviour is testable without sleeping.
    """

    def __init__(self, timeout_s=0.25):
        self.timeout_s = timeout_s
        self._last_t = None

    def tick(self, now):
        """Record this frame's time; return True if the gap exceeded the timeout."""
        prev = self._last_t
        self._last_t = now
        if prev is None:
            return False
        return (now - prev) > self.timeout_s
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_control_map.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai/car/__init__.py src/ai/car/control_map.py tests/test_car_control_map.py
git commit -m "feat(car): mapeamento seguro de steer para o servo + watchdog de frame"
```

---

### Task 2: Parser canônico do COIN-D6

**Files:**
- Create: `src/ai/car/coin_d6.py`
- Test: `tests/test_coin_d6.py`

**Interfaces:**
- Consumes: nada.
- Produces: `CoinD6Parser(max_range_m=12.0)` com `.feed(data_bytes) -> list de (angle_deg, dist_m)` e o atributo `.parse_errors`.

**Contexto para quem implementa:** hoje existem DUAS cópias deste parser no repo (`hardware/coin_d6.py` e `hardware/controle_teste.py`) e elas **já divergiram** — a primeira devolve 3-tuplas com intensidade, a segunda 2-tuplas. Esta é a cópia canônica, baseada na de 2-tuplas (a que casa com `scan_to_sectors_m`). **Diferença deliberada em relação às duas:** este parser devolve só **pontos**, sem agrupar em "scans". O agrupamento por volta é da Task 3, porque o corte a cada 300 pontos das versões antigas não é uma volta completa.

Layout do pacote (do parser validado): `AA 55 | tipo(1) | count(1) | start_angle(2, LE, 0.01°) | end_angle(2) | extra(2) | count × [intensidade(1), dist_lo(1), dist_hi(1)]`, total `10 + count*3` bytes.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_coin_d6.py`:

```python
"""Parser do LiDAR COIN-D6 (WitMotion), a copia canonica.

Devolve (angulo_graus, distancia_m) -- exatamente a assinatura que
ai.shared.lidar_pipeline.scan_to_sectors_m consome. Diferente das copias antigas
em hardware/, este parser NAO agrupa pontos em scans: quem fecha a volta e o
ScanAssembler, porque cortar a cada 300 pontos nao e uma volta completa.
"""
import struct

import pytest

from ai.car.coin_d6 import CoinD6Parser


def _packet(start_deg, end_deg, dists_mm):
    """Monta um pacote COIN-D6 valido com as distancias dadas (em milimetros)."""
    n = len(dists_mm)
    pkt = bytearray(b"\xAA\x55")
    pkt += bytes([0x00, n])
    pkt += struct.pack("<H", int(round(start_deg * 100)))
    pkt += struct.pack("<H", int(round(end_deg * 100)))
    pkt += b"\x00\x00"
    for d in dists_mm:
        pkt += bytes([0x30, d & 0xFF, (d >> 8) & 0xFF])
    return bytes(pkt)


def test_single_packet_yields_angle_and_distance_in_metres():
    p = CoinD6Parser()
    pts = p.feed(_packet(10.0, 20.0, [1000, 2000]))
    assert len(pts) == 2
    assert pts[0][0] == pytest.approx(10.0)
    assert pts[0][1] == pytest.approx(1.0)      # 1000 mm -> 1.0 m
    assert pts[1][0] == pytest.approx(20.0)
    assert pts[1][1] == pytest.approx(2.0)


def test_zero_and_one_millimetre_mean_no_return():
    # o sensor usa 0/1 mm para "sem leitura"; deixar passar viraria um obstaculo
    # colado no carro, que e a leitura mais perigosa possivel
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [0, 1, 1500]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(1.5)


def test_distance_beyond_max_range_is_dropped():
    p = CoinD6Parser(max_range_m=12.0)
    pts = p.feed(_packet(0.0, 10.0, [20000, 3000]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(3.0)


def test_packet_split_across_two_reads_is_buffered():
    # a serial entrega pedaços arbitrarios; um pacote partido nao pode virar lixo
    p = CoinD6Parser()
    pkt = _packet(30.0, 40.0, [1000, 1100])
    assert p.feed(pkt[:7]) == []
    pts = p.feed(pkt[7:])
    assert len(pts) == 2


def test_garbage_before_the_header_is_skipped():
    p = CoinD6Parser()
    pts = p.feed(b"\x01\x02\x03" + _packet(50.0, 60.0, [1200]))
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(50.0)


def test_two_packets_in_one_read():
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [1000]) + _packet(20.0, 30.0, [2000]))
    assert [round(a) for a, _ in pts] == [0, 20]


def test_angles_wrap_into_zero_360():
    p = CoinD6Parser()
    pts = p.feed(_packet(350.0, 359.0, [1000, 1000]))
    for ang, _ in pts:
        assert 0.0 <= ang < 360.0


def test_absurd_sample_count_is_rejected_without_hanging():
    p = CoinD6Parser()
    bad = bytearray(b"\xAA\x55\x00\xFF")        # count 255 > limite
    bad += b"\x00" * 6
    pts = p.feed(bytes(bad) + _packet(10.0, 20.0, [1000]))
    assert len(pts) == 1                        # recupera e acha o pacote bom
    assert p.parse_errors >= 1
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coin_d6.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car.coin_d6'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/coin_d6.py`:

```python
"""COIN-D6 (WitMotion) 2D LiDAR packet parser -- the canonical copy.

Emits ``(angle_deg, dist_m)`` tuples, which is exactly what
``ai.shared.lidar_pipeline.scan_to_sectors_m`` consumes. Python 3.6-safe.

Two divergent copies of this parser already exist under ``hardware/`` (one of them
returns 3-tuples with intensity). This is the single source; the hardware scripts
should migrate to it rather than a third copy being made.

Deliberate difference from both: this parser does NOT group points into scans. The
old copies closed a "scan" every 300 points, which is not a full revolution -- a
sector never swept would read ``max_range`` = "free", a hole in the wall. Grouping
by revolution belongs to :mod:`ai.car.scan_assembly`.
"""
import struct

MAX_RANGE_M = 12.0          # alcance fisico do sensor
MAX_SAMPLES_PER_PACKET = 50
HEADER = b"\xAA\x55"


class CoinD6Parser:
    """Feed it raw serial bytes, get back polar points."""

    def __init__(self, max_range_m=MAX_RANGE_M):
        self.max_range_m = max_range_m
        self.buffer = bytearray()
        self.parse_errors = 0

    def feed(self, data):
        """Consume bytes and return the ``(angle_deg, dist_m)`` points completed."""
        self.buffer.extend(data)
        points = []
        while True:
            idx = self.buffer.find(HEADER)
            if idx < 0 or idx + 4 > len(self.buffer):
                if idx < 0:
                    # Guarda o ultimo byte: o header pode estar partido entre reads.
                    self.buffer = self.buffer[-1:]
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            sample_count = self.buffer[3]
            if sample_count == 0 or sample_count > MAX_SAMPLES_PER_PACKET:
                self.buffer = self.buffer[2:]   # descarta este header e recomeca
                self.parse_errors += 1
                continue

            pkt_len = 10 + sample_count * 3
            if len(self.buffer) < pkt_len:
                break                            # pacote incompleto: espera mais
            pkt = bytes(self.buffer[:pkt_len])
            self.buffer = self.buffer[pkt_len:]
            points.extend(self._points_from_packet(pkt, sample_count))
        return points

    def _points_from_packet(self, pkt, sample_count):
        start_angle = (struct.unpack_from("<H", pkt, 4)[0] * 0.01) % 360.0
        end_angle = (struct.unpack_from("<H", pkt, 6)[0] * 0.01) % 360.0

        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 360.0
        if angle_diff > 180:                     # pacote cruzando o zero
            angle_diff -= 360.0
        step = angle_diff / max(sample_count - 1, 1)

        out = []
        for i in range(sample_count):
            offset = 10 + i * 3
            if offset + 2 >= len(pkt):
                break
            dist_mm = pkt[offset + 1] | (pkt[offset + 2] << 8)
            if dist_mm <= 1:                     # 0/1 mm = sem leitura
                continue
            dist_m = dist_mm / 1000.0
            if dist_m > self.max_range_m:
                continue
            out.append(((start_angle + i * step) % 360.0, dist_m))
        return out
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coin_d6.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai/car/coin_d6.py tests/test_coin_d6.py
git commit -m "feat(car): parser canonico do COIN-D6, emitindo pontos em vez de scans

Havia duas copias divergentes em hardware/ (uma com 3-tuplas, outra com 2).
Esta e a unica, com 2-tuplas -- a assinatura de scan_to_sectors_m. E NAO agrupa
em scans: cortar a cada 300 pontos nao e uma volta, e um setor nao varrido leria
'livre', que e um buraco na parede."
```

---

### Task 3: Montagem do scan por volta completa

**Files:**
- Create: `src/ai/car/scan_assembly.py`
- Test: `tests/test_scan_assembly.py`

**Interfaces:**
- Consumes: os pontos `(angle_deg, dist_m)` da Task 2.
- Produces: `ScanAssembler(wrap_drop_deg=180.0, min_points=50)` com `.feed(points) -> list de scans`, onde cada scan é uma `list` de `(angle_deg, dist_m)`.

**Por que existe:** este é o conserto nº 5 do spec. Um scan tem de ser uma **volta completa**; senão o vetor de setores fica com buracos que a rede lê como "livre".

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_scan_assembly.py`:

```python
"""Agrupa os pontos do LiDAR em VOLTAS completas (Fase 6b).

O parser entrega pontos continuamente. Se a gente fechasse um "scan" por contagem
de pontos (como as versoes antigas em hardware/ faziam, a cada 300), um setor nunca
varrido entraria no vetor como max_range = "livre" -- um buraco na parede, que e
justamente o sinal que faz o carro bater nela. Aqui a volta fecha no WRAP do
angulo (359 -> 0).
"""
from ai.car.scan_assembly import ScanAssembler


def _sweep(start, stop, step=1.0, dist=2.0):
    """Pontos de `start` a `stop` (exclusivo) de `step` em `step` graus."""
    pts = []
    a = start
    while a < stop:
        pts.append((a, dist))
        a += step
    return pts


def test_no_wrap_emits_nothing():
    asm = ScanAssembler()
    assert asm.feed(_sweep(0.0, 90.0)) == []


def test_first_partial_revolution_is_discarded():
    # Comecamos a ouvir a serial no meio de uma volta: esse pedaco NAO e uma volta
    # completa e emiti-lo entregaria ao modelo um vetor cheio de buracos.
    asm = ScanAssembler(min_points=10)
    scans = asm.feed(_sweep(300.0, 360.0) + _sweep(0.0, 60.0))
    assert scans == []


def test_full_revolution_after_the_first_wrap_is_emitted():
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(300.0, 360.0))          # fragmento inicial, descartado
    scans = asm.feed(_sweep(0.0, 360.0) + [(1.0, 2.0)])   # volta inteira + wrap
    assert len(scans) == 1
    assert len(scans[0]) == 360


def test_two_revolutions_yield_two_scans():
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(350.0, 360.0))          # fragmento inicial
    scans = asm.feed(_sweep(0.0, 360.0) + _sweep(0.0, 360.0) + [(0.5, 2.0)])
    assert len(scans) == 2
    assert all(len(s) == 360 for s in scans)


def test_points_accumulate_across_feed_calls():
    # a serial entrega pedaços; uma volta quase sempre chega em varios reads
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(350.0, 360.0))
    asm.feed(_sweep(0.0, 180.0))
    scans = asm.feed(_sweep(180.0, 360.0) + [(2.0, 2.0)])
    assert len(scans) == 1
    assert len(scans[0]) == 360


def test_revolution_with_too_few_points_is_dropped():
    # LiDAR engasgando: uma "volta" com 3 pontos nao descreve o mundo
    asm = ScanAssembler(min_points=50)
    asm.feed(_sweep(350.0, 360.0))
    scans = asm.feed([(10.0, 2.0), (200.0, 2.0), (350.0, 2.0), (1.0, 2.0)])
    assert scans == []


def test_scan_keeps_the_angles_and_distances_intact():
    asm = ScanAssembler(min_points=2)
    asm.feed([(350.0, 1.0), (5.0, 9.9)])            # wrap inicial, descarta
    # o (355, 7.7) antes do (1.0) e necessario: so uma QUEDA maior que 180 graus
    # conta como wrap, entao 20 -> 1 nao fecharia a volta
    scans = asm.feed([(10.0, 3.3), (20.0, 4.4), (355.0, 7.7), (1.0, 5.5)])
    assert scans[0][0] == (5.0, 9.9)
    assert (10.0, 3.3) in scans[0]
    assert (355.0, 7.7) in scans[0]
    assert (1.0, 5.5) not in scans[0]               # ja pertence a volta seguinte
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scan_assembly.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car.scan_assembly'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/scan_assembly.py`:

```python
"""Group a stream of LiDAR points into full revolutions (Fase 6b).

The parser emits points continuously. Closing a "scan" by point count -- what the
older ``hardware/`` copies did, every 300 points -- is NOT a revolution: a sector
that was never swept would enter the vector as ``max_range`` = "free", a hole in
the wall, which is exactly the signal that makes the car drive into it.

Here a revolution closes on the angle WRAP (359 -> 0). The first fragment is always
discarded: we start listening mid-revolution, so it is incomplete by construction.

Python 3.6-safe -- this runs on the Jetson.
"""


class ScanAssembler:
    """Feed it points, get back one list per completed revolution."""

    def __init__(self, wrap_drop_deg=180.0, min_points=50):
        """
        Args:
            wrap_drop_deg: an angle drop larger than this counts as the wrap.
                Generous on purpose -- a real sweep never steps backwards by 180
                deg, but jitter between packets can step back by a few degrees.
            min_points: revolutions with fewer points than this are dropped (a
                stuttering sensor does not describe the world).
        """
        self.wrap_drop_deg = wrap_drop_deg
        self.min_points = min_points
        self._current = []
        self._last_angle = None
        self._seen_wrap = False

    def feed(self, points):
        """Consume ``(angle_deg, dist_m)`` points; return completed revolutions."""
        scans = []
        for ang, dist in points:
            if self._last_angle is not None and (self._last_angle - ang) > self.wrap_drop_deg:
                # Fechou uma volta. A primeira e o fragmento inicial: descarta.
                if self._seen_wrap and len(self._current) >= self.min_points:
                    scans.append(self._current)
                self._seen_wrap = True
                self._current = []
            self._current.append((ang, dist))
            self._last_angle = ang
        return scans
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scan_assembly.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai/car/scan_assembly.py tests/test_scan_assembly.py
git commit -m "feat(car): fecha o scan do LiDAR por volta completa, nao por contagem

Setor nao varrido entraria no vetor como 'livre' -- um buraco na parede, que e o
sinal que faz o carro bater nela. Fecha no wrap do angulo e descarta o fragmento
inicial, que e incompleto por construcao."
```

---

### Task 4: Recorte central da imagem (compensa a lente de 130°)

**Files:**
- Create: `src/ai/car/image_crop.py`
- Test: `tests/test_car_image_crop.py`

**Interfaces:**
- Consumes: nada.
- Produces: `center_crop(frame_bgr, frac) -> frame_bgr` e `prepare_frame(frame_bgr, crop_frac, out_size=(640, 360)) -> frame_bgr`.

**Contexto:** a câmera do carro é de **130°**, o modelo treinou com **62.2°**. Recortar o centro aproxima o enquadramento. A fração certa depende da projeção da lente (~0.28 se retilínea, ~0.48 se fisheye) e **ainda não foi medida** — por isso é parâmetro, com `1.0` significando "sem recorte".

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_car_image_crop.py`:

```python
"""Recorte central da imagem do carro (Fase 6b).

A camera real e de 130 graus; o modelo treinou com 62.2. Recortar o centro
aproxima o enquadramento. A fracao certa depende da projecao da lente e AINDA NAO
foi medida -- por isso e parametro, e 1.0 significa "sem recorte" (para gravar um
log cru durante a calibracao).
"""
import numpy as np
import pytest

from ai.car.image_crop import center_crop, prepare_frame


def _frame(h, w):
    """Frame com um valor unico por pixel, para dar para rastrear o recorte."""
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_frac_one_is_a_noop():
    f = _frame(360, 640)
    assert np.array_equal(center_crop(f, 1.0), f)


def test_half_crop_halves_both_dimensions():
    out = center_crop(_frame(360, 640), 0.5)
    assert out.shape == (180, 320, 3)


def test_crop_preserves_the_aspect_ratio():
    # mesma fracao nos dois eixos: 16:9 entra, 16:9 sai. Esticar mudaria a
    # geometria da cena e a rede nunca viu o mundo esticado.
    out = center_crop(_frame(720, 1280), 0.48)
    assert out.shape[1] / float(out.shape[0]) == pytest.approx(1280 / 720.0, rel=1e-2)


def test_crop_takes_the_centre_not_a_corner():
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    f[4:6, 4:6, :] = 255                      # marca so o centro
    out = center_crop(f, 0.4)                 # 4x4 central
    assert out.shape == (4, 4, 3)
    assert out.max() == 255


def test_invalid_fraction_is_rejected_loudly():
    # um 0.0 silencioso viraria um frame vazio e a rede receberia lixo
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            center_crop(_frame(10, 10), bad)


def test_prepare_frame_crops_then_resizes_to_the_model_size():
    out = prepare_frame(_frame(720, 1280), crop_frac=0.5, out_size=(640, 360))
    assert out.shape == (360, 640, 3)


def test_prepare_frame_without_crop_still_resizes():
    out = prepare_frame(_frame(720, 1280), crop_frac=1.0, out_size=(640, 360))
    assert out.shape == (360, 640, 3)
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_image_crop.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car.image_crop'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/image_crop.py`:

```python
"""Centre-crop the car's camera frame to approximate the training field of view.

The car's lens is 130 deg; the model trained on 62.2 deg. Cropping the centre
brings the framing closer. The right fraction depends on the lens projection
(~0.28 if rectilinear, ~0.48 if closer to equidistant/fisheye) and has NOT been
measured yet -- hence a parameter, with 1.0 meaning "no crop".

Cropping happens BEFORE the resize to the model size, so as little resolution as
possible is thrown away. Python 3.6-safe.
"""
import cv2


def center_crop(frame_bgr, frac):
    """Keep the central ``frac`` of width AND height (aspect ratio preserved).

    The same fraction on both axes matters: stretching one axis would change the
    scene geometry, and the network never saw a stretched world.
    """
    if frac <= 0.0:
        raise ValueError("crop fraction must be > 0, got {0!r}".format(frac))
    if frac >= 1.0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return frame_bgr[y0:y0 + ch, x0:x0 + cw]


def prepare_frame(frame_bgr, crop_frac, out_size=(640, 360)):
    """Crop the centre then resize to what the model pipeline expects.

    ``out_size`` is ``(width, height)`` -- 640x360 is what ``ai.shared.image_pipeline``
    assumes, and its 130/30 row crops are calibrated against that.
    """
    cropped = center_crop(frame_bgr, crop_frac)
    return cv2.resize(cropped, out_size, interpolation=cv2.INTER_AREA)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_image_crop.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai/car/image_crop.py tests/test_car_image_crop.py
git commit -m "feat(car): recorte central para aproximar a lente de 130 do FOV de treino"
```

---

### Task 5: Sidecar JSON do modelo (escrita no export, leitura no carro)

**Files:**
- Create: `src/ai/car/config.py`
- Modify: `src/ai/export_onnx.py` (função `export_onnx`, e o `print` do `main`)
- Test: `tests/test_car_config.py`
- Test: `tests/test_export_onnx.py` (acrescentar)

**Interfaces:**
- Consumes: o `meta` que `export_onnx` já monta.
- Produces: `load_model_config(path) -> dict` com chaves `arch`, `fov_deg` (float), `n_sectors` (int), `max_range_m` (float); e `car_max_range(model_cfg, scale=12.0) -> float`. `export_onnx()` passa a devolver o `meta` **e** escrever `<mesmo_stem>.json`.

**Por que existe:** a engine TensorRT **não** carrega os `metadata_props` do ONNX — o `fov_deg` se perderia no `trtexec`. O JSON é lido no Jetson com a stdlib, sem exigir o pacote `onnx` lá.

**Cuidado importante:** `max_range_m` no sidecar é o valor de **treino** (12.0). O carro usa **1.0**, que é 12.0 dividido pela escala 1:12. Por isso `car_max_range()` deriva o número em vez de alguém escrever `1.0` na mão.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_car_config.py`:

```python
"""Config do modelo lida no carro (Fase 6b).

A engine TensorRT NAO carrega os metadata_props do ONNX -- o fov_deg se perderia
no trtexec. O sidecar JSON e lido no Jetson com a stdlib, sem exigir o pacote onnx
instalado la.
"""
import io
import json

import pytest

from ai.car.config import car_max_range, load_model_config


def _write(path, data):
    with io.open(str(path), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data))
    return str(path)


def _valid():
    return {"arch": "DrivingNet", "fov_deg": 180.0, "n_sectors": 72,
            "max_range_m": 12.0, "opset": 11}


def test_reads_the_training_fov(tmp_path):
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert cfg["fov_deg"] == 180.0
    assert cfg["n_sectors"] == 72
    assert cfg["max_range_m"] == 12.0


def test_types_are_numbers_not_strings(tmp_path):
    # os metadata_props do ONNX sao strings; o sidecar tem de entregar numeros,
    # senao um "180.0" viraria comparacao de texto la na frente
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert isinstance(cfg["fov_deg"], float)
    assert isinstance(cfg["n_sectors"], int)


def test_missing_file_fails_loudly(tmp_path):
    # um default silencioso aqui = mascara de FOV errada = a condicao da ablacao
    with pytest.raises(IOError):
        load_model_config(str(tmp_path / "nao_existe.json"))


def test_missing_key_fails_loudly(tmp_path):
    incomplete = _valid()
    del incomplete["fov_deg"]
    with pytest.raises(KeyError):
        load_model_config(_write(tmp_path / "m.json", incomplete))


def test_car_max_range_is_the_training_range_over_the_scale(tmp_path):
    # A normalizacao divide por max_range, entao usar 12.0/12 = 1.0 no carro e
    # matematicamente identico a multiplicar as leituras reais por 12.
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert car_max_range(cfg, scale=12.0) == pytest.approx(1.0)


def test_car_max_range_follows_a_different_training_range(tmp_path):
    cfg = _valid()
    cfg["max_range_m"] = 24.0
    loaded = load_model_config(_write(tmp_path / "m.json", cfg))
    assert car_max_range(loaded, scale=12.0) == pytest.approx(2.0)
```

Acrescentar ao final de `tests/test_export_onnx.py`:

```python
def test_export_writes_a_json_sidecar_next_to_the_onnx(tmp_path):
    # a engine TensorRT nao carrega os metadata_props; o Jetson le este JSON
    import json
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=180.0), out, opset=11)
    with open(str(tmp_path / "m.json")) as fh:
        side = json.load(fh)
    assert side["fov_deg"] == 180.0
    assert side["n_sectors"] == 72
    assert side["max_range_m"] == 12.0
    assert side["arch"] == "DrivingNet"


def test_sidecar_holds_numbers_not_strings(tmp_path):
    import json
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=180.0), out, opset=11)
    with open(str(tmp_path / "m.json")) as fh:
        side = json.load(fh)
    assert isinstance(side["fov_deg"], float)
    assert isinstance(side["n_sectors"], int)
```

- [ ] **Step 2: Rodar os testes e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_config.py tests/test_export_onnx.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai.car.config'` e, nos dois novos do export, `FileNotFoundError` no `m.json`.

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/config.py`:

```python
"""Read the model's sidecar config on the car (Fase 6b).

A TensorRT engine does NOT carry the ONNX ``metadata_props`` -- ``fov_deg`` would be
lost in ``trtexec``. ``export_onnx.py`` writes a small JSON next to the ``.onnx``,
which the Jetson reads with the stdlib, no ``onnx`` package required there.

Fails loudly on a missing file or key: a silent default here means the wrong FOV
mask, which is exactly the LiDAR-ablation condition (0/3 tracks).

Python 3.6-safe.
"""
import io
import json
import os

REQUIRED_KEYS = ("arch", "fov_deg", "n_sectors", "max_range_m")


def load_model_config(path):
    """Load the sidecar JSON, with numeric types and every required key present."""
    if not os.path.exists(path):
        raise IOError("model config not found: {0}".format(path))
    with io.open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise KeyError("model config {0} is missing '{1}'".format(path, key))
    cfg = dict(raw)
    cfg["fov_deg"] = float(raw["fov_deg"])
    cfg["n_sectors"] = int(raw["n_sectors"])
    cfg["max_range_m"] = float(raw["max_range_m"])
    return cfg


def car_max_range(model_cfg, scale=12.0):
    """The car's ``max_range``: the training range divided by the 1:12 scale.

    Normalisation divides by ``max_range``, so feeding the pipeline 12.0/12 = 1.0 m
    on the car is *mathematically identical* to multiplying the real readings by 12.
    Derived rather than hardcoded, so a model retrained with a different range
    still works without anyone remembering to edit a constant.
    """
    return float(model_cfg["max_range_m"]) / float(scale)
```

Em `src/ai/export_onnx.py`, adicionar `import json` ao topo (junto dos outros imports da stdlib) e, dentro de `export_onnx`, logo **antes** do `return meta`, inserir:

```python
    # A engine TensorRT nao carrega os metadata_props: o fov_deg se perderia no
    # trtexec. Este sidecar e lido no Jetson com a stdlib (sem o pacote onnx la).
    sidecar_path = os.path.splitext(out_path)[0] + ".json"
    sidecar = {
        "arch": meta["arch"],
        "fov_deg": float(meta["fov_deg"]),
        "n_sectors": int(meta["n_sectors"]),
        "max_range_m": float(meta["max_range_m"]),
        "opset": int(meta["opset"]),
        "source_checkpoint": meta["source_checkpoint"],
    }
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
```

E em `main()`, logo depois da linha `print("onnx.checker: OK")`, acrescentar:

```python
    print("sidecar:   %s" % (os.path.splitext(out_path)[0] + ".json"))
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_config.py tests/test_export_onnx.py -q`
Expected: PASS, 17 passed (6 novos de config + 11 do export)

- [ ] **Step 5: Regerar o sidecar do modelo real**

Run:
```bash
.venv/Scripts/python.exe -u src/ai/export_onnx.py --model D:/tcc_data/runs/driving_track_180.pt --out D:/tcc_data/runs/driving_track_180.onnx --data D:/tcc_data/dataset_track_v1
```
Expected: imprime `sidecar: D:/tcc_data/runs/driving_track_180.json` e a paridade segue `< 1e-4`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/car/config.py src/ai/export_onnx.py tests/test_car_config.py tests/test_export_onnx.py
git commit -m "feat(car): sidecar JSON com o fov_deg, porque a engine TRT nao carrega metadados

O trtexec descarta os metadata_props do ONNX. O sidecar e lido no Jetson com a
stdlib. car_max_range() deriva o 1.0 do carro de 12.0/12 em vez de hardcodar."
```

---

### Task 6: Gravação do log da corrida

**Files:**
- Create: `src/ai/car/run_log.py`
- Test: `tests/test_car_run_log.py`

**Interfaces:**
- Consumes: nada dos módulos anteriores.
- Produces: `RunLogger(out_dir, meta, jpeg_every=10)` com `.log_frame(t, sectors, control, servo_us, dt, frame_bgr=None)`, `.log_scan(t, points)` e `.close()`.

**Atenção ao nome:** o módulo chama-se `run_log.py`, **não** `logging.py` — o `requirements.txt` do projeto tem um aviso explícito sobre não sombrear o `logging` da stdlib.

Formato do diretório (fixado no spec):

```
meta.json          config do modelo, fracao de corte, ESC armado (sempre false aqui)
frames.jsonl       1 linha JSON por frame: t, steer, throttle, brake, servo_us, dt
sectors.npy        (N, 72) float32 -- o vetor EXATO entregue ao modelo
scans.jsonl        1 linha por volta: t + lista de (angulo, distancia) crus
frames/000000.jpg  1 frame a cada N
```

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_car_run_log.py`:

```python
"""Log da corrida do carro (Fase 6b).

O log E o entregavel deste teste: e dele que saem, no PC, o arco de oclusao da
carroceria e o sentido do angulo do LiDAR. Formatos deliberadamente banais (jsonl
e npy) para nao precisar de dependencia nenhuma para ler.
"""
import io
import json
import os

import numpy as np

from ai.car.run_log import RunLogger


def _read_jsonl(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_creates_the_run_directory_and_meta(tmp_path):
    lg = RunLogger(str(tmp_path / "run1"), meta={"fov_deg": 180.0, "esc_armado": False})
    lg.close()
    with io.open(str(tmp_path / "run1" / "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["fov_deg"] == 180.0
    assert meta["esc_armado"] is False


def test_one_jsonl_line_per_frame(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    for i in range(3):
        lg.log_frame(t=float(i), sectors=np.ones(72, dtype=np.float32),
                     control=(0.1, 0.5, 0.0), servo_us=1520, dt=0.05)
    lg.close()
    rows = _read_jsonl(str(tmp_path / "r" / "frames.jsonl"))
    assert len(rows) == 3
    assert rows[0]["steer"] == 0.1
    assert rows[0]["servo_us"] == 1520
    assert rows[0]["dt"] == 0.05


def test_sectors_are_saved_as_a_matrix(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    for i in range(4):
        lg.log_frame(t=float(i), sectors=np.full(72, i, dtype=np.float32),
                     control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05)
    lg.close()
    arr = np.load(str(tmp_path / "r" / "sectors.npy"))
    assert arr.shape == (4, 72)
    assert arr[2][0] == 2.0


def test_raw_scans_are_logged_for_the_bench_measurements(tmp_path):
    # e daqui que saem o arco de oclusao e o zero/sentido do angulo
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.log_scan(t=1.0, points=[(0.0, 1.5), (180.0, 0.2)])
    lg.close()
    rows = _read_jsonl(str(tmp_path / "r" / "scans.jsonl"))
    assert len(rows) == 1
    assert rows[0]["points"][1] == [180.0, 0.2]


def test_saves_one_jpeg_every_n_frames(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={}, jpeg_every=2)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for i in range(5):
        lg.log_frame(t=float(i), sectors=np.ones(72, dtype=np.float32),
                     control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05, frame_bgr=frame)
    lg.close()
    jpegs = sorted(os.listdir(str(tmp_path / "r" / "frames")))
    assert len(jpegs) == 3          # frames 0, 2 e 4


def test_close_is_safe_to_call_twice(tmp_path):
    # o runtime fecha o log no finally; uma excecao pode fazer isso rodar duas vezes
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.log_frame(t=0.0, sectors=np.ones(72, dtype=np.float32),
                 control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05)
    lg.close()
    lg.close()
    assert np.load(str(tmp_path / "r" / "sectors.npy")).shape == (1, 72)


def test_empty_run_still_writes_a_readable_sectors_file(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.close()
    assert np.load(str(tmp_path / "r" / "sectors.npy")).shape == (0, 72)
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_run_log.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car.run_log'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/run_log.py`:

```python
"""Record what the car saw and did, for analysis back on the PC (Fase 6b).

The log IS the deliverable of the wheels-up test: the chassis occlusion arc and
the LiDAR's angle zero/direction come out of it, with numbers. Formats are
deliberately dull -- jsonl and npy read on the PC with no dependencies.

Named ``run_log`` and not ``logging`` on purpose: never shadow the stdlib module
(see the warning in ``requirements.txt``).

Python 3.6-safe.
"""
import io
import json
import os

import cv2
import numpy as np

N_SECTORS = 72


class RunLogger:
    """One directory per run; ``close()`` flushes the sector matrix."""

    def __init__(self, out_dir, meta, jpeg_every=10, n_sectors=N_SECTORS):
        self.out_dir = out_dir
        self.jpeg_every = jpeg_every
        self.n_sectors = n_sectors
        self._sectors = []
        self._frame_i = 0
        self._closed = False

        os.makedirs(os.path.join(out_dir, "frames"))
        with io.open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, indent=2, sort_keys=True))
        self._frames_fh = io.open(os.path.join(out_dir, "frames.jsonl"), "w", encoding="utf-8")
        self._scans_fh = io.open(os.path.join(out_dir, "scans.jsonl"), "w", encoding="utf-8")

    def log_frame(self, t, sectors, control, servo_us, dt, frame_bgr=None):
        """Record one control cycle. ``control`` is ``(steer, throttle, brake)``."""
        steer, throttle, brake = control
        row = {"t": float(t), "steer": float(steer), "throttle": float(throttle),
               "brake": float(brake), "servo_us": int(servo_us), "dt": float(dt)}
        self._frames_fh.write(json.dumps(row) + "\n")
        self._sectors.append(np.asarray(sectors, dtype=np.float32))
        if frame_bgr is not None and self._frame_i % self.jpeg_every == 0:
            cv2.imwrite(os.path.join(self.out_dir, "frames",
                                     "%06d.jpg" % self._frame_i), frame_bgr)
        self._frame_i += 1

    def log_scan(self, t, points):
        """Record one raw LiDAR revolution -- the bench measurements come from these."""
        row = {"t": float(t), "points": [[float(a), float(d)] for a, d in points]}
        self._scans_fh.write(json.dumps(row) + "\n")

    def close(self):
        """Flush and close. Safe to call twice (the runtime closes it in a finally)."""
        if self._closed:
            return
        self._closed = True
        self._frames_fh.close()
        self._scans_fh.close()
        if self._sectors:
            arr = np.stack(self._sectors)
        else:
            arr = np.zeros((0, self.n_sectors), dtype=np.float32)
        np.save(os.path.join(self.out_dir, "sectors.npy"), arr)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_run_log.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai/car/run_log.py tests/test_car_run_log.py
git commit -m "feat(car): grava o log da corrida (jsonl + npy) para analisar no PC"
```

---

### Task 7: O laço de controle, com dependências injetadas

**Files:**
- Create: `src/ai/car/loop.py`
- Test: `tests/test_car_loop.py`

**Interfaces:**
- Consumes: `ai.car.control_map.steer_to_us` / `FrameWatchdog`, `ai.car.scan_assembly.ScanAssembler`, `ai.car.image_crop.prepare_frame`, `ai.shared.image_pipeline.preprocess`, `ai.shared.lidar_pipeline.{scan_to_sectors_m, apply_fov_mask, normalize_sectors_m}`.
- Produces: `DriveLoop(camera, lidar, engine, actuator, logger, fov_deg, max_range, crop_frac, n_sectors=72, clock=None, watchdog=None)` com `.step() -> dict`.

**Os dublês que o laço espera** (mesmos métodos que os adaptadores reais da Task 8 implementam):
- `camera.read() -> frame_bgr` ou `None`
- `lidar.read_points() -> lista de (angle_deg, dist_m)`
- `engine.infer(img_chw, lidar_vec) -> (steer, throttle, brake)`
- `actuator.set_servo_us(us)` e `actuator.set_esc_us(us)`
- `logger.log_frame(...)` e `logger.log_scan(...)`

**Este é o teste do envelope de segurança** — é aqui que se prova, sem Jetson, que o servo centraliza quando não há dado e que o ESC nunca sai de neutro.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_car_loop.py`:

```python
"""O laco de controle do carro (Fase 6b), testado com dubles.

Este e o teste do ENVELOPE DE SEGURANCA: prova, sem Jetson, que o servo centraliza
quando falta dado e que o ESC nunca sai de neutro nesta fase.
"""
import numpy as np
import pytest

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US
from ai.car.loop import DriveLoop


class FakeCamera:
    def __init__(self, frame=None):
        self.frame = frame if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

    def read(self):
        return self.frame


class FakeLidar:
    """Emite pontos como a serial real: em pedaços, com o wrap fechando a volta.

    O ScanAssembler descarta o primeiro fragmento por construcao (comecamos a
    ouvir no meio de uma volta), entao sao precisos DOIS wraps para um scan sair.
    A 1a leitura traz fragmento + volta inteira; a 2a fecha essa volta.
    Distancia 0.5 m: dentro do max_range=1.0 do carro, entao gera retorno de
    verdade em vez de virar "livre" e mascarar o teste.
    """

    def __init__(self):
        frag = [(float(a), 0.5) for a in range(300, 360, 2)]
        rev = [(float(a), 0.5) for a in range(0, 360, 2)]
        self.batches = [frag + rev, list(rev), list(rev)]

    def read_points(self):
        return self.batches.pop(0) if self.batches else []


class FakeEngine:
    def __init__(self, out=(0.5, 0.4, 0.0)):
        self.out = out
        self.calls = 0

    def infer(self, img, lidar):
        self.calls += 1
        return self.out


class FakeActuator:
    def __init__(self):
        self.servo_history = []
        self.esc_history = []

    def set_servo_us(self, us):
        self.servo_history.append(us)

    def set_esc_us(self, us):
        self.esc_history.append(us)


class FakeLogger:
    def __init__(self):
        self.frames = []
        self.scans = []

    def log_frame(self, **kw):
        self.frames.append(kw)

    def log_scan(self, **kw):
        self.scans.append(kw)


def _loop(**over):
    kw = dict(camera=FakeCamera(), lidar=FakeLidar(), engine=FakeEngine(),
              actuator=FakeActuator(), logger=FakeLogger(),
              fov_deg=180.0, max_range=1.0, crop_frac=0.5)
    kw.update(over)
    return DriveLoop(**kw), kw


def test_servo_is_centred_before_the_first_complete_revolution():
    # Sem uma volta completa o vetor de setores teria buracos que a rede leria como
    # "livre". Inferir nesse estado seria dirigir com um mapa falso.
    lidar = FakeLidar()
    lidar.batches = [[(0.0, 2.0), (10.0, 2.0)]]      # meia volta, sem wrap
    eng = FakeEngine()
    loop, kw = _loop(lidar=lidar, engine=eng)
    tele = loop.step()
    assert kw["actuator"].servo_history == [STEER_CENTER_US]
    assert eng.calls == 0
    assert tele["has_scan"] is False


def test_servo_follows_the_model_once_a_revolution_arrives():
    loop, kw = _loop(engine=FakeEngine(out=(0.5, 0.4, 0.0)))
    loop.step()                                   # fragmento + 1a volta acumulada
    tele = loop.step()                            # o 2o wrap fecha e roda o modelo
    assert tele["has_scan"] is True
    assert tele["steer"] == pytest.approx(0.5)
    assert kw["actuator"].servo_history[-1] == 1600


def test_esc_is_always_neutral():
    # a trava desta fase: o carro nao anda, ponto
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert kw["actuator"].esc_history
    assert set(kw["actuator"].esc_history) == {ESC_NEUTRAL_US}


def test_a_stalled_frame_centres_the_servo():
    # 3 passos de proposito: no 2o ja existe scan e o servo SEGUE o modelo, entao
    # o unico motivo de centralizar no 3o e o watchdog. Com 2 passos este teste
    # passaria mesmo sem watchdog nenhum.
    clock = iter([100.0, 100.05, 100.6]).__next__
    loop, kw = _loop(clock=clock, engine=FakeEngine(out=(0.5, 0.4, 0.0)))
    loop.step()
    loop.step()
    assert kw["actuator"].servo_history[-1] == 1600      # seguindo o modelo
    tele = loop.step()                                    # gap 0.55 s > 0.25
    assert tele["stalled"] is True
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US


def test_missing_camera_frame_centres_the_servo():
    class Blind:
        def read(self):
            return None

    loop, kw = _loop(camera=Blind())
    loop.step()
    tele = loop.step()
    assert tele["has_scan"] is True                # o LiDAR esta ok...
    assert set(kw["actuator"].servo_history) == {STEER_CENTER_US}   # ...mas sem imagem


def test_the_lidar_vector_is_masked_and_normalised():
    loop, kw = _loop()
    loop.step()
    tele = loop.step()
    vec = tele["lidar_vec"]
    assert vec.shape == (72,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0
    # frente: retorno real a 0.5 m com max_range 1.0 -> 0.5
    assert vec[0] == pytest.approx(0.5, abs=1e-6)
    # traseira: cega pela mascara de FOV 180 -> "livre"
    assert np.allclose(vec[18:54], 1.0)


def test_every_step_is_logged():
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert len(kw["logger"].frames) == 2


def test_completed_revolutions_are_logged_raw():
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert len(kw["logger"].scans) == 1
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_loop.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'ai.car.loop'`

- [ ] **Step 3: Implementação mínima**

Criar `src/ai/car/loop.py`:

```python
"""The car's control loop, with its hardware injected (Fase 6b).

Every dependency is passed in, so the whole safety envelope is testable on the PC
with fakes: servo centred when data is missing, ESC never leaving neutral. The
Jetson wiring lives in ``hardware/jetson_runtime.py`` and only supplies the real
camera, LiDAR, engine and actuator.

The perception chain reuses ``ai.shared.*`` unchanged -- the same functions the
simulator ran. Reimplementing any of them here would be a silent inference bug.

Python 3.6-safe.
"""
import time

import numpy as np

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US, FrameWatchdog, steer_to_us
from ai.car.image_crop import prepare_frame
from ai.car.scan_assembly import ScanAssembler
from ai.shared.image_pipeline import preprocess
from ai.shared.lidar_pipeline import apply_fov_mask, normalize_sectors_m, scan_to_sectors_m


class DriveLoop:
    """One ``step()`` = one control cycle."""

    def __init__(self, camera, lidar, engine, actuator, logger, fov_deg, max_range,
                 crop_frac, n_sectors=72, clock=None, watchdog=None):
        self.camera = camera
        self.lidar = lidar
        self.engine = engine
        self.actuator = actuator
        self.logger = logger
        self.fov_deg = fov_deg
        self.max_range = max_range
        self.crop_frac = crop_frac
        self.n_sectors = n_sectors
        self.clock = clock or time.monotonic
        self.watchdog = watchdog or FrameWatchdog()
        self.assembler = ScanAssembler()
        self._last_vec = None
        self._t_prev = None

    def _sectors_from_scan(self, scan):
        angles = [a for a, _ in scan]
        dists = [d for _, d in scan]
        sectors_m = scan_to_sectors_m(angles, dists, n_sectors=self.n_sectors,
                                      max_range=self.max_range)
        sectors_m = apply_fov_mask(sectors_m, self.fov_deg, self.max_range)
        return normalize_sectors_m(sectors_m, self.max_range)

    def step(self):
        """Read sensors, run the model, command the servo, log. Returns telemetry."""
        now = self.clock()
        stalled = self.watchdog.tick(now)
        dt = 0.0 if self._t_prev is None else (now - self._t_prev)
        self._t_prev = now

        for scan in self.assembler.feed(self.lidar.read_points()):
            self._last_vec = self._sectors_from_scan(scan)
            self.logger.log_scan(t=now, points=scan)

        frame = self.camera.read()
        control = (0.0, 0.0, 0.0)
        # Sem uma volta completa o vetor teria buracos que a rede leria como "livre";
        # sem imagem nao ha o que inferir; um frame estourado significa laco travado.
        # Nos tres casos o comando seguro e o mesmo: servo ao centro.
        can_drive = (self._last_vec is not None) and (frame is not None) and (not stalled)
        if can_drive:
            img = preprocess(prepare_frame(frame, self.crop_frac))
            control = self.engine.infer(img, self._last_vec)
        servo_us = steer_to_us(control[0]) if can_drive else STEER_CENTER_US

        self.actuator.set_servo_us(servo_us)
        self.actuator.set_esc_us(ESC_NEUTRAL_US)   # nunca sai de neutro nesta fase

        vec = self._last_vec
        self.logger.log_frame(
            t=now,
            sectors=vec if vec is not None else np.ones(self.n_sectors, dtype=np.float32),
            control=control, servo_us=servo_us, dt=dt, frame_bgr=frame)
        return {"t": now, "steer": control[0], "throttle": control[1],
                "brake": control[2], "servo_us": servo_us, "dt": dt,
                "stalled": stalled, "has_scan": self._last_vec is not None,
                "lidar_vec": vec}
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_car_loop.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, **182 passed** (125 do baseline + 57: 12 control_map + 8 coin_d6 + 7 scan_assembly + 7 image_crop + 8 config/export + 7 run_log + 8 loop). Se o número divergir, confira qual arquivo trouxe menos testes que o esperado antes de seguir.

- [ ] **Step 6: Commit**

```bash
git add src/ai/car/loop.py tests/test_car_loop.py
git commit -m "feat(car): laco de controle com dependencias injetadas

O envelope de seguranca fica testado no PC com dubles: servo centraliza sem volta
completa do LiDAR, sem imagem e com frame estourado; ESC nunca sai de neutro."
```

---

### Task 8: Adaptadores de hardware e o executável do Jetson

**Files:**
- Create: `hardware/jetson_runtime.py`

**Interfaces:**
- Consumes: tudo de `ai.car.*` e `ai.shared.*`.
- Produces: nada para outras tasks — é o ponto de entrada.

**Esta task não tem teste automatizado.** Os adaptadores só existem no Jetson (TensorRT, PCA9685, GStreamer, serial). A carga de correção foi deliberadamente empurrada para o laço da Task 7, que é testado. A verificação aqui é manual, no hardware, e está no Step 3.

**Aviso para quem implementa:** o código de TensorRT abaixo segue a API documentada da 8.x (JetPack 4.6) mas **não foi executado num Jetson**. Se a deserialização ou os bindings falharem, é o primeiro lugar para olhar — não o laço.

- [ ] **Step 1: Escrever o runtime**

Criar `hardware/jetson_runtime.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roda o modelo treinado no carro (Fase 6b, primeiro teste com rodas no ar).

Camera CSI + LiDAR COIN-D6 -> engine TensorRT -> servo. O ESC fica TRAVADO EM
NEUTRO: nesta fase o carro nao anda.

Ambiente: Jetson Nano / JetPack 4.6.x / Python 3.6.

Antes de rodar, gere a engine NO PROPRIO JETSON (engines sao especificas da
maquina e da versao):

    /usr/src/tensorrt/bin/trtexec --onnx=driving_track_180.onnx \\
        --saveEngine=driving_track_180.engine --fp16

Uso:
    python3 hardware/jetson_runtime.py \\
        --engine driving_track_180.engine \\
        --config driving_track_180.json \\
        --out runs/car_teste1 --seconds 60
"""
import argparse
import glob
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import cv2
import numpy as np
import serial

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (inicializa o contexto CUDA)
import Adafruit_PCA9685

from ai.car.coin_d6 import CoinD6Parser
from ai.car.config import car_max_range, load_model_config
from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US
from ai.car.loop import DriveLoop
from ai.car.run_log import RunLogger

# ============================ SEGURANCA ============================
# O carro NAO anda nesta fase. Nao e flag de linha de comando de proposito:
# para armar o ESC alguem tem de editar este arquivo e saber o que faz.
ESC_ARMADO = False
# ===================================================================

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15
ESC_CHANNEL = 12

BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP = bytes([0xAA, 0x55, 0xF5, 0x0A])

CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1280, 720, 60
FLIP_METHOD = 2          # o modulo IMX219 vem de cabeca pra baixo (ver camera_teste.py)


class CsiCamera:
    """Camera CSI IMX219 via GStreamer (nao abre como webcam USB comum)."""

    def __init__(self, sensor_id=0, flip_method=FLIP_METHOD):
        pipeline = (
            "nvarguscamerasrc sensor-id={sid} ! "
            "video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, "
            "framerate=(fraction){fps}/1 ! "
            "nvvidconv flip-method={flip} ! "
            "video/x-raw, format=(string)BGRx ! videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        ).format(sid=sensor_id, cw=CAP_WIDTH, ch=CAP_HEIGHT, fps=CAP_FPS, flip=flip_method)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("nao consegui abrir a camera CSI")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class SerialLidar:
    """COIN-D6 na serial, devolvendo pontos (angulo, distancia)."""

    def __init__(self, port=None, max_range_m=12.0):
        port = port or self._find_port()
        if not port:
            raise RuntimeError("nenhuma porta de LiDAR encontrada (/dev/ttyUSB*)")
        self.ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=0)
        self.ser.reset_input_buffer()
        time.sleep(0.1)
        self.ser.write(CMD_START)
        time.sleep(0.3)
        # max_range aqui e o FISICO do sensor (12 m), nao o do modelo: filtrar
        # antes na escala do carro descartaria leituras validas.
        self.parser = CoinD6Parser(max_range_m=max_range_m)

    @staticmethod
    def _find_port():
        portas = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        return portas[0] if portas else None

    def read_points(self):
        n = self.ser.in_waiting
        if not n:
            return []
        return self.parser.feed(self.ser.read(n))

    def close(self):
        try:
            self.ser.write(CMD_STOP)
        except Exception:
            pass
        self.ser.close()


class TrtEngine:
    """Engine TensorRT com dois inputs (image, lidar) e um output (control)."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as fh, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError("falha ao deserializar a engine: {0}".format(engine_path))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.host = {}
        self.device = {}
        self.bindings = []
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            shape = tuple(self.context.get_binding_shape(idx))
            shape = tuple(1 if d < 0 else d for d in shape)   # batch dinamico -> 1
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.host[name] = host
            self.device[name] = dev
            self.bindings.append(int(dev))
        self._out_name = [self.engine.get_binding_name(i)
                          for i in range(self.engine.num_bindings)
                          if not self.engine.binding_is_input(i)][0]

    def infer(self, img_chw, lidar_vec):
        """(3,66,200) float32 + (72,) float32 -> (steer, throttle, brake)."""
        np.copyto(self.host["image"], np.ascontiguousarray(img_chw, dtype=np.float32).ravel())
        np.copyto(self.host["lidar"], np.ascontiguousarray(lidar_vec, dtype=np.float32).ravel())
        cuda.memcpy_htod_async(self.device["image"], self.host["image"], self.stream)
        cuda.memcpy_htod_async(self.device["lidar"], self.host["lidar"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host[self._out_name], self.device[self._out_name],
                               self.stream)
        self.stream.synchronize()
        out = self.host[self._out_name]
        return float(out[0]), float(out[1]), float(out[2])


class Pca9685Actuator:
    """Servo e ESC no PCA9685. O ESC so sai de neutro se ESC_ARMADO for True."""

    def __init__(self):
        self.pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
        self.pwm.set_pwm_freq(PWM_FREQ)
        self.period_us = 1000000.0 / PWM_FREQ

    def _set_us(self, channel, microseconds):
        steps = int(microseconds / (self.period_us / 4096))
        self.pwm.set_pwm(channel, 0, max(0, min(4095, steps)))

    def set_servo_us(self, us):
        self._set_us(SERVO_CHANNEL, us)

    def set_esc_us(self, us):
        if not ESC_ARMADO:
            us = ESC_NEUTRAL_US
        self._set_us(ESC_CHANNEL, us)

    def safe_state(self):
        """Servo ao centro, ESC em neutro. Chamado na saida, aconteca o que acontecer."""
        self.set_servo_us(STEER_CENTER_US)
        self._set_us(ESC_CHANNEL, ESC_NEUTRAL_US)


def main():
    p = argparse.ArgumentParser(description="Runtime do modelo no carro (Fase 6b)")
    p.add_argument("--engine", required=True, help="Arquivo .engine do TensorRT")
    p.add_argument("--config", required=True, help="Sidecar .json do modelo")
    p.add_argument("--out", required=True, help="Diretorio do log desta corrida")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--crop-frac", type=float, default=0.48,
                   help="Fracao central da imagem (lente de 130 vs 62.2 do treino). "
                        "0.48 e um CHUTE a calibrar; 1.0 desliga o recorte.")
    p.add_argument("--scale", type=float, default=12.0, help="Escala do modelo (1:12)")
    p.add_argument("--jpeg-every", type=int, default=10)
    a = p.parse_args()

    cfg = load_model_config(a.config)
    max_range = car_max_range(cfg, scale=a.scale)
    print("modelo: fov={0} deg  n_sectors={1}  max_range no carro={2:.3f} m"
          .format(cfg["fov_deg"], cfg["n_sectors"], max_range))
    print("ESC ARMADO: {0}".format(ESC_ARMADO))
    if not ESC_ARMADO:
        print("O carro NAO anda. Rodas no ar recomendado mesmo assim.")

    camera = CsiCamera()
    lidar = SerialLidar()
    engine = TrtEngine(a.engine)
    actuator = Pca9685Actuator()
    actuator.safe_state()
    logger = RunLogger(a.out, meta={
        "fov_deg": cfg["fov_deg"], "n_sectors": cfg["n_sectors"],
        "max_range_m_car": max_range, "scale": a.scale,
        "crop_frac": a.crop_frac, "esc_armado": ESC_ARMADO,
        "engine": os.path.basename(a.engine),
    }, jpeg_every=a.jpeg_every)

    loop = DriveLoop(camera=camera, lidar=lidar, engine=engine, actuator=actuator,
                     logger=logger, fov_deg=cfg["fov_deg"], max_range=max_range,
                     crop_frac=a.crop_frac, n_sectors=cfg["n_sectors"])

    t_end = time.monotonic() + a.seconds
    n, t_report = 0, time.monotonic()
    try:
        while time.monotonic() < t_end:
            tele = loop.step()
            n += 1
            if time.monotonic() - t_report >= 1.0:
                fps = n / (time.monotonic() - t_report)
                print("fps={0:5.1f}  steer={1:+.3f}  servo={2}us  scan={3}".format(
                    fps, tele["steer"], tele["servo_us"],
                    "ok" if tele["has_scan"] else "AGUARDANDO"))
                n, t_report = 0, time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrompido")
    finally:
        actuator.safe_state()
        logger.close()
        camera.close()
        lidar.close()
        print("estado seguro. log em {0}".format(a.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificar que o arquivo é sintaticamente válido no PC**

O arquivo não importa no PC (falta `serial`, `tensorrt`, `Adafruit_PCA9685`), mas a sintaxe dá para conferir:

Run: `.venv/Scripts/python.exe -m py_compile hardware/jetson_runtime.py`
Expected: sem saída (sucesso)

- [ ] **Step 3: Verificação manual no Jetson**

No Jetson, com o repo atualizado (`git pull`) e o `.onnx` + `.json` copiados:

```bash
# 1) gerar a engine (so precisa uma vez por maquina/versao)
/usr/src/tensorrt/bin/trtexec --onnx=driving_track_180.onnx \
    --saveEngine=driving_track_180.engine --fp16

# 2) RODAS NO AR. Rodar 60 s.
python3 hardware/jetson_runtime.py --engine driving_track_180.engine \
    --config driving_track_180.json --out runs/car_teste1 --seconds 60
```

Confirmar: o `fps` aparece; `scan=ok` depois dos primeiros segundos; o servo **se mexe** ao apontar o carro para uma parede; ao sair, imprime `estado seguro`.

- [ ] **Step 4: Commit**

```bash
git add hardware/jetson_runtime.py
git commit -m "feat(car): runtime do Jetson -- camera + LiDAR -> TensorRT -> servo

ESC travado em neutro (ESC_ARMADO=False fixo no codigo, nao e flag). Os adaptadores
sao finos de proposito: a carga de correcao esta no DriveLoop, que e testado no PC."
```

---

### Task 9: Análise do log no PC

**Files:**
- Create: `scripts/analyze_car_log.py`
- Test: `tests/test_analyze_car_log.py`

**Interfaces:**
- Consumes: o diretório de log da Task 6.
- Produces: `occlusion_arc(sectors, max_range) -> lista de índices`, `sector_to_deg(i, n_sectors) -> float`, `fps_stats(frames) -> dict`, `load_run(run_dir) -> dict`.

**Isto é o que transforma o teste com rodas no ar nas medições de bancada #1 e #2** — com número, não no olho.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_analyze_car_log.py`:

```python
"""Analise do log do carro, no PC (Fase 6b).

E isto que transforma o teste com rodas no ar nas medicoes de bancada #1 (arco de
oclusao da carroceria) e #2 (zero e sentido do angulo) -- com numero.
"""
import numpy as np
import pytest

from ai.car.run_log import RunLogger
from scripts_analyze import fps_stats, load_run, occlusion_arc, sector_to_deg


def test_occlusion_arc_finds_sectors_that_are_always_blocked():
    # A carroceria e o unico obstaculo que nunca se move: distancia curta E
    # praticamente constante. O mundo la fora varia conforme o carro gira.
    rng = np.random.default_rng(0)
    sectors = rng.uniform(0.5, 1.0, (200, 72)).astype(np.float32)
    sectors[:, 20:50] = 0.12                     # carroceria: curta e constante
    arc = occlusion_arc(sectors, max_range=1.0)
    assert set(arc) == set(range(20, 50))


def test_occlusion_arc_ignores_a_close_but_moving_wall():
    # uma parede perto varia conforme o carro se move: nao e a carroceria
    rng = np.random.default_rng(1)
    sectors = rng.uniform(0.5, 1.0, (200, 72)).astype(np.float32)
    sectors[:, 10:15] = rng.uniform(0.10, 0.40, (200, 5))
    assert occlusion_arc(sectors, max_range=1.0) == []


def test_occlusion_arc_empty_when_nothing_is_blocked():
    sectors = np.ones((50, 72), dtype=np.float32)
    assert occlusion_arc(sectors, max_range=1.0) == []


def test_sector_to_degrees_uses_the_sector_centre():
    # setor i cobre [i*5, (i+1)*5); o centro e (i+0.5)*5 -- a mesma convencao do
    # apply_fov_mask, senao o angulo relatado nao casaria com a mascara
    assert sector_to_deg(0, 72) == pytest.approx(2.5)
    assert sector_to_deg(36, 72) == pytest.approx(182.5)


def test_fps_stats_reports_the_median_and_the_worst_frame():
    frames = [{"dt": 0.05}, {"dt": 0.05}, {"dt": 0.05}, {"dt": 0.40}]
    st = fps_stats(frames)
    assert st["fps_median"] == pytest.approx(20.0)
    assert st["dt_max"] == pytest.approx(0.40)
    assert st["n"] == 4


def test_fps_stats_ignores_the_first_frame_with_zero_dt():
    # o primeiro step nao tem frame anterior, entao dt=0 e nao e um FPS infinito
    st = fps_stats([{"dt": 0.0}, {"dt": 0.05}, {"dt": 0.05}])
    assert st["fps_median"] == pytest.approx(20.0)


def test_load_run_reads_back_what_the_logger_wrote(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={"fov_deg": 180.0}, jpeg_every=100)
    lg.log_scan(t=0.0, points=[(0.0, 0.5)])
    for i in range(3):
        lg.log_frame(t=float(i), sectors=np.full(72, 0.5, dtype=np.float32),
                     control=(0.2, 0.0, 0.0), servo_us=1540, dt=0.05)
    lg.close()
    run = load_run(str(tmp_path / "r"))
    assert run["meta"]["fov_deg"] == 180.0
    assert run["sectors"].shape == (3, 72)
    assert len(run["frames"]) == 3
    assert len(run["scans"]) == 1
```

Criar também `tests/scripts_analyze.py` (um atalho de import, porque `scripts/` não está no `pythonpath` do pytest):

```python
"""Ponte para importar scripts/analyze_car_log.py nos testes.

O pytest tem pythonpath = ["src"], entao scripts/ nao e importavel diretamente.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from analyze_car_log import fps_stats, load_run, occlusion_arc, sector_to_deg  # noqa: E402,F401
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_car_log.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'analyze_car_log'`

- [ ] **Step 3: Implementação mínima**

Criar `scripts/analyze_car_log.py`:

```python
"""Le o log de uma corrida do carro e devolve as medicoes de bancada (Fase 6b).

Roda no PC, sobre o diretorio que o ai.car.run_log gravou no Jetson. E isto que
transforma o teste com rodas no ar em numero:

  1. arco de oclusao da carroceria  (medicao de bancada #1)
  2. onde estao os retornos mais proximos (ajuda a medicao #2, zero/sentido)
  3. FPS real do laco
  4. ocupacao dos setores, para comparar com o dataset do simulador

Uso:
    python scripts/analyze_car_log.py runs/car_teste1
"""
import argparse
import io
import json
import os

import numpy as np


def load_run(run_dir):
    """Read a run directory back into memory."""
    with io.open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    sectors = np.load(os.path.join(run_dir, "sectors.npy"))
    frames = _read_jsonl(os.path.join(run_dir, "frames.jsonl"))
    scans = _read_jsonl(os.path.join(run_dir, "scans.jsonl"))
    return {"meta": meta, "sectors": sectors, "frames": frames, "scans": scans}


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sector_to_deg(i, n_sectors=72):
    """Centre angle of sector ``i`` -- the same convention ``apply_fov_mask`` uses."""
    return (i + 0.5) * (360.0 / n_sectors)


def occlusion_arc(sectors, max_range, near_frac=0.5, blocked_frac=0.95, spread_frac=0.05):
    """Sectors blocked by the car's own chassis.

    The chassis is the only obstacle that never moves: it reads SHORT in nearly
    every frame AND barely varies. Everything out in the world changes as the car
    turns, so a close-but-varying wall is correctly not reported.

    Args:
        sectors: ``(N, n_sectors)`` of normalized values, as fed to the model.
        max_range: the run's ``max_range`` (values are in [0, 1] after normalising,
            so this is used for the "near" threshold in the same scale).
        near_frac: a reading below this fraction of full range counts as "something
            there".
        blocked_frac: fraction of frames that must be near for the sector to count.
        spread_frac: max (p95 - p5) spread, as a fraction of full range, for the
            sector to count as "not moving".
    """
    if sectors.size == 0:
        return []
    arr = np.asarray(sectors, dtype=np.float64)
    near = (arr < near_frac).mean(axis=0)
    spread = np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)
    blocked = (near >= blocked_frac) & (spread <= spread_frac)
    return [int(i) for i in np.nonzero(blocked)[0]]


def fps_stats(frames):
    """Loop timing. The sim ran at 20 Hz; well under that and the car reacts late."""
    dts = [f["dt"] for f in frames if f.get("dt", 0.0) > 0.0]
    if not dts:
        return {"n": len(frames), "fps_median": 0.0, "fps_min": 0.0, "dt_max": 0.0}
    dts = np.asarray(dts, dtype=np.float64)
    return {"n": len(frames), "fps_median": float(1.0 / np.median(dts)),
            "fps_min": float(1.0 / dts.max()), "dt_max": float(dts.max())}


def main():
    p = argparse.ArgumentParser(description="Analisa o log de uma corrida do carro")
    p.add_argument("run_dir")
    a = p.parse_args()

    run = load_run(a.run_dir)
    sectors, meta = run["sectors"], run["meta"]
    n_sectors = int(meta.get("n_sectors", 72))
    max_range = float(meta.get("max_range_m_car", 1.0))
    print("corrida: %s  (%d frames, %d voltas de LiDAR)"
          % (a.run_dir, len(run["frames"]), len(run["scans"])))
    print("meta: fov=%s deg  max_range no carro=%.3f m  crop_frac=%s"
          % (meta.get("fov_deg"), max_range, meta.get("crop_frac")))

    st = fps_stats(run["frames"])
    print("\n--- FPS (o sim rodava a 20 Hz) ---")
    print("mediana %.1f Hz   pior frame %.1f Hz (dt %.3f s)"
          % (st["fps_median"], st["fps_min"], st["dt_max"]))

    print("\n--- MEDICAO #1: arco de oclusao da carroceria ---")
    arc = occlusion_arc(sectors, max_range)
    if not arc:
        print("nenhum setor sempre bloqueado. Se o LiDAR estava montado no carro,")
        print("desconfie: ou o feixe passa por cima da carroceria, ou faltou dado.")
    else:
        graus = [sector_to_deg(i, n_sectors) for i in arc]
        print("%d de %d setores (%.0f%% do circulo) sempre bloqueados"
              % (len(arc), n_sectors, 100.0 * len(arc) / n_sectors))
        print("angulos: %.1f deg a %.1f deg" % (min(graus), max(graus)))
        print("-> fov_deg sugerido para o retreino: %.0f"
              % (360.0 * (n_sectors - len(arc)) / n_sectors))

    print("\n--- MEDICAO #2: onde estao os retornos mais proximos ---")
    if sectors.size:
        medianas = np.median(sectors, axis=0)
        ordem = np.argsort(medianas)[:5]
        print("5 setores mais proximos (compare com onde voce pos o objeto):")
        for i in ordem:
            print("   setor %2d = %6.1f deg   mediana %.3f" % (i, sector_to_deg(i, n_sectors), medianas[i]))

    print("\n--- ocupacao dos setores (quanto a entrada real parece a de treino) ---")
    if sectors.size:
        ocupados = float((sectors < 0.999).mean())
        print("%.1f%% dos valores tem retorno (no dataset_track_v1: frente 45.1%%)"
              % (100.0 * ocupados))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_car_log.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, **189 passed** (182 da Task 7 + 7 desta)

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_car_log.py tests/test_analyze_car_log.py tests/scripts_analyze.py
git commit -m "feat(car): analise do log -- arco de oclusao, FPS e ocupacao dos setores

Transforma o teste com rodas no ar nas medicoes de bancada #1 e #2, com numero.
A carroceria e identificada por ser o unico obstaculo que nunca se move: curta E
constante, enquanto uma parede perto varia conforme o carro gira."
```

---

### Task 10: Documentação e fechamento

**Files:**
- Modify: `docs/ESTADO_FASE6.md`
- Modify: `docs/ESTADO_IA.md`

- [ ] **Step 1: Atualizar o handoff**

Em `docs/ESTADO_FASE6.md`, seção `### 4.0`, substituir o parágrafo que começa com "**O que falta construir:**" por:

```markdown
**Construído (2026-08-23).** `hardware/jetson_runtime.py` é o executável: adaptadores de
câmera CSI, LiDAR serial, engine TensorRT e PCA9685, mais o `main`. A lógica que dá para
errar vive em `src/ai/car/` e é testada no PC: parser do COIN-D6, montagem da volta,
mapa do servo, config e o `DriveLoop` (que prova, com dublês, que o servo centraliza sem
dado e que o ESC nunca sai de neutro). O `scripts/analyze_car_log.py` transforma o log
nas medições #1 e #2.

Rodar:
    /usr/src/tensorrt/bin/trtexec --onnx=driving_track_180.onnx \
        --saveEngine=driving_track_180.engine --fp16
    python3 hardware/jetson_runtime.py --engine driving_track_180.engine \
        --config driving_track_180.json --out runs/car_teste1 --seconds 60
    # de volta no PC:
    python scripts/analyze_car_log.py runs/car_teste1
```

- [ ] **Step 2: Atualizar o mapa do código**

Em `docs/ESTADO_IA.md`, seção 3, logo depois do item de `shared/lidar_pipeline.py`, inserir:

```markdown
**Carro (`src/ai/car/`, Fase 6b — puro e Python 3.6-safe, testado no PC):**
- `coin_d6.py` — parser canônico do LiDAR real, emitindo `(ângulo, distância)`. Substitui
  as duas cópias divergentes que havia em `hardware/`.
- `scan_assembly.py` — agrupa os pontos em **voltas completas** (fecha no wrap do ângulo).
  Fechar por contagem deixaria setores não varridos lendo "livre" — buraco na parede.
- `control_map.py` — `steer` → microssegundos do servo, clampado; watchdog de frame.
- `image_crop.py` — recorte central, para aproximar a lente de 130° do FOV de 62.2° do treino.
- `config.py` — lê o sidecar JSON do modelo; `car_max_range()` deriva o 1.0 do carro de 12.0/12.
- `run_log.py` — grava o log da corrida (jsonl + npy).
- `loop.py` — `DriveLoop` com hardware injetado; é aqui que o envelope de segurança é testado.
```

E atualizar a contagem de testes de `125` para o número que o `pytest -q` imprimir.

- [ ] **Step 3: Commit**

```bash
git add docs/ESTADO_FASE6.md docs/ESTADO_IA.md
git commit -m "docs(ai): registra o runtime do carro e o script de analise do log"
```
