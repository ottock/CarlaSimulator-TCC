"""Model output -> car actuator commands (Fase 6b).

Last stop before the hardware, so it must be safe with garbage input: a NaN or an
out-of-range value must never become an extreme servo command. Python 3.6-safe --
this runs on the Jetson.

Numbers come from ``hardware/controle_teste.py``, already validated on the car.
"""

STEER_CENTER_US = 1500
# Batente MEDIDO no carro em 2026-09-12 com hardware/calibra_servo.py, passo a
# passo ate a roda parar de responder: delta maximo 300 us para cada lado.
# USAMOS 260, com 40 us de folga do batente: comandar exatamente no limite
# mecanico faz o servo forcar e aquecer quando o modelo pede esterco total.
# Antes eram 200, herdados do controle por teclado do controle_teste.py --
# conservadores e nunca medidos. Com 200, steer=1 entregava dois tercos do curso
# real, entao TODO comando de esterco saia reduzido em 33% e o carro subvirava em
# tudo, saturando ou nao. O modelo aprendeu steer=+/-1 significando BATENTE
# TOTAL, entao este numero tem de ser o batente fisico -- nao e ajuste fino.
# RE-MEDIR se a geometria da direcao mudar.
# SINAL MEDIDO em 2026-09-12 com hardware/teste_esquerda.py: comandando
# steer = -1 (ESQUERDA na convencao do CARLA) as rodas foram para a DIREITA.
# O servo deste carro responde ESPELHADO.
#
# Isso importa muito mais do que parece: o espelho fica DEPOIS do modelo, entao a
# rede pedia uma coisa e o carro fazia a oposta. Todas as corridas de pista ate
# aqui mostraram o carro executando o INVERSO da decisao do modelo, e nenhuma
# delas testou o modelo de fato.
#
# A inversao mora numa constante so, de proposito. Um sinal trocado no meio da
# expressao seria encontrado meses depois, por alguem refazendo este mesmo teste.
STEER_SIGN = -1

# us que fisicamente viram cada lado NESTE carro (com o espelho ja aplicado).
STEER_LEFT_US = 1760
STEER_RIGHT_US = 1240
STEER_SPAN_US = 260
ESC_NEUTRAL_US = 1500


def steer_to_us(steer, span_us=STEER_SPAN_US):
    """Map ``steer`` in [-1, 1] to servo microseconds, clamped to the stops.

    CARLA's convention (+1 = right) matches the car's (higher us = right).
    Anything unusable -- ``None``, NaN, a non-number -- returns centre, which is
    the safe command. Infinity is clamped to the nearest extreme.

    ``span_us`` is the half-range that ``steer = 1`` must reach, and it has to be
    the servo's REAL mechanical limit: the model learned ``+/-1`` meaning full
    lock, so a span narrower than the car's actual lock scales down every single
    steering command and the car understeers everywhere. Measure it with
    ``hardware/controle_pwm_steering.py``, wheels off the ground.
    """
    try:
        s = float(steer)
    except (TypeError, ValueError):
        return STEER_CENTER_US
    if s != s:  # NaN
        return STEER_CENTER_US
    # Clamp s to [-1, 1] before rounding to handle infinity safely while
    # preserving intent: +inf means "hard right", -inf means "hard left".
    s = max(-1.0, min(1.0, s))
    us = int(round(STEER_CENTER_US + STEER_SIGN * s * span_us))
    return max(STEER_CENTER_US - span_us, min(STEER_CENTER_US + span_us, us))


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


# ---------------------------------------------------------------------------
# Velocidade de cruzeiro constante (Fase 6b parte 2)
#
# Decisao: o carro anda a uma velocidade CONSTANTE, sem aceleracao; e se ela nao
# puder ser mantida, vai a ZERO. O modelo comanda apenas o esterco -- as cabecas
# de throttle/brake sao ignoradas de proposito (a de freio esta inerte: o
# dataset_track_v1 tem 0% de frenagem, entao ela regride sempre para ~0).
# ---------------------------------------------------------------------------

ESC_MIN_MOVE_US = 1600   # abaixo disto o ESC nao move (zona morta medida na bancada)
ESC_MAX_US = 1700


def clamp_cruise_us(us):
    """Sanitise the constant-cruise PWM before it reaches the ESC.

    Rules, in order of how badly they would end:
      - unusable input (``None``, NaN, not a number) -> neutral;
      - anything at or below neutral, including reverse -> neutral (reverse is
        not part of this phase);
      - inside the ESC dead zone (below ``ESC_MIN_MOVE_US``) -> neutral, because
        the car would not move there anyway and reporting neutral keeps the
        commanded value honest instead of pretending;
      - above ``ESC_MAX_US`` -> clamped down to the calibrated maximum.
    """
    try:
        value = float(us)
    except (TypeError, ValueError):
        return ESC_NEUTRAL_US
    if value != value:  # NaN
        return ESC_NEUTRAL_US
    if value >= ESC_MAX_US:
        return ESC_MAX_US
    if value < ESC_MIN_MOVE_US:
        return ESC_NEUTRAL_US
    return int(round(value))


FRONT_HALF_ANGLE_DEG = 10.0


def _front_indices(n_sectors, half_angle_deg):
    """Sector indices whose centre falls inside the frontal cone.

    Same centre convention as ``apply_fov_mask`` ((i+0.5) of a sector), so the
    stop cone and the FOV mask talk about the same angles.
    """
    width = 360.0 / n_sectors
    out = []
    for i in range(n_sectors):
        angle = (i + 0.5) * width
        if angle > 180.0:
            angle -= 360.0
        if abs(angle) <= half_angle_deg:
            out.append(i)
    return out


def front_min(sectors, half_angle_deg=FRONT_HALF_ANGLE_DEG):
    """Nearest normalised reading inside the frontal cone (1.0 = free)."""
    n = len(sectors)
    idx = _front_indices(n, half_angle_deg)
    if not idx:
        return 1.0
    return float(min(float(sectors[i]) for i in idx))


def front_blocked(sectors, threshold, half_angle_deg=FRONT_HALF_ANGLE_DEG):
    """True when something sits closer than ``threshold`` straight ahead.

    This is the emergency stop, and deliberately a *safety* item rather than a
    learned behaviour: the model's brake head is inert, so stopping cannot be
    left to it. Only the frontal cone counts -- a wall passing by on the side is
    normal on a 0.53 m wide track and must not halt the car.
    """
    return bool(front_min(sectors, half_angle_deg) < float(threshold))
