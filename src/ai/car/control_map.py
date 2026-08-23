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
    the safe command. Infinity is clamped to the nearest extreme.
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
