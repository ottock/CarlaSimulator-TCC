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
