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
