"""Quando cutucar o ego (recuperacao) e quais frames contam como tal — puro, sem CARLA.

A recuperacao foi O conserto do covariate shift na Fase 2 (§5.2 do docs/ESTADO_IA.md):
o ego e' empurrado pra fora do centro e gravamos o expert voltando. O agendamento e'
so aritmetica de passos, entao mora aqui, testavel sem subir o simulador.

O estado (`_last`) e' o passo do ultimo teleporte e vale POR EPISODIO: como o `step`
recomeca em 0 a cada episodio, herdar esse relogio do episodio anterior o deixa no
futuro — nunca alcanca o intervalo (zero cutucao) e ainda rotula o episodio inteiro
como recuperacao. Chame `reset()` ao abrir cada episodio.
"""

_NEVER = -10 ** 9


class RecoveryScheduler:
    def __init__(self, interval_steps, window_steps, enabled=True):
        self.interval = max(1, int(interval_steps))
        self.window = int(window_steps)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        """Zera o relogio. Obrigatorio no inicio de cada episodio."""
        self._last = _NEVER

    def should_teleport(self, step, moving):
        """So cutuca com o ego em movimento e passado o intervalo desde a ultima."""
        return self.enabled and bool(moving) and (step - self._last) >= self.interval

    def mark(self, step):
        """Registra que o teleporte aconteceu neste passo."""
        self._last = step

    def is_recovering(self, step):
        """True nos frames da janela logo apos a cutucada (o expert voltando ao centro)."""
        return self.enabled and (step - self._last) < self.window
