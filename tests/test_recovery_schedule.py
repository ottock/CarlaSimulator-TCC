"""Agendador da recuperacao (cutucao) da coleta — matematica pura, sem CARLA."""
from ai.recovery_schedule import RecoveryScheduler


def _run_episode(sch, steps, moving=True):
    """Roda um episodio; devolve (n_teleportes, n_frames_rotulados_recuperacao)."""
    teleports = recovering = 0
    for step in range(steps):
        if sch.should_teleport(step, moving):
            sch.mark(step)
            teleports += 1
        recovering += 1 if sch.is_recovering(step) else 0
    return teleports, recovering


def test_teleporta_no_intervalo_e_rotula_so_a_janela():
    sch = RecoveryScheduler(interval_steps=100, window_steps=30)   # 5 s / 1,5 s @ 20 Hz
    teleports, recovering = _run_episode(sch, 1200)                # episodio de 60 s
    assert teleports == 12
    assert recovering == 12 * 30       # so os frames logo apos cada cutucao


def test_nao_cutuca_o_carro_parado():
    """A recuperacao so vale com o ego em movimento (§5.2): teleportar o carro
    parado corrompe o exemplo."""
    sch = RecoveryScheduler(interval_steps=100, window_steps=30)
    assert _run_episode(sch, 1200, moving=False) == (0, 0)


def test_reset_por_episodio_restaura_a_recuperacao():
    """Regressao do bug do dataset_track_v1: o `step` recomeca em 0 a cada episodio,
    entao um `last_teleport` herdado do episodio anterior fica no futuro — nunca
    alcanca o intervalo (zero cutucao) e deixa o episodio inteiro rotulado como
    recuperacao. So o reset por episodio evita isso."""
    sch = RecoveryScheduler(interval_steps=100, window_steps=30)
    primeiro = _run_episode(sch, 1200)
    sch.reset()
    assert _run_episode(sch, 1200) == primeiro


def test_sem_reset_o_episodio_seguinte_degenera():
    """Documenta explicitamente o modo de falha que o reset previne."""
    sch = RecoveryScheduler(interval_steps=100, window_steps=30)
    _run_episode(sch, 1200)                          # episodio 1, sem reset depois
    teleports, recovering = _run_episode(sch, 1200)  # episodio 2 herda o relogio
    assert teleports == 0
    assert recovering > 1000


def test_desligado_nao_faz_nada():
    sch = RecoveryScheduler(interval_steps=100, window_steps=30, enabled=False)
    assert _run_episode(sch, 1200) == (0, 0)
