"""Quantos episodios cada pista recebe na coleta (2026-10-09)."""
import pytest

pytest.importorskip("carla")
from ai.collect_track import pista_e_episodios  # noqa: E402


def test_a_star_gives_one_track_its_own_episode_count():
    assert pista_e_episodios("oval_tcc*8", 2) == ("oval_tcc", 8)


def test_without_a_star_the_default_holds():
    assert pista_e_episodios("grade:SDSES", 2) == ("grade:SDSES", 2)


def test_nonsense_counts_are_refused():
    with pytest.raises(ValueError):
        pista_e_episodios("oval_tcc*0", 2)
    with pytest.raises(ValueError):
        pista_e_episodios("*3", 2)


def test_the_collection_hands_the_inner_margin_to_the_expert_line():
    import numpy as np
    from ai.collect_track import _trajeto_do_expert
    from ai.steer_scale import CAR_LENGTH_M, CAR_MIN_RADIUS_M, CAR_WIDTH_M, SCALE
    from ai.tracado_geral import tracado_geral
    from ai.track_ref import track_centerline, track_width

    cfg = {"preset": "grade:SDSESESES", "escala": "real"}   # a da coleta
    eixo = track_centerline(cfg)
    com = _trajeto_do_expert("geral", eixo, cfg, 0.36, fechado=False, margem_dentro=0.84)
    esperado = tracado_geral(np.array(eixo), False, track_width(cfg) / 2.0,
                             CAR_WIDTH_M * SCALE, CAR_LENGTH_M * SCALE, 0.36,
                             raio_ref=CAR_MIN_RADIUS_M * SCALE, margin_dentro=0.84)
    assert np.allclose(np.array(com), esperado)
