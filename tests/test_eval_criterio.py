"""Criterio de aprovacao da avaliacao em malha fechada (Fase 6c).

Por que existe: o eval_track reportou "1/1 pistas limpas" para uma corrida em
que o carro NAO SAIU DO LUGAR -- mean_speed=0.0, mean_dev=0.00, offlane=0,
collisions=0. Sem sair da pista e sem bater, o criterio dava aprovado.

Um carro parado nao bate e nao sai da pista. "Limpo" tem de exigir que ele
tenha ANDADO.
"""
import pytest

from ai.eval_criterio import corrida_limpa, motivo


def _s(**kw):
    base = {"offlane": 0, "collisions": 0, "mean_speed": 1.5, "distance_m": 100.0}
    base.update(kw)
    return base


def test_a_normal_clean_run_passes():
    assert corrida_limpa(_s()) is True


def test_a_car_that_never_moved_FAILS():
    # O caso real: driving_oval_v2 na primeira avaliacao.
    parado = _s(mean_speed=0.0, distance_m=0.0)
    assert corrida_limpa(parado) is False
    assert "parado" in motivo(parado)


def test_a_collision_fails():
    assert corrida_limpa(_s(collisions=3)) is False
    assert "colis" in motivo(_s(collisions=3))


def test_leaving_the_lane_fails():
    assert corrida_limpa(_s(offlane=12)) is False
    assert "pista" in motivo(_s(offlane=12))


def test_crawling_counts_as_not_driving():
    # 0,2 m/s nao e dirigir, e arrastar. O expert anda a 1,74.
    assert corrida_limpa(_s(mean_speed=0.2, distance_m=20.0)) is False


def test_a_short_run_fails_even_at_speed():
    # Velocidade media boa mas distancia curta: o carro andou um pouco e travou,
    # e a media so conta enquanto ele andava.
    assert corrida_limpa(_s(mean_speed=1.7, distance_m=5.0)) is False


def test_the_reason_is_empty_when_it_passes():
    assert motivo(_s()) == ""


# Pista com fim (2026-10-09)

def test_an_open_track_is_clean_when_the_car_stops_at_the_end():
    from ai.eval_criterio import corrida_limpa
    curta = {"collisions": 0, "offlane": 0, "mean_speed": 1.5, "distance_m": 12.0,
             "chegou_ao_fim": True, "restante_m": 5.0}
    assert corrida_limpa(curta)                      # 12 m e normal numa pista curta


def test_stopping_before_the_end_fails_an_open_track():
    from ai.eval_criterio import motivo
    m = motivo({"collisions": 0, "offlane": 0, "mean_speed": 1.5, "distance_m": 30.0,
                "chegou_ao_fim": False, "restante_m": 20.0})
    assert "nao chegou ao fim" in m


def test_a_crash_still_fails_an_open_track():
    from ai.eval_criterio import motivo
    assert motivo({"collisions": 1, "offlane": 0, "mean_speed": 1.5,
                   "chegou_ao_fim": True}).startswith("colisoes")
