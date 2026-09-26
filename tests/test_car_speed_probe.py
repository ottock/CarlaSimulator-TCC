"""Velocidade real do carro a um PWM fixo (medicao de bancada #3 da Fase 6b).

Por que isto existe: no carro o modelo comanda APENAS o esterco -- o acelerador e
um PWM constante. Mas o modelo aprendeu a estercar a uma velocidade so: nos 14.400
frames do `dataset_track_v1` a distribuicao e quase uma delta (p5 1.69, p50 1.73,
p95 1.80 m/s). Esterco e velocidade nao sao independentes: a mesma curva ao dobro
da velocidade pede comando diferente. Se o ESC andar bem mais rapido que o alvo
escalado, o carro corta as curvas -- e isso e falha LATERAL, dentro do escopo.

A conta e trivial e e exatamente por isso que ela mora aqui, testada: um erro de
divisao aqui vira um numero que alguem copia para o texto do TCC.
"""
import pytest

from ai.car.speed_probe import (
    SIM_SPEED_P50_MS,
    TARGET_CAR_SPEED_MS,
    average_speed_ms,
    speed_report,
    steady_speed_ms,
    to_sim_ms,
)


def test_the_target_comes_from_the_dataset_not_from_the_expert_setpoint():
    # 1.73 e a velocidade ATINGIDA (p50 medido), nao os 2.0 do target_speed do
    # expert. Os docs ja citaram 0.167 por confundir os dois.
    assert SIM_SPEED_P50_MS == pytest.approx(1.73)
    assert TARGET_CAR_SPEED_MS == pytest.approx(1.73 / 12.0)
    assert TARGET_CAR_SPEED_MS == pytest.approx(0.1442, abs=1e-4)


def test_average_speed_is_distance_over_time():
    assert average_speed_ms(1.44, 10.0) == pytest.approx(0.144)


def test_a_zero_or_negative_duration_raises_instead_of_dividing():
    # Sem isto a saida seria inf m/s, que parece um numero e nao e.
    with pytest.raises(ValueError):
        average_speed_ms(1.0, 0.0)
    with pytest.raises(ValueError):
        average_speed_ms(1.0, -3.0)


def test_a_negative_distance_raises():
    with pytest.raises(ValueError):
        average_speed_ms(-0.5, 10.0)


def test_two_point_cancels_the_startup_ramp():
    # O carro parte do zero, entao a media de uma corrida curta SUBESTIMA a
    # velocidade de regime. Com duas corridas a perda da partida se cancela:
    # d(t) = v*t - c  =>  (d2-d1)/(t2-t1) = v, qualquer que seja c.
    v, c = 0.20, 0.06
    d1, d2 = v * 5.0 - c, v * 15.0 - c
    assert steady_speed_ms(5.0, d1, 15.0, d2) == pytest.approx(v)
    # e a media simples de fato erra para baixo
    assert average_speed_ms(d1, 5.0) < v


def test_two_point_needs_the_longer_run_second():
    with pytest.raises(ValueError):
        steady_speed_ms(15.0, 3.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        steady_speed_ms(5.0, 1.0, 5.0, 1.0)


def test_the_car_cannot_travel_backwards_between_the_two_runs():
    with pytest.raises(ValueError):
        steady_speed_ms(5.0, 2.0, 15.0, 1.0)


def test_scaling_back_to_the_simulator_is_the_inverse_of_the_target():
    assert to_sim_ms(TARGET_CAR_SPEED_MS) == pytest.approx(SIM_SPEED_P50_MS)
    assert to_sim_ms(0.25) == pytest.approx(3.0)


def test_a_speed_inside_the_training_spread_is_reported_as_seen():
    r = speed_report(TARGET_CAR_SPEED_MS)
    assert r["veredito"] == "dentro"
    assert r["razao"] == pytest.approx(1.0)
    assert r["sim_equivalente_ms"] == pytest.approx(SIM_SPEED_P50_MS)


def test_the_training_band_is_the_datasets_own_p5_p95_not_a_round_number():
    # A faixa "o modelo ja viu isto" e estreita de proposito: e a largura real da
    # distribuicao do dataset, nao uma tolerancia inventada.
    assert speed_report(1.69 / 12.0)["veredito"] == "dentro"
    assert speed_report(1.80 / 12.0)["veredito"] == "dentro"
    assert speed_report(1.60 / 12.0)["veredito"] == "lento"
    assert speed_report(1.95 / 12.0)["veredito"] == "rapido"


def test_double_the_speed_is_flagged_as_invalidating_the_learned_steering():
    r = speed_report(2 * TARGET_CAR_SPEED_MS)
    assert r["veredito"] == "rapido"
    assert r["corta_curvas"] is True
    # perto do alvo o alerta nao dispara
    assert speed_report(1.80 / 12.0)["corta_curvas"] is False


def test_the_report_carries_the_numbers_needed_to_write_it_down():
    r = speed_report(0.30, distancia_m=3.0, tempo_s=10.0)
    assert r["velocidade_ms"] == pytest.approx(0.30)
    assert r["distancia_m"] == pytest.approx(3.0)
    assert r["tempo_s"] == pytest.approx(10.0)
    assert r["alvo_ms"] == pytest.approx(TARGET_CAR_SPEED_MS)
