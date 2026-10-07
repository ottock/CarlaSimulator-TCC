"""O que o expert REALMENTE rotula no oval -- a medicao que explicou a sonda.

Contexto. A sonda estatica (2026-10-07) devolveu 4 de 6 poses "erradas", e a
coluna "esperado" dela era intuicao minha, nao medicao. Avaliando o proprio
Pure Pursuit nas poses, descobriu-se que a intuicao estava errada E que o
dataset tem um problema bem maior:

    rotulo na LINHA DE PILOTAGEM, volta inteira:
        media -0.584   desvio 0.257   min -0.729   max +0.000
        -0.73 em 71.7% da volta       0.0% positivo

Ou seja: em 70% da volta o rotulo e UM NUMERO SO. O oval tem 63 m de perimetro
no simulador e as curvas (raio constante 6.36 m) sao ~76% dele, entao o expert
passa a volta quase toda com o mesmo esterco. O sinal de treino quase nao
carrega informacao.

Com `--mirror` (ligado no v3) metade das amostras e espelhada e o rotulo troca
de sinal, o que torna o alvo BIMODAL em +/-0.73. A tarefa aprendida deixa de
ser "quanto esterçar" e passa a ser "escolher o lado, depois aplicar 0,73" --
e a sonda mostra exatamente isso: magnitudes certas (0,2 a 0,8) com sinais
instaveis, e o desvio explodindo (0,275 com o carro PARADO) onde a escolha fica
no fio da navalha.

Estes testes existem para esses numeros nao virarem folclore: se alguem mexer no
lookahead, na linha ou no max_steer, eles dizem o que mudou.
"""
import math

import pytest

from ai.racing_line import expert_path, min_radius_m, signed_offsets
from ai.steer_scale import CAR_LENGTH_M, CAR_MIN_RADIUS_M, CAR_WIDTH_M, SCALE
from ai.track_ref import track_centerline, track_width
from core.carlaClient.professor import PurePursuit

# Exatamente o que settings/pistaTCC.json passa ao professor.
LOOKAHEAD = 4.0
MAX_STEER_DEG = 27.0
WHEELBASE = 2.875
CFG = {"preset": "oval_tcc", "escala": "real", "flip_y": -1, "flip_yaw": -1, "z": 0.05}


@pytest.fixture(scope="module")
def pista():
    centerline = track_centerline(CFG)
    half = track_width(CFG) / 2.0
    trajeto = expert_path(centerline, half_width=half,
                          vehicle_width=CAR_WIDTH_M * SCALE,
                          vehicle_length=CAR_LENGTH_M * SCALE,
                          margin=0.0,
                          min_radius_required=CAR_MIN_RADIUS_M * SCALE,
                          modo="estadio")
    return centerline, half, trajeto


def _esterco(trajeto, x, y, yaw):
    """Esterco do expert numa pose isolada.

    `idx` e fixado no ponto globalmente mais proximo porque o PurePursuit busca
    so para frente numa janela: sem isto uma pose avaliada fora de sequencia
    miraria um trecho ja passado.
    """
    pp = PurePursuit(trajeto, wheelbase=WHEELBASE, lookahead=LOOKAHEAD,
                     target_speed=2.0, k_throttle=0.5, max_steer_deg=MAX_STEER_DEG)
    pp.idx = min(range(len(trajeto)),
                 key=lambda k: (trajeto[k][0] - x) ** 2 + (trajeto[k][1] - y) ** 2)
    return pp.control(x, y, yaw, 2.0)[0]


def _rotulos_na_linha(trajeto, passo=2):
    return [_esterco(trajeto, x, y, yaw) for x, y, yaw in trajeto[::passo]]


def test_the_oval_is_the_size_we_measured_with_a_tape(pista):
    """159 x 106 cm de eixo. Se isto mudar, todo o resto abaixo muda."""
    centerline, _half, _t = pista
    xs = [p[0] for p in centerline]
    ys = [p[1] for p in centerline]
    cm = 100.0 / SCALE
    assert (max(xs) - min(xs)) * cm == pytest.approx(159.0, abs=1.0)
    assert (max(ys) - min(ys)) * cm == pytest.approx(106.0, abs=1.0)


def test_the_expert_never_steers_right_on_this_track(pista):
    """O oval vira so para um lado. E a razao de `--mirror` existir -- e,
    descoberto depois, a razao de ele atrapalhar: o mundo que ele inventa
    (virar para a direita) nao existe nesta pista."""
    _c, _h, trajeto = pista
    rotulos = _rotulos_na_linha(trajeto)
    assert max(rotulos) <= 0.05, "apareceu esterco para a direita: %+.3f" % max(rotulos)


def test_most_of_the_lap_carries_a_single_label(pista):
    """O achado central: o rotulo e quase constante, entao ha pouco a aprender.

    Se um dia isto cair abaixo de 50%, otimo -- quer dizer que a volta deixou de
    ser um giro de raio constante e o sinal de treino ficou mais rico."""
    _c, _h, trajeto = pista
    rotulos = _rotulos_na_linha(trajeto)
    na_curva = [v for v in rotulos if abs(v + 0.729) < 0.03]
    fracao = float(len(na_curva)) / len(rotulos)
    assert fracao > 0.6, ("o rotulo de curva constante caiu para %.0f%% da volta"
                          % (100 * fracao))


def test_the_constant_is_the_geometry_of_the_corner(pista):
    """-0.73 nao e numero magico: e atan(L/R)/max_steer no raio DA LINHA.

    O raio e o da linha de pilotagem (8.03 m), nao o do eixo (6.36 m) -- usar o
    do eixo da 0.90 e foi o meu primeiro palpite, errado. A linha abre a curva
    de proposito, e essa abertura e o que cabe no carro: 8.03/12 = 67 cm contra
    os 47 cm de raio minimo medidos, ou seja 31% de reserva.

    Amarra o rotulo a geometria, para um erro de escala aparecer aqui em vez de
    virar "o modelo esterça pouco" na pista.
    """
    _c, _h, trajeto = pista
    raio = min_radius_m([(x, y) for x, y, _ in trajeto])
    assert raio == pytest.approx(8.03, abs=0.2)
    esperado = math.atan(WHEELBASE / raio) / math.radians(MAX_STEER_DEG)
    assert esperado == pytest.approx(0.729, abs=0.03)
    # e sobra reserva de esterco sobre o batente real do carro
    assert raio / SCALE > CAR_MIN_RADIUS_M * 1.2


def test_the_racing_line_runs_near_the_outer_wall(pista):
    """A linha corre ~17 cm do eixo, deixando ~6 cm entre o flanco e a parede.

    E a instrucao que a sonda da ao operador; se a linha mudar, a instrucao
    precisa mudar junto.
    """
    centerline, half, trajeto = pista
    offs = signed_offsets(trajeto, centerline)
    media_cm = sum(offs) / len(offs) * 100.0 / SCALE
    assert media_cm == pytest.approx(11.6, abs=2.0)
    folga_cm = (half * 100.0 / SCALE) - media_cm - (CAR_WIDTH_M * 100.0) / 2.0
    assert folga_cm == pytest.approx(6.0, abs=2.0)


def test_off_the_line_the_expert_wants_a_big_correction(pista):
    """Colado na parede o expert comanda esterco forte para o corredor -- e sao
    justamente esses quadros que o dataset quase nao tem (36 de ~14.400).

    Isto separa as duas acusacoes possiveis da sonda: a pose existe no treino ou
    nao existe.
    """
    centerline, half, trajeto = pista
    lim = half - (CAR_WIDTH_M * SCALE) / 2.0
    fortes = 0
    amostras = 0
    for x, y, yaw in centerline[::6]:
        rx, ry = -math.sin(yaw), math.cos(yaw)
        for lado in (-1.0, 1.0):
            s = _esterco(trajeto, x + lado * rx * lim, y + lado * ry * lim, yaw)
            amostras += 1
            if abs(s) > 0.5:
                fortes += 1
    assert fortes > amostras * 0.6, (
        "colado na parede o expert deveria pedir correcao forte quase sempre; "
        "pediu em %d de %d" % (fortes, amostras))
