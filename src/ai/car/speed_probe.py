"""Velocidade real do carro a um PWM fixo, e o que ela significa (Fase 6b).

Medicao de bancada #3, a unica das quatro que ficou pendente. No carro o modelo
comanda SOMENTE o esterco: o acelerador e um PWM constante (`--cruise-us`),
porque o controle longitudinal ficou fora do escopo. Isso nao torna a velocidade
irrelevante -- torna ela um parametro que alguem escolheu e ninguem mediu.

O motivo de medir: nos 14.400 frames do `dataset_track_v1` a velocidade e quase
uma delta (p5 1.69, p50 1.73, p95 1.80 m/s). O modelo nunca viu outra. Esterco e
velocidade nao sao independentes -- a mesma curva ao dobro da velocidade pede
comando diferente -- entao, se o ESC andar bem mais rapido que o alvo escalado, o
carro corta as curvas. Isso e falha LATERAL, dentro do escopo, e o conserto
honesto seria recoletar com `target_speed` maior e retreinar, nao ajustar ganho.

Python 3.6-safe: roda no Jetson.
"""

# Velocidade MEDIDA no dataset, nao o `target_speed=2.0` do expert. Os docs ja
# citaram 0.167 m/s por trocar um pelo outro; 2.0 era o pedido, 1.73 o atingido.
SIM_SPEED_P5_MS = 1.69
SIM_SPEED_P50_MS = 1.73
SIM_SPEED_P95_MS = 1.80

SCALE = 12.0

TARGET_CAR_SPEED_MS = SIM_SPEED_P50_MS / SCALE          # 0.144 m/s
BAND_LOW_MS = SIM_SPEED_P5_MS / SCALE
BAND_HIGH_MS = SIM_SPEED_P95_MS / SCALE

# Acima disto o esterco aprendido deixa de valer. Nao ha numero exato para isto;
# 1.25 e um julgamento, deixado como constante para ficar visivel e discutivel em
# vez de enterrado num `if`. Bem antes de 2x ja da para ver o carro cortando.
RAZAO_CORTA_CURVAS = 1.25


def average_speed_ms(distancia_m, tempo_s):
    """Velocidade media de uma corrida: distancia percorrida / tempo decorrido.

    Recusa tempo <= 0 em vez de dividir: o resultado seria ``inf``, que parece um
    numero, entra num print e acaba no texto do TCC.

    Cuidado de interpretacao: o carro parte do repouso, entao esta media inclui a
    arrancada e SUBESTIMA a velocidade de regime. Use :func:`steady_speed_ms`
    quando precisar do valor de regime.
    """
    d = float(distancia_m)
    t = float(tempo_s)
    if t <= 0.0:
        raise ValueError("tempo_s tem de ser > 0 (recebi {0})".format(t))
    if d < 0.0:
        raise ValueError("distancia_m nao pode ser negativa (recebi {0})".format(d))
    return d / t


def steady_speed_ms(t1_s, d1_m, t2_s, d2_m):
    """Velocidade de regime a partir de DUAS corridas de duracoes diferentes.

    A arrancada custa uma distancia fixa ``c``, a mesma nas duas corridas:
    ``d(t) = v*t - c``. Subtrair uma da outra elimina ``c`` sem precisar medir a
    rampa de aceleracao::

        (d2 - d1) / (t2 - t1) = v

    E o jeito de obter a velocidade de regime tendo so uma trena e um cronometro.
    A corrida mais longa vai em segundo lugar.
    """
    t1, d1, t2, d2 = float(t1_s), float(d1_m), float(t2_s), float(d2_m)
    if t2 <= t1:
        raise ValueError("t2_s tem de ser maior que t1_s (a corrida longa vem depois)")
    if d2 < d1:
        raise ValueError("d2_m menor que d1_m: a corrida mais longa andou menos?")
    return (d2 - d1) / (t2 - t1)


def startup_loss_m(velocidade_ms, tempo_s, distancia_m):
    """Distancia perdida na arrancada, em metros: ``v*t - d``.

    Util como sanidade: se der negativa ou enorme, uma das duas corridas nao foi
    em linha reta, ou o cronometro nao cobriu o mesmo intervalo do PWM.
    """
    return float(velocidade_ms) * float(tempo_s) - float(distancia_m)


def to_sim_ms(car_speed_ms, scale=SCALE):
    """Converte a velocidade do carro para a escala do simulador (x12)."""
    return float(car_speed_ms) * float(scale)


def speed_report(velocidade_ms, distancia_m=None, tempo_s=None):
    """Compara a velocidade medida com a que o modelo viu no treino.

    ``veredito`` e um de ``"lento"``, ``"dentro"``, ``"rapido"``, medido contra a
    faixa p5-p95 do PROPRIO dataset em vez de uma tolerancia redonda inventada:
    fora dela o modelo esta operando numa velocidade que nunca observou.
    """
    v = float(velocidade_ms)
    if v < BAND_LOW_MS:
        veredito = "lento"
    elif v > BAND_HIGH_MS:
        veredito = "rapido"
    else:
        veredito = "dentro"
    razao = v / TARGET_CAR_SPEED_MS
    return {
        "velocidade_ms": v,
        "sim_equivalente_ms": to_sim_ms(v),
        "alvo_ms": TARGET_CAR_SPEED_MS,
        "faixa_ms": (BAND_LOW_MS, BAND_HIGH_MS),
        "razao": razao,
        "veredito": veredito,
        "corta_curvas": bool(razao >= RAZAO_CORTA_CURVAS),
        "distancia_m": None if distancia_m is None else float(distancia_m),
        "tempo_s": None if tempo_s is None else float(tempo_s),
    }
