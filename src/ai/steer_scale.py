"""Calibracao do esterco entre o carro real e o simulador (Fase 6c).

O bug que este modulo existe para consertar, medido em 2026-10-03:

O ``steer`` normalizado que o modelo aprende e ``angulo_da_roda / angulo_maximo``.
No simulador esse maximo e o do Tesla Model 3 do CARLA, **70 graus**. No carro, o
curso total do servo corresponde ao maximo FISICO do WLtoys 124017 V2, **27 graus
medidos**. O mesmo numero normalizado significava angulos quase 3x diferentes: o
modelo pedia 42 graus de roda e o carro entregava 15. Resultado -- subesterco em
toda curva, sem nenhum erro no terminal.

**O que tem de casar e o RAIO escalado, nao o angulo.** O Tesla tem 2.875 m de
entre-eixos e o carro 0.21 m, que multiplicado por 12 da 2.52 m: 14% de
diferenca. Copiar os 27 graus para o simulador produziria raios diferentes.
Igualando o raio, o angulo do sim sai em ~30.2 graus.

Depois desta calibracao ``steer = 1`` significa o MESMO raio nos dois lados, e o
ganho de esterco no runtime volta a 1.0 -- o remendo deixa de existir.

Numeros medidos, nao estimados:
  - 27 graus: batente do servo, medido com o carro no chao em 2026-10-03;
  - 0.21 m / 0.208 m: entre-eixos e largura do WLtoys 124017 V2 (catalogo
    35.6 x 20.8 cm);
  - 0.53 m: faixa util da pista (``track_builder.LARGURA_PISTA``). A curva e um
    1/4 de disco com raio interno ZERO, entao a unica parede que limita e a
    externa.

Matematica pura -- nao importa ``carla``, roda nos testes sem simulador.
"""
import math

# --- carro real, MEDIDO ---
#
# O RAIO e a medida primaria, nao o angulo. Medido em 2026-10-03 tracando a
# circunferencia do centro do eixo TRASEIRO com trava total: 125 cm de diametro.
# Derivar o raio de um angulo de protractor errou 34% (27 graus previam 0.412 m),
# porque o protractor mede a geometria parada e o carro e 4WD -- em trava total as
# dianteiras arrastam e ele abre. O que vale e o que o carro FAZ.
CAR_MIN_RADIUS_M = 0.625
CAR_WHEELBASE_M = 0.21          # usado so para relatar o angulo equivalente
# 36 x 21 cm MEDIDOS no carro (o catalogo diz 35,6 x 20,8; vale a trena).
CAR_LENGTH_M = 0.36
CAR_WIDTH_M = 0.21

# Faixa util MEDIDA na pista fisica em 2026-10-03. O track_builder modela 0.53
# (as pecas vieram do Blender com essa medida), entao o simulador e 3 cm mais
# APERTADO que a realidade -- erro na direcao segura: um modelo treinado na pista
# estreita transfere para a larga, nao o contrario. Nao vale re-modelar os props.
LANE_WIDTH_M = 0.56
LANE_WIDTH_SIM_M = 0.53

# --- escala do gemeo digital ---
SCALE = 12.0

# Entre-eixos do vehicle.tesla.model3 no CARLA. Usado como PADRAO; quando ha um
# veiculo de verdade em maos, passe o valor lido de `get_physics_control()`, que
# e o que o Pure Pursuit ja faz com `_wheelbase()`.
SIM_WHEELBASE_M = 2.875


def turning_radius_m(wheelbase_m, steer_deg):
    """Raio de curva do modelo de bicicleta: ``R = L / tan(delta)``.

    Esterco zero devolve ``inf`` -- ir reto e raio infinito, nao raio zero.
    Devolver 0 inverteria o significado e o chamador leria "curva fechadissima".
    """
    L = float(wheelbase_m)
    d = float(steer_deg)
    if L <= 0.0:
        raise ValueError("entre-eixos tem de ser > 0 (recebi {0})".format(wheelbase_m))
    if d < 0.0 or d >= 90.0:
        raise ValueError(
            "angulo de roda tem de estar em [0, 90) graus (recebi {0})".format(steer_deg))
    if d == 0.0:
        return float("inf")
    return L / math.tan(math.radians(d))


def steer_deg_for_radius(wheelbase_m, radius_m):
    """Angulo de roda que traca um raio dado: ``delta = atan(L / R)``."""
    L = float(wheelbase_m)
    R = float(radius_m)
    if L <= 0.0:
        raise ValueError("entre-eixos tem de ser > 0 (recebi {0})".format(wheelbase_m))
    if R <= 0.0:
        raise ValueError("raio tem de ser > 0 (recebi {0})".format(radius_m))
    return math.degrees(math.atan(L / R))


def sim_max_steer_deg(car_min_radius_m=CAR_MIN_RADIUS_M,
                      sim_wheelbase_m=SIM_WHEELBASE_M,
                      scale=SCALE):
    """Angulo maximo de roda a impor ao carro do SIMULADOR.

    Escolhido para que o raio minimo do simulador, dividido pela escala, seja
    igual ao raio minimo do carro real. Nao e o angulo do carro: veiculos de
    entre-eixos diferentes precisam de angulos diferentes para o mesmo raio.

    Entra o RAIO medido, nao um angulo. Partir do angulo exigiria tambem o
    entre-eixos do carro, e os dois entrariam com erro -- foi assim que a versao
    anterior deste modulo errou 34%.

    Este numero vai para DOIS lugares, e os dois sao obrigatorios:
      1. ``max_steer_deg`` do Pure Pursuit -- o denominador da normalizacao;
      2. ``max_steer_angle`` das rodas dianteiras na physics control do CARLA.

    Mudar so o (1) faz o expert comandar 1.0 enquanto o carro do sim ainda gira
    70 graus: ele corta a curva, oscila, e o traçado gravado nao vale nada.
    """
    return steer_deg_for_radius(sim_wheelbase_m, float(car_min_radius_m) * float(scale))


def max_usable_radius_m(lane_width_m=LANE_WIDTH_M, vehicle_width_m=CAR_WIDTH_M):
    """Maior raio que atravessa a curva sem encostar na parede EXTERNA.

    A curva de 90 graus e 1/4 de disco com raio externo igual a largura da faixa
    e raio interno zero, entao nao ha parede interna para limitar. Um arco que
    entra por uma reta e sai pela outra a 90 graus e obrigatoriamente concentrico
    com o disco (o centro cai no vertice), logo o unico limite e meia largura do
    veiculo ate a borda externa.
    """
    return float(lane_width_m) - float(vehicle_width_m) / 2.0


def corner_fits(min_radius_m, lane_width_m=LANE_WIDTH_M,
                vehicle_width_m=CAR_WIDTH_M):
    """O veiculo consegue atravessar a curva de 90 graus com esse raio minimo?

    Serve de porta antes de coletar: se o carro do simulador nao couber na curva
    do simulador, o expert bate e o dataset inteiro nasce invalido.
    """
    return bool(float(min_radius_m) <= max_usable_radius_m(lane_width_m, vehicle_width_m))


def steering_wheel_indices(max_steer_angles):
    """Indices das rodas que esterçam, dadas as suas angulacoes maximas atuais.

    A ordem das rodas na physics control do CARLA e convencao, nao garantia.
    Assumir "as duas primeiras sao as dianteiras" e errar em silencio no veiculo
    em que a ordem difere: limitariamos as traseiras e deixariamos as dianteiras
    nos 70 graus, ou seja, o bug original intacto.

    Recusa um veiculo sem nenhuma roda esterçante em vez de devolver lista
    vazia: uma coleta inteira rodaria com o esterco errado sem nenhum aviso.
    """
    idx = [i for i, a in enumerate(max_steer_angles) if float(a) > 0.0]
    if not idx:
        raise ValueError(
            "nenhuma roda com angulo maximo > 0: nao da para calibrar o esterco "
            "deste veiculo (angulos recebidos: {0})".format(list(max_steer_angles)))
    return idx
