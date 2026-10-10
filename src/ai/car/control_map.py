"""Model output -> car actuator commands (Fase 6b).

Last stop before the hardware, so it must be safe with garbage input: a NaN or an
out-of-range value must never become an extreme servo command. Python 3.6-safe --
this runs on the Jetson.

Numbers come from ``hardware/controle_teste.py``, already validated on the car.
"""

STEER_CENTER_US = 1500
# Batente MEDIDO no carro em 2026-09-12 com hardware/calibra_servo.py, passo a
# passo ate a roda parar de responder: delta maximo 300 us para cada lado.
# USAMOS 350 desde 2026-10-04, CONFIRMADO no servo: entre 300 e 350 a roda ainda
# ganha angulo (pouco, mas ganha), sem zumbido e sem o servo travado -- ou seja,
# 350 esta no fim do curso util e nao forcando contra ele. Em 400 o circulo e
# identico ao de 350, entao dali em diante so sobra forca.
# Historico: Os 40 us de folga que
#   span 260 -> circulo 125 cm -> R 0,625 m ->  9% de reserva: o expert travava
#   span 300 -> circulo 107 cm -> R 0,535 m -> 22% de reserva: dirige, raspa
#   span 350 -> circulo  94 cm -> R 0,470 m -> 31% de reserva
# Os 260 originais deixavam 40 us de folga de um batente que, descobriu-se,
# nem era o batente: havia um fio solto no servo quando ele foi medido.
# O risco de forcar e aceitavel porque o traçado pede 78% do batente em regime;
# so transitorio chega ao limite.
# Antes eram 200, herdados do controle por teclado do controle_teste.py --
# conservadores e nunca medidos. Com 200, steer=1 entregava dois tercos do curso
# real, entao TODO comando de esterco saia reduzido em 33%.
# RE-MEDIR se a geometria da direcao mudar.
#
# CORRECAO IMPORTANTE (2026-09-26). Este bloco dizia "o modelo aprendeu
# steer=+/-1 significando BATENTE TOTAL". E FALSO, e foi medido:
#
#     nos 14.400 quadros do dataset_track_v1 o MAIOR |steer| do expert e 0.826,
#     e o p90 e 0.634. steer = 1.0 nao aparece uma unica vez.
#
# O Pure Pursuit a 2.0 m/s com lookahead 4.0 nunca precisou de mais que isso, e
# a rede nao produz o que nunca viu. Consequencia pratica: mesmo com span = 260
# o carro so alcanca 0.826 * 260 = 215 us dos 300 us de batente fisico -- 72%.
# Na curva real medida (runs/Diag PT2) ele usou o pico de 0.640, ou seja 55% do
# batente, e sustentou 0.284, ou 25%. O carro subvirou com tres quartos do
# esterco sobrando.
#
# Por isso `apply_steer_gain` NAO e' um remendo ate ~1.21 (= 1/0.826): ate ai ele
# so leva a faixa real de saida da rede ate a faixa real do servo, que e o que
# esta frase errada supunha ja acontecer. Acima disso vira compensacao para a
# velocidade (o carro anda ~7x mais rapido que o treino) e ai sim e' ajuste.
# SINAL MEDIDO em 2026-09-12 com hardware/teste_esquerda.py: comandando
# steer = -1 (ESQUERDA na convencao do CARLA) as rodas foram para a DIREITA.
# O servo deste carro responde ESPELHADO.
#
# Isso importa muito mais do que parece: o espelho fica DEPOIS do modelo, entao a
# rede pedia uma coisa e o carro fazia a oposta. Todas as corridas de pista ate
# aqui mostraram o carro executando o INVERSO da decisao do modelo, e nenhuma
# delas testou o modelo de fato.
#
# A inversao mora numa constante so, de proposito. Um sinal trocado no meio da
# expressao seria encontrado meses depois, por alguem refazendo este mesmo teste.
STEER_SIGN = -1

# us que fisicamente viram cada lado NESTE carro (com o espelho ja aplicado).
STEER_LEFT_US = 1850
STEER_RIGHT_US = 1150
STEER_SPAN_US = 350
ESC_NEUTRAL_US = 1500


def steer_to_us(steer, span_us=STEER_SPAN_US):
    """Map ``steer`` in [-1, 1] to servo microseconds, clamped to the stops.

    CARLA's convention (+1 = right) matches the car's (higher us = right).
    Anything unusable -- ``None``, NaN, a non-number -- returns centre, which is
    the safe command. Infinity is clamped to the nearest extreme.

    ``span_us`` is the half-range that ``steer = 1`` must reach, and it has to be
    the servo's REAL mechanical limit: the model learned ``+/-1`` meaning full
    lock, so a span narrower than the car's actual lock scales down every single
    steering command and the car understeers everywhere. Measure it with
    ``hardware/controle_pwm_steering.py``, wheels off the ground.
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
    us = int(round(STEER_CENTER_US + STEER_SIGN * s * span_us))
    return max(STEER_CENTER_US - span_us, min(STEER_CENTER_US + span_us, us))


def apply_steer_gain(steer, gain=1.0):
    """Amplify the model's steering before it becomes microseconds.

    MEDIDO em 2026-09-26: na curva a esquerda o carro virou para o lado certo e
    faltou angulo -- bateu com a quina frontal esquerda na parede externa, que e
    subesterco. Este ganho existe para testar isso em minutos, sem recoletar.

    E honesto sobre o que e: um ajuste de ATUADOR, aplicado depois da rede, nao
    um conserto do modelo. Duas consequencias que importam para interpretar o
    resultado:

      - se a rede ja estiver pedindo +/-1 dentro da curva, ganho nenhum muda
        nada (o clamp abaixo garante isso), e a causa esta em outro lugar --
        provavelmente na velocidade, que hoje e ~7x a do treino;
      - o valor usado precisa ficar no log da corrida, senao duas corridas com
        ganhos diferentes viram a mesma linha no relatorio.

    Um ganho NEGATIVO e recusado: ele espelharia o esterco inteiro, que e
    exatamente a falha silenciosa que custou todas as corridas ate 2026-09-12.
    """
    try:
        g = float(gain)
    except (TypeError, ValueError):
        g = 1.0
    if g != g:  # NaN
        g = 1.0
    if g < 0.0:
        raise ValueError("ganho de esterco nao pode ser negativo (recebi {0}): "
                         "isso espelharia o carro".format(gain))
    try:
        s = float(steer)
    except (TypeError, ValueError):
        return 0.0
    if s != s:  # NaN
        return 0.0
    return max(-1.0, min(1.0, s * g))


class SteerMedian:
    """Causal median filter on the steering command.

    MEDIDO em 2026-10-03 (runs/Diag/Diag, os dois modelos rodando sobre as MESMAS
    imagens gravadas, sem malha fechada): o modelo com aumento fotometrico inverte
    de lado em 22% dos quadros e salta de um batente ao outro em quadros isolados
    -- o padrao +0.4, +0.4, -0.88, +0.4. O servo nao acompanha isso: ele entrega a
    media, que e quase o centro.

    A mediana de 3 corta o ``|d steer|`` medio de 0.156 para 0.091 custando 10% da
    magnitude. Uma EMA alisa parecido mas custa 27%, e magnitude e exatamente o que
    falta para a curva.

    CAUSAL de proposito: usa apenas o quadro atual e os anteriores, porque no carro
    o quadro seguinte ainda nao existe. Custa ~1 quadro de atraso num degrau, que a
    13 Hz sao 7,7 cm de pista.

    HONESTIDADE SOBRE O ESCOPO: isto remove o tremor, NAO faz o carro curvar. No
    trecho de curva sustentada dos logs o comando e 0.241 com e sem filtro, e a
    curva exige o equivalente a 0.44. O conserto daquilo e retreinar com a
    normalizacao do esterco corrigida -- ver o bloco STEER_SPAN_US acima.

    Python 3.6-safe.
    """

    def __init__(self, window=3):
        w = int(window)
        if w < 1 or w % 2 == 0:
            raise ValueError(
                "janela da mediana tem de ser impar e >= 1 (recebi {0}): uma janela "
                "par faz a media dos dois centrais e devolve o pico que o filtro "
                "existe para remover".format(window))
        self.window = w
        self._buf = []

    def reset(self):
        """Esquece o historico -- entre corridas, para uma nao contaminar a outra."""
        self._buf = []

    def push(self, steer):
        """Add a sample and return the filtered command.

        Um valor inutilizavel (``None``, NaN, nao-numero) e DESCARTADO em vez de
        entrar na janela: envenenar o historico faria um unico quadro ruim sujar
        os proximos dois.
        """
        try:
            s = float(steer)
        except (TypeError, ValueError):
            s = None
        if s is not None and s != s:  # NaN
            s = None
        if s is not None:
            self._buf.append(s)
            if len(self._buf) > self.window:
                self._buf.pop(0)
        if not self._buf:
            return 0.0
        ordenado = sorted(self._buf)
        n = len(ordenado)
        if n % 2:
            return ordenado[n // 2]
        return (ordenado[n // 2 - 1] + ordenado[n // 2]) / 2.0


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


# ---------------------------------------------------------------------------
# Velocidade de cruzeiro constante (Fase 6b parte 2)
#
# Decisao: o carro anda a uma velocidade CONSTANTE, sem aceleracao; e se ela nao
# puder ser mantida, vai a ZERO. O modelo comanda apenas o esterco -- as cabecas
# de throttle/brake sao ignoradas de proposito (a de freio esta inerte: o
# dataset_track_v1 tem 0% de frenagem, entao ela regride sempre para ~0).
# ---------------------------------------------------------------------------

ESC_MIN_MOVE_US = 1600   # abaixo disto o ESC nao move (zona morta medida na bancada)
ESC_MAX_US = 1700


def clamp_cruise_us(us):
    """Sanitise the constant-cruise PWM before it reaches the ESC.

    Rules, in order of how badly they would end:
      - unusable input (``None``, NaN, not a number) -> neutral;
      - anything at or below neutral, including reverse -> neutral (reverse is
        not part of this phase);
      - inside the ESC dead zone (below ``ESC_MIN_MOVE_US``) -> neutral, because
        the car would not move there anyway and reporting neutral keeps the
        commanded value honest instead of pretending;
      - above ``ESC_MAX_US`` -> clamped down to the calibrated maximum.
    """
    try:
        value = float(us)
    except (TypeError, ValueError):
        return ESC_NEUTRAL_US
    if value != value:  # NaN
        return ESC_NEUTRAL_US
    if value >= ESC_MAX_US:
        return ESC_MAX_US
    if value < ESC_MIN_MOVE_US:
        return ESC_NEUTRAL_US
    return int(round(value))


FRONT_HALF_ANGLE_DEG = 10.0


def _front_indices(n_sectors, half_angle_deg):
    """Sector indices whose centre falls inside the frontal cone.

    Same centre convention as ``apply_fov_mask`` ((i+0.5) of a sector), so the
    stop cone and the FOV mask talk about the same angles.
    """
    width = 360.0 / n_sectors
    out = []
    for i in range(n_sectors):
        angle = (i + 0.5) * width
        if angle > 180.0:
            angle -= 360.0
        if abs(angle) <= half_angle_deg:
            out.append(i)
    return out


def front_min(sectors, half_angle_deg=FRONT_HALF_ANGLE_DEG):
    """Nearest normalised reading inside the frontal cone (1.0 = free)."""
    n = len(sectors)
    idx = _front_indices(n, half_angle_deg)
    if not idx:
        return 1.0
    return float(min(float(sectors[i]) for i in idx))


def front_blocked(sectors, threshold, half_angle_deg=FRONT_HALF_ANGLE_DEG):
    """True when something sits closer than ``threshold`` straight ahead.

    This is the emergency stop, and deliberately a *safety* item rather than a
    learned behaviour: the model's brake head is inert, so stopping cannot be
    left to it. Only the frontal cone counts -- a wall passing by on the side is
    normal on a 0.53 m wide track and must not halt the car.
    """
    return bool(front_min(sectors, half_angle_deg) < float(threshold))


# Fim de pista (2026-10-09). Uma pista aberta termina numa parede branca, igual
# as laterais -- para a camera nao ha nada novo. Quem reconhece o fim e o LiDAR,
# pela geometria: NENHUMA direcao livre a frente. Numa curva sempre sobra uma (a
# saida); num beco, nao.
#
# Usa a MAIOR leitura do cone, nao a menor: um setor sem retorno le 1.0 ("livre")
# e impede o disparo -- a falha cai do lado seguro, e a parada de emergencia
# continua valendo. Medido nas rodadas de 2026-10-07 (cone +/-60, limiar
# 0.50 m): zero disparos em 7874 quadros do oval, onde a menor "maior leitura"
# foi 0.67 m; no fim do pista2, disparo 0.4-0.5 s antes da parada de emergencia.
# A versao "frente E os dois lados perto" tambem zerava o oval, mas com a menor
# leitura: numa reta cruzada na diagonal (o "S") as tres janelas podem ficar
# perto ao mesmo tempo sem que o caminho acabe.
DEAD_END_HALF_ANGLE_DEG = 60.0


def front_max(sectors, half_angle_deg=DEAD_END_HALF_ANGLE_DEG):
    """Farthest normalised reading inside the cone (1.0 = some way out is free)."""
    idx = _front_indices(len(sectors), half_angle_deg)
    if not idx:
        return 1.0
    return float(max(float(sectors[i]) for i in idx))


def dead_end(sectors, threshold, half_angle_deg=DEAD_END_HALF_ANGLE_DEG):
    """True when every direction inside the cone is closed nearer than ``threshold``.

    Sectors outside the model's FOV read 1.0, so a FOV narrower than the cone
    can never trip it -- deliberately: the rule was measured with FOV 180.
    """
    return bool(front_max(sectors, half_angle_deg) < float(threshold))
