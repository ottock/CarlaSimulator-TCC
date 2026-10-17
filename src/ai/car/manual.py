"""Dirigir o carro com o controle 8BitDo: estado do controle -> servo e ESC.

Python 3.6-safe. O equivalente do teclado e o ``hardware/controle_teste.py``;
aqui o volante e o acelerador sao proporcionais.

  - analogico esquerdo, para os lados: volante. Mesma conversao do runtime
    (``steer_to_us``), entao o sentido e o batente sao os que o modelo usa;
  - RT: acelerador, de 1600 us (o minimo que move o ESC) ate ``max_us``
    (1700 us, o maximo calibrado). Decisao do Rafael em 2026-10-14: "qualquer
    coisa a gente diminui depois" -- por isso o teto e um parametro;
  - LT: re, a 1400 us fixos, como a seta para baixo do teclado;
  - Back ou Home: parada. O carro so volta a andar depois de os dois gatilhos
    serem soltos. O Home entra porque e apertado antes de o controle desligar.

E a regra que o teclado nao precisava: o adaptador so manda relatorio quando
algo MUDA, e quando o controle desliga ele nao avisa nem zera (medido em
2026-10-13, docs/CONTROLE_8BITDO.md). Um RT "preso" seguiria acelerando ate
bater. Por isso, com o carro andando, ficar ``silencio_s`` sem relatorio novo
para o carro. O preco: segurar o RT no fundo com o analogico parado tambem e
silencio, e pode parar o carro numa reta. Se isso acontecer demais, o
``hardware/pad_silencio.py`` mede quanto tempo o controle fica mudo de verdade.

Sem Jetson: nada aqui toca em hardware, e o relogio e injetado.
"""
import collections
import time

from ai.car.control_map import (ESC_MAX_US, ESC_MIN_MOVE_US, ESC_NEUTRAL_US,
                                STEER_CENTER_US, STEER_SPAN_US, steer_to_us)
from ai.car.gamepad import BACK, HOME

# Re do controle_teste.py, a que ja foi usada no carro pelo teclado.
ESC_RE_US = 1400

# Abaixo de 10% do curso o gatilho conta como solto: um dedo apoiado nao pode
# fazer o carro sair andando.
ZONA_MORTA_GATILHO = 0.10

# Sentido do analogico. Pelo padrao do Xbox 360 direita e positivo, igual a
# convencao do steer (+1 = direita). A CONFERIR no carro: se as rodas forem
# para o lado contrario do analogico, e aqui que se troca, e so aqui.
SINAL_DO_ANALOGICO = 1

#: O que mandar aos atuadores agora. ``motivo`` diz por que o carro esta
#: parado quando ele deveria andar: ``"parado"`` (Back/Home) ou
#: ``"silencio"`` (controle mudo); ``None`` no resto.
Comando = collections.namedtuple("Comando", "servo_us esc_us motivo")


class Manual(object):
    """Guarda o ultimo estado do controle e decide servo e ESC.

    Args:
        max_us: teto do acelerador, entre 1600 e 1700 us.
        span_us: meia-faixa do servo; o padrao e o do modelo (``STEER_SPAN_US``).
        silencio_s: quanto tempo sem relatorio para o carro, se estiver andando.
        relogio: ``time.monotonic``; existe para os testes.
    """

    def __init__(self, max_us=ESC_MAX_US, span_us=STEER_SPAN_US, silencio_s=0.5,
                 relogio=time.monotonic):
        if max_us < ESC_MIN_MOVE_US:
            raise ValueError(
                "max_us {0} esta abaixo de {1}, o minimo que move o ESC: o carro "
                "nunca andaria".format(max_us, ESC_MIN_MOVE_US))
        self._max_us = min(int(max_us), ESC_MAX_US)
        self._span_us = span_us
        self._silencio_s = silencio_s
        self._relogio = relogio
        self._estado = None
        self._visto_em = None
        self._travado = False

    def recebe(self, estado):
        """Um relatorio novo, ja decodificado (``gamepad.estado``)."""
        if estado is None:
            return
        if estado.botoes & (BACK | HOME):
            self._travado = True
        elif self._travado and not (_apertado(estado.rt) or _apertado(estado.lt)):
            self._travado = False
        self._estado = estado
        self._visto_em = self._relogio()

    def trava(self):
        """Parado ate os dois gatilhos serem soltos, como depois de um Back."""
        self._travado = True

    def comando(self):
        """Servo e ESC para agora. Chamar a cada giro do laco, haja relatorio ou nao."""
        e = self._estado
        if e is None:
            return Comando(STEER_CENTER_US, ESC_NEUTRAL_US, None)
        servo = steer_to_us(SINAL_DO_ANALOGICO * e.ex / 32767.0, span_us=self._span_us)
        frente, re = _apertado(e.rt), _apertado(e.lt)
        if not (frente or re):
            return Comando(servo, ESC_NEUTRAL_US, None)
        if self._travado:
            return Comando(servo, ESC_NEUTRAL_US, "parado")
        if self._relogio() - self._visto_em > self._silencio_s:
            return Comando(servo, ESC_NEUTRAL_US, "silencio")
        if frente and re:
            return Comando(servo, ESC_NEUTRAL_US, None)
        if re:
            return Comando(servo, ESC_RE_US, None)
        return Comando(servo, self._acelerador_us(e.rt), None)

    def _acelerador_us(self, rt):
        fracao = (rt / 255.0 - ZONA_MORTA_GATILHO) / (1.0 - ZONA_MORTA_GATILHO)
        us = ESC_MIN_MOVE_US + fracao * (self._max_us - ESC_MIN_MOVE_US)
        return int(round(max(ESC_MIN_MOVE_US, min(self._max_us, us))))


def _apertado(gatilho):
    return gatilho / 255.0 >= ZONA_MORTA_GATILHO
