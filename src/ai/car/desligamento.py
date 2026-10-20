"""Segurar LB + RB para desligar o Jetson. Python 3.6-safe.

Por que existe. Na feira o carro nao tem monitor nem teclado, e tirar a bateria
com o Jetson ligado ja corrompeu o sistema de arquivos uma vez (o computador do
carro nao ligou mais; foi consertado a mao). O Rafael escolheu em 2026-10-20
desligar pelo controle: LB + RB juntos por 3 s. Os dois nao faziam nada, entao
nao ha como largar uma corrida sem querer.

Diferente do Start e do Back, aqui vale o NIVEL (segurar), nao a borda: segurar
parado e silencio no adaptador (``None``), e o ultimo estado continua valendo.
O preco: se o controle desligar no meio dos 3 s com os dois apertados, o
estado fica preso e o Jetson desliga -- que era o que a pessoa estava pedindo.

So conta fora de corrida, e o aperto tem que COMECAR fora dela: segurar os dois
durante a corrida e continuar depois nao desliga ate soltar e apertar de novo.
"""
import time

from ai.car.gamepad import LB, RB

SEGURAR_S = 3.0
_OS_DOIS = LB | RB


class SegurarParaDesligar(object):
    """Diz quando LB + RB ficaram apertados juntos ``segurar_s`` segundos.

    Args:
        segurar_s: quanto tempo segurar.
        relogio: ``time.monotonic``; existe para os testes.
    """

    def __init__(self, segurar_s=SEGURAR_S, relogio=time.monotonic):
        self._segurar_s = segurar_s
        self._relogio = relogio
        self._botoes = 0
        self._desde = None
        # Precisa soltar antes de valer de novo: depois de disparar, ou se o
        # aperto comecou numa corrida.
        self._travado = False

    def passo(self, estado, correndo):
        """Um giro do laco. ``estado`` e o relatorio decodificado ou ``None``.

        Returns:
            ``True`` uma vez por aperto, quando completa ``segurar_s``.
        """
        if estado is not None:
            self._botoes = estado.botoes
        if self._botoes & _OS_DOIS != _OS_DOIS:
            self._desde, self._travado = None, False
            return False
        if correndo or self._travado:
            self._desde, self._travado = None, True
            return False
        agora = self._relogio()
        if self._desde is None:
            self._desde = agora
        if agora - self._desde < self._segurar_s:
            return False
        self._desde, self._travado = None, True
        return True
