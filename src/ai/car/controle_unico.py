"""Um programa so para o controle: dirige a mao e larga a corrida autonoma.

Python 3.6-safe. Junta o que eram dois programas (pedido do Rafael em
2026-10-17): o controle manual (``ai.car.manual``) e a largada da corrida
autonoma (``ai.car.largada``).

A regra que manda em tudo: o PCA9685 tem UM dono por vez.

  - Sem corrida, o dono e o controle manual: ``passo`` devolve o ``Comando``
    a escrever.
  - Start: o carro e estacionado (servo no centro, ESC em neutro) e so entao
    o runtime e lancado. A partir dai ``passo`` devolve ``None`` -- nada vai
    para a placa, que passa a ser do runtime. Dois processos escrevendo no
    mesmo servo e ESC fariam o carro obedecer a quem escreveu por ultimo.
  - Back com corrida: para a corrida (o mesmo sinal do Ctrl+C). Sem corrida,
    e a parada do controle manual.
  - Quando a corrida acaba, o manual volta TRAVADO: parado ate os gatilhos
    serem soltos. Um RT que ficou apertado durante a corrida nao arranca o carro.

Sem Jetson: quem lanca o runtime, quem estaciona o carro e o relogio sao
injetados.
"""
import time

from ai.car.largada import Largada


class ControleUnico(object):
    """Decide, a cada giro do laco, quem manda no carro e o que mandar.

    Args:
        inicia_runtime: funcao sem argumentos que lanca o jetson_runtime.py e
            devolve o processo.
        manual: um ``ai.car.manual.Manual``.
        estaciona: funcao sem argumentos que poe servo no centro e ESC em
            neutro na placa; devolve ``True`` se conseguiu.
        avisa: recebe os eventos da corrida (``"largou"``, ``"parando"``,
            ``"acabou"``, ``"nao_parou"``).
        relogio: ``time.monotonic``; existe para os testes.
    """

    def __init__(self, inicia_runtime, manual, estaciona, avisa=None,
                 relogio=time.monotonic):
        self._inicia_runtime = inicia_runtime
        self._manual = manual
        self._estaciona = estaciona
        self._avisa = avisa or (lambda evento: None)
        self._largada = Largada(self._lanca, avisa=self._evento, relogio=relogio)

    @property
    def correndo(self):
        """A placa esta com o runtime."""
        return self._largada.correndo

    def passo(self, estado):
        """Um giro do laco. ``estado`` e o relatorio decodificado, ou ``None``.

        Returns:
            O ``Comando`` para escrever na placa, ou ``None`` se ela e do runtime.
        """
        self._largada.passo(None if estado is None else estado.botoes)
        if self._largada.correndo:
            return None
        self._manual.recebe(estado)
        return self._manual.comando()

    def encerra(self, sinaliza=True):
        """Para a corrida, se houver, e SO DEPOIS estaciona o carro.

        Mesmo com um runtime que nao respondeu, o carro e estacionado: travado,
        ele ja nao escreve nada, e o neutro daqui e o que resta.

        Returns:
            ``True`` se nao sobrou corrida viva e o carro foi estacionado.
        """
        sem_corrida = self._largada.encerra(sinaliza=sinaliza)
        estacionou = self._estaciona()
        return sem_corrida and estacionou

    def _lanca(self):
        self._estaciona()
        return self._inicia_runtime()

    def _evento(self, evento):
        if evento == "acabou":
            self._manual.trava()
        self._avisa(evento)
