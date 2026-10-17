"""Um programa so para o controle: dirige a mao e roda a corrida autonoma.

Python 3.6-safe. Usado pelo ``hardware/controle_8bitdo.py``.

Historia. Ate 2026-10-17 o Start lancava o ``jetson_runtime.py`` como OUTRO
programa: cada corrida carregava CUDA, camera, LiDAR e modelo do zero (1 a 2
minutos, nunca cronometrado), e depois do Back o manual so voltava quando
aquele programa terminava de fechar tudo. A pedido do Rafael ("o objetivo e ser
algo mais rapido"), a corrida passou a rodar DENTRO do programa do controle:
modelo, camera e LiDAR sao abertos uma vez, na abertura do programa, e o Start
so comeca a chamar o laco de direcao do runtime (o mesmo ``DriveLoop``).

A regra que manda em tudo: quem escreve na placa e UM so.

  - Sem corrida, o controle manual: ``passo`` devolve o ``Comando`` a escrever.
  - Start: o carro e estacionado (servo no centro, ESC em neutro) e uma
    corrida nova comeca. Daí em diante cada ``passo`` e um quadro do modelo, e
    devolve ``None`` -- quem escreveu na placa foi o laco do modelo.
  - Back, o limite de tempo ou um erro no laco: o carro e estacionado na hora,
    o log da corrida e fechado, e o manual volta TRAVADO -- parado ate os
    gatilhos serem soltos.

Os botoes valem na BORDA (o instante em que foram apertados): quando o controle
desliga, o adaptador nao zera o ultimo estado, e um Start "preso" largaria uma
corrida atras da outra.

Sem Jetson: quem cria a corrida, quem estaciona o carro e o relogio sao
injetados.
"""
import time
import traceback

from ai.car.gamepad import BACK, START


class ControleUnico(object):
    """Decide, a cada giro do laco, quem manda no carro.

    Args:
        nova_corrida: funcao sem argumentos que prepara uma corrida e devolve
            um objeto com ``passo()`` (um quadro do modelo; devolve a
            telemetria) e ``fecha()`` (fecha o log).
        manual: um ``ai.car.manual.Manual``.
        estaciona: funcao sem argumentos que poe servo no centro e ESC em
            neutro; devolve ``True`` se conseguiu.
        avisa: recebe ``"largou"`` e, quando a corrida acaba, o motivo:
            ``"parada"`` (Back), ``"tempo"`` ou ``"erro"``.
        relogio: ``time.monotonic``; existe para os testes.
        limite_s: duracao maxima de uma corrida (o ``--seconds`` do runtime).
    """

    def __init__(self, nova_corrida, manual, estaciona, avisa=None,
                 relogio=time.monotonic, limite_s=60.0):
        self._nova_corrida = nova_corrida
        self._manual = manual
        self._estaciona = estaciona
        self._avisa = avisa or (lambda evento: None)
        self._relogio = relogio
        self._limite_s = limite_s
        self._corrida = None
        self._inicio = None
        self._antes = 0
        #: Telemetria do ultimo quadro do modelo, para a tela.
        self.telemetria = None
        #: Texto do ultimo erro que encerrou uma corrida.
        self.erro = None

    @property
    def correndo(self):
        return self._corrida is not None

    def passo(self, estado):
        """Um giro do laco. ``estado`` e o relatorio decodificado, ou ``None``.

        Returns:
            O ``Comando`` para escrever na placa, ou ``None`` quando quem
            mandou neste giro foi o modelo (ou a corrida acabou de parar).
        """
        apertou = 0
        if estado is not None:
            apertou = estado.botoes & ~self._antes
            self._antes = estado.botoes

        if self._corrida is not None:
            if apertou & BACK:
                self._termina("parada")
            elif self._relogio() - self._inicio >= self._limite_s:
                self._termina("tempo")
            else:
                self._quadro()
            return None

        if apertou & START and not apertou & BACK:
            self._estaciona()
            self._corrida = self._nova_corrida()
            self._inicio = self._relogio()
            self.erro = None
            self._avisa("largou")
            self._quadro()
            return None

        self._manual.recebe(estado)
        return self._manual.comando()

    def encerra(self):
        """Para a corrida, se houver, e estaciona o carro.

        Returns:
            ``True`` se o carro foi estacionado.
        """
        if self._corrida is not None:
            return self._termina("parada")
        return self._estaciona()

    def _quadro(self):
        try:
            self.telemetria = self._corrida.passo()
        except Exception:
            # Erro no laco do modelo (camera, I2C, ...): o programa nao cai, a
            # corrida para e o carro e estacionado. Ctrl+C nao e Exception e
            # passa direto.
            self.erro = traceback.format_exc()
            self._termina("erro")

    def _termina(self, motivo):
        estacionou = self._estaciona()
        corrida, self._corrida = self._corrida, None
        try:
            corrida.fecha()
        finally:
            self._manual.trava()
            self._avisa(motivo)
        return estacionou
