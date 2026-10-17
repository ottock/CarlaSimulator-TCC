"""Largar e parar a corrida pelo controle: Start larga, Back para. Python 3.6-safe.

"Corrida" e o ``hardware/jetson_runtime.py`` rodando o modelo, como um processo
a parte. Este modulo nao dirige nada: decide QUANDO lancar esse processo e
quando interrompe-lo, a partir dos botoes do controle (``ai.car.gamepad``).

Parar e mandar ao runtime o mesmo sinal do Ctrl+C. Ele ja trata isso: sai do
laco e, no ``finally``, centraliza o servo e poe o ESC em neutro.

As tres regras que nao sao obvias, e por que:

  1. BORDA, nao nivel. So conta o instante em que o botao foi apertado. Quando
     o controle desliga, o adaptador nao zera o ultimo estado; agindo no nivel,
     um Start "preso" largaria uma corrida atras da outra.
  2. NUNCA dois sinais seguidos. Um segundo Ctrl+C pode cair dentro da limpeza
     do runtime e interrompe-la antes de o ESC ir a neutro. Depois de um sinal,
     outro so sai passada a ``CARENCIA_S``.
  3. NUNCA matar o runtime. Matar pula a limpeza, e o PCA9685 continua mandando
     o ultimo pulso sozinho: o carro seguiria andando. Se o runtime nao
     responde, o que resta e avisar e desligar a chave do carro.

O laco que le o controle e divide a placa com o runtime fica em
``ai.car.controle_unico``, usado pelo ``hardware/controle_8bitdo.py``.

Sem Jetson: ``Largada`` recebe quem lanca o processo, entao e testavel com
substitutos.
"""
import os
import signal
import subprocess
import time

from ai.car.gamepad import BACK, START

# Tempo depois de um sinal em que outro nao e mandado. A limpeza do runtime
# (ESC em neutro, fechar log, camera e LiDAR) leva bem menos que isso.
CARENCIA_S = 3.0


class Largada(object):
    """Start lanca uma corrida, Back a interrompe.

    Args:
        inicia: funcao sem argumentos que lanca o runtime e devolve o processo
            (``subprocess.Popen``).
        avisa: chamada com o nome de cada mudanca -- ``"largou"``,
            ``"parando"``, ``"acabou"``, ``"nao_parou"``.
        relogio: ``time.monotonic``; existe para os testes.
    """

    def __init__(self, inicia, avisa=None, relogio=time.monotonic):
        self._inicia = inicia
        self._avisa = avisa or (lambda evento: None)
        self._relogio = relogio
        self._corrida = None
        self._antes = 0
        self._sinal_em = None

    @property
    def correndo(self):
        """Ha uma corrida lancada que ainda nao terminou (visto no ultimo ``passo``)."""
        return self._corrida is not None

    def passo(self, botoes=None):
        """Um giro do laco. ``botoes`` e a mascara nova, ou ``None`` se o
        controle nao mandou nada desde o ultimo giro."""
        if self._corrida is not None and self._corrida.poll() is not None:
            self._corrida = None
            self._sinal_em = None
            self._avisa("acabou")
        if botoes is None:
            return
        apertou = botoes & ~self._antes
        self._antes = botoes
        if apertou & BACK:
            # Back ganha de Start: na duvida, o carro nao sai.
            if self._corrida is not None and self._pode_sinalizar():
                self._sinaliza()
                self._avisa("parando")
        elif apertou & START and self._corrida is None:
            self._corrida = self._inicia()
            self._avisa("largou")

    def encerra(self, sinaliza=True, espera_s=10.0):
        """Para a corrida em andamento, se houver, e espera ela sair.

        ``sinaliza=False`` e para o Ctrl+C no terminal: ele ja chegou ao
        runtime (mesmo grupo de processos), e repetir o sinal poderia cortar a
        limpeza dele. So manda se a corrida nao sair sozinha em ``espera_s``.

        Returns:
            ``True`` se nao sobrou corrida viva.
        """
        if self._corrida is None:
            return True
        if sinaliza and self._pode_sinalizar():
            self._sinaliza()
        if self._esperou_sair(espera_s):
            return True
        self._sinaliza()
        if self._esperou_sair(espera_s):
            return True
        self._avisa("nao_parou")
        return False

    def _pode_sinalizar(self):
        return (self._sinal_em is None
                or self._relogio() - self._sinal_em >= CARENCIA_S)

    def _sinaliza(self):
        self._corrida.send_signal(signal.SIGINT)
        self._sinal_em = self._relogio()

    def _esperou_sair(self, espera_s):
        try:
            self._corrida.wait(timeout=espera_s)
        except subprocess.TimeoutExpired:
            return False
        self._corrida = None
        self._sinal_em = None
        self._avisa("acabou")
        return True


def comando_da_corrida(python, runtime, engine, config, out, cruise_us,
                       beco_dist, extras=()):
    """A linha de comando do runtime, igual a digitada a mao (ESTADO_IA.md §2).

    ``extras`` vai no fim, sem mexer: e por onde entra ``--seconds 120`` ou
    qualquer outro ajuste do runtime.
    """
    return [python, runtime,
            "--engine", engine, "--config", config, "--out", out,
            "--cruise-us", str(cruise_us), "--beco-dist", str(beco_dist)
            ] + list(extras)


def pasta_da_corrida(engine, quando):
    """``runs/<modelo>_<data>_<hora>``: uma pasta de log por corrida."""
    modelo = os.path.splitext(os.path.basename(engine))[0]
    return "runs/{0}_{1}".format(modelo, quando.strftime("%Y-%m-%d_%H%M%S"))

