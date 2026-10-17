"""Escrever no PCA9685 aguentando um fio que solta por um instante. Python 3.6-safe.

Visto em 2026-10-14 no controle manual (hardware/controle_8bitdo.py): um jumper
do I2C soltou por um instante, uma escrita falhou com Errno 121 (Remote I/O
error) e o programa caiu. A escrita seguinte -- a do estado seguro -- passou:
o contato ja tinha voltado.

Cair nao deixa o carro mais seguro. O PCA9685 segue mandando o ultimo pulso
sozinho, e o "parar" sai pelo mesmo fio que acabou de falhar. Continuar
tentando a cada quadro faz o comando ATUAL (inclusive soltar o RT ou apertar
Back) chegar no instante em que o contato voltar. So quando as falhas duram
``limite_s`` seguidos a placa e dada como perdida, e ai quem chama tenta o
neutro com insistencia e manda desligar a chave.

O conserto de verdade e o fio (docs/CONTROLE_8BITDO.md). Isto so impede que um
mau contato de milissegundos derrube o programa.

Sem Jetson: quem escreve na placa e o relogio sao injetados.
"""
import time


class PCA9685Perdido(Exception):
    """As escritas no I2C falham ha ``limite_s`` seguidos."""


class EscritaTolerante(object):
    """Manda servo e ESC; uma falha isolada de I2C nao derruba ninguem.

    Args:
        escreve: funcao ``(servo_us, esc_us)`` que escreve na placa e levanta
            ``IOError`` (o ``OSError`` do smbus) quando o I2C falha.
        limite_s: por quanto tempo seguido as falhas sao toleradas.
        relogio: ``time.monotonic``; existe para os testes.
    """

    def __init__(self, escreve, limite_s=0.1, relogio=time.monotonic):
        self._escreve = escreve
        self._limite_s = limite_s
        self._relogio = relogio
        self._falhando_desde = None
        #: Total de escritas que falharam, para mostrar na tela.
        self.falhas = 0

    def manda(self, servo_us, esc_us):
        """``True`` se a escrita passou, ``False`` se falhou e ainda da para esperar.

        Raises:
            PCA9685Perdido: as falhas ja duram ``limite_s``.
        """
        try:
            self._escreve(servo_us, esc_us)
        except IOError as erro:
            self.falhas += 1
            agora = self._relogio()
            if self._falhando_desde is None:
                self._falhando_desde = agora
            if agora - self._falhando_desde >= self._limite_s:
                raise PCA9685Perdido(
                    "o PCA9685 nao responde no I2C ha {0:.2f} s ({1})".format(
                        agora - self._falhando_desde, erro))
            return False
        self._falhando_desde = None
        return True


def para_com_insistencia(escreve, servo_us, esc_us, tentativas=10, pausa_s=0.05,
                         dorme=time.sleep):
    """Tenta mandar o estado seguro ate uma escrita passar.

    Returns:
        ``True`` se passou; ``False`` se todas falharam -- o ESC pode ter ficado
        no ultimo pulso, e quem chama precisa mandar desligar a chave do carro.
    """
    for _ in range(int(tentativas)):
        try:
            escreve(servo_us, esc_us)
            return True
        except IOError:
            dorme(pausa_s)
    return False
