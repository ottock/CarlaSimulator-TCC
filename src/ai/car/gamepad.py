"""Controle 8BitDo Ultimate 2C lido direto da USB. Python 3.6-safe.

Por que direto da USB. O kernel 4.9 do Jetson nao reconhece o adaptador de 2.4G
(lsusb 2dc8:310a), e forcar o driver xpad cria o /dev/input/js0 mas nenhum
evento chega (2026-10-13). O que funcionou foi falar com o adaptador pelo
pyusb, depois de mandar a mensagem de inicializacao E o comando de LED do Xbox
360. A historia inteira esta em docs/CONTROLE_8BITDO.md.

Duas coisas que quem usa este modulo precisa saber:
  - o adaptador so manda relatorio quando algo MUDA. ``Controle.le`` devolver
    ``None`` e o normal com a mao parada;
  - quando o controle desliga, o adaptador nao avisa nem zera: o ultimo estado
    fica valendo. Por isso quem age em cima dos botoes deve agir na BORDA (o
    instante em que apertou), nunca no nivel.

Sem Jetson: ``botoes`` so decodifica bytes e ``Controle`` recebe o dispositivo
pronto, entao os dois sao testaveis sem pyusb.
"""
import errno

ID_FABRICANTE = 0x2DC8
ID_PRODUTO = 0x310A

_INTERFACE = 0
_ENTRADA = 0x84
_SAIDA = 0x05
_TAMANHO_LEITURA = 64

# Mascara de botoes: primeiro byte de botoes do relatorio nos bits 0-7, segundo
# nos bits 8-15. Posicoes do padrao de controle de Xbox 360; so o Home foi
# visto de fato no Jetson.
START = 1 << 4
BACK = 1 << 5
HOME = 1 << 10


class ControlePerdido(Exception):
    """O adaptador sumiu ou a USB deu erro: nao ha mais como ler botoes."""


def botoes(relatorio):
    """Mascara dos botoes apertados, ou ``None`` se o relatorio nao e de botoes.

    Devolver ``None`` em vez de chutar e deliberado: um relatorio cortado lido
    como "nada apertado" esconderia um Back.
    """
    if relatorio is None or len(relatorio) < 4 or relatorio[0] != 0x00:
        return None
    return relatorio[2] | (relatorio[3] << 8)


def acha_adaptador():
    """Dispositivo do pyusb para o adaptador, ou ``ControlePerdido``.

    O import fica aqui dentro porque o pyusb so existe no Jetson
    (``sudo apt install python3-usb``).
    """
    import usb.core

    dispositivo = usb.core.find(idVendor=ID_FABRICANTE, idProduct=ID_PRODUTO)
    if dispositivo is None:
        raise ControlePerdido(
            "adaptador 2dc8:310a nao encontrado (esta plugado no Jetson?)")
    return dispositivo


class Controle(object):
    """O adaptador aberto e pronto para leitura.

    ``dispositivo`` e o objeto do pyusb (``acha_adaptador()``); nos testes, um
    substituto com os mesmos metodos.
    """

    def __init__(self, dispositivo):
        self._usb = dispositivo

    def abre(self):
        """Manda o que faz o adaptador comecar a enviar relatorios."""
        # So ha driver preso se alguem registrou o adaptador no xpad neste
        # boot; soltar o que nao esta preso da erro no pyusb.
        if self._usb.is_kernel_driver_active(_INTERFACE):
            self._usb.detach_kernel_driver(_INTERFACE)
        # Mensagem que os kernels a partir de 2023 mandam a controles de
        # terceiros, e o comando de LED do Xbox 360. Nao isolamos qual dos dois
        # destrava o envio; os dois juntos, nesta ordem, funcionaram.
        self._usb.ctrl_transfer(0xC1, 0x01, 0x0100, 0x00, 20)
        self._usb.write(_SAIDA, bytes([0x01, 0x03, 0x02]), 1000)

    def le(self, timeout_ms):
        """Proximo relatorio em ``bytes``, ou ``None`` se nada mudou no prazo.

        Raises:
            ControlePerdido: qualquer erro de USB que nao seja o prazo vencer.
        """
        try:
            dados = self._usb.read(_ENTRADA, _TAMANHO_LEITURA, timeout_ms)
        except IOError as erro:   # usb.core.USBError e um IOError
            if erro.errno == errno.ETIMEDOUT or "timed out" in str(erro).lower():
                return None
            raise ControlePerdido(str(erro))
        return bytes(bytearray(dados))
