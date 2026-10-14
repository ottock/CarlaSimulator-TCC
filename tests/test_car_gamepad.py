"""O controle 8BitDo lido direto da USB (docs/CONTROLE_8BITDO.md).

REPOUSO e ANALOGICO_NO_FIM sao relatorios REAIS, copiados da saida do adaptador
no Jetson em 2026-10-13. O bit do Home tambem foi visto la. As posicoes de Start
e Back vem do padrao de controle de Xbox 360 e ainda NAO foram conferidas
apertando o botao no carro.
"""
import errno
from array import array

import pytest

from ai.car.gamepad import (BACK, HOME, START, Controle, ControlePerdido, Estado,
                            botoes, estado)

REPOUSO = bytes.fromhex("0014000000000000000000000000101c30640010")
ANALOGICO_NO_FIM = bytes.fromhex("001400000000ff7f030400000000101c30640010")


def _relatorio(byte2=0, byte3=0):
    return bytes([0x00, 0x14, byte2, byte3]) + REPOUSO[4:]


def test_a_report_at_rest_has_no_button_pressed():
    assert botoes(REPOUSO) == 0


def test_moving_a_stick_does_not_press_any_button():
    # Os bytes do analogico (ff 7f) ficam depois dos botoes: ler a posicao
    # errada faria o analogico no fim de curso virar "tudo apertado".
    assert botoes(ANALOGICO_NO_FIM) == 0


def test_start_is_bit_4_of_the_first_button_byte():
    b = botoes(_relatorio(byte2=0x10))
    assert b & START and not b & BACK


def test_back_is_bit_5_of_the_first_button_byte():
    b = botoes(_relatorio(byte2=0x20))
    assert b & BACK and not b & START


def test_home_is_bit_2_of_the_second_button_byte():
    b = botoes(_relatorio(byte3=0x04))
    assert b & HOME and not b & (START | BACK)


def test_buttons_of_both_bytes_come_in_one_mask():
    # Start + Back + Home juntos: o segundo byte entra acima do primeiro.
    assert botoes(_relatorio(byte2=0x30, byte3=0x04)) == 0x0430


def test_a_truncated_report_is_not_decoded():
    # Chutar "nada apertado" esconderia um Back; chutar o contrario largaria.
    assert botoes(REPOUSO[:3]) is None


def test_a_message_of_another_type_is_not_decoded():
    assert botoes(b"\x08" + REPOUSO[1:]) is None


# ------------------------------------------------- analogicos e gatilhos

def test_a_report_at_rest_has_sticks_and_triggers_at_zero():
    assert estado(REPOUSO) == Estado(botoes=0, lt=0, rt=0, ex=0, ey=0, dx=0, dy=0)


def test_the_left_stick_at_the_end_of_its_travel():
    # Bytes reais: "ff 7f" = +32767 no lado, "03 04" = 1027 na vertical.
    e = estado(ANALOGICO_NO_FIM)
    assert (e.ex, e.ey) == (32767, 1027)


def test_stick_values_are_signed():
    # Linha real do meio da varredura: "02 1b" = 6914 e "e1 d8" = -10015.
    e = estado(bytes.fromhex("001400000000021be1d800000000101c30640010"))
    assert (e.ex, e.ey) == (6914, -10015)


def test_the_triggers_come_in_bytes_4_and_5():
    e = estado(bytes([0x00, 0x14, 0x00, 0x00, 0x80, 0xFF]) + REPOUSO[6:])
    assert (e.lt, e.rt) == (128, 255)


def test_the_right_stick_comes_after_the_left_one():
    r = REPOUSO[:10] + bytes([0x00, 0x80, 0xFF, 0x7F]) + REPOUSO[14:]
    e = estado(r)
    assert (e.dx, e.dy, e.ex, e.ey) == (-32768, 32767, 0, 0)


def test_a_report_too_short_for_the_sticks_is_not_decoded():
    # Ler analogico de um relatorio cortado inventaria um volante.
    assert estado(REPOUSO[:13]) is None


def test_a_message_of_another_type_has_no_sticks():
    assert estado(b"\x08" + REPOUSO[1:]) is None


class Adaptador(object):
    """Faz as vezes do dispositivo do pyusb: guarda o que foi pedido a ele."""

    def __init__(self, driver_do_kernel=False, leitura=None):
        self.pedidos = []
        self._driver = driver_do_kernel
        self._leitura = leitura

    def is_kernel_driver_active(self, interface):
        return self._driver

    def detach_kernel_driver(self, interface):
        self.pedidos.append(("solta", interface))

    def ctrl_transfer(self, tipo, pedido, valor, indice, tamanho):
        self.pedidos.append(("controle", tipo, pedido, valor, indice, tamanho))
        return array("B", [0] * tamanho)

    def write(self, endpoint, dados, timeout=None):
        self.pedidos.append(("escreve", endpoint, bytes(dados)))

    def read(self, endpoint, tamanho, timeout=None):
        self.pedidos.append(("le", endpoint, timeout))
        if isinstance(self._leitura, Exception):
            raise self._leitura
        return self._leitura


def test_opening_sends_the_init_message_and_then_the_led_command():
    # Foi esta sequencia que fez o adaptador comecar a mandar dados no Jetson.
    usb = Adaptador()
    Controle(usb).abre()
    assert usb.pedidos == [
        ("controle", 0xC1, 0x01, 0x0100, 0x00, 20),
        ("escreve", 0x05, b"\x01\x03\x02"),
    ]


def test_opening_releases_the_interface_when_the_kernel_holds_it():
    # Acontece se alguem registrou o adaptador no xpad (new_id) nesse boot.
    usb = Adaptador(driver_do_kernel=True)
    Controle(usb).abre()
    assert usb.pedidos[0] == ("solta", 0)


def test_opening_does_not_release_an_interface_nobody_holds():
    # Soltar o que nao esta preso da erro no pyusb.
    usb = Adaptador(driver_do_kernel=False)
    Controle(usb).abre()
    assert ("solta", 0) not in usb.pedidos


def test_reading_returns_the_report_as_bytes():
    usb = Adaptador(leitura=array("B", REPOUSO))
    assert Controle(usb).le(100) == REPOUSO
    assert usb.pedidos == [("le", 0x84, 100)]


def test_a_timeout_means_nothing_changed():
    # O adaptador so manda relatorio quando algo muda: silencio e o normal.
    usb = Adaptador(leitura=OSError(errno.ETIMEDOUT, "Operation timed out"))
    assert Controle(usb).le(100) is None


def test_a_timeout_without_errno_is_still_a_timeout():
    usb = Adaptador(leitura=OSError("Operation timed out"))
    assert Controle(usb).le(100) is None


def test_an_unplugged_adapter_is_reported_as_lost():
    # Engolir este erro deixaria o programa rodando com o botao de parar morto.
    usb = Adaptador(leitura=OSError(errno.ENODEV, "No such device"))
    with pytest.raises(ControlePerdido):
        Controle(usb).le(100)
