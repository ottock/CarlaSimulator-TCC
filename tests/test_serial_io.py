"""A leitura do LiDAR esvazia a fila inteira, nao so o primeiro lote.

O buffer de leitura do Linux entrega no maximo ~4 KB por vez. A leitura antiga
pegava um lote por quadro e deixava o resto para depois -- e o modelo passava
segundos recebendo LiDAR velho (runs/Diag/lidar_x4_2026-10-07_2).
"""
from ai.car.serial_io import ler_fila


class SerialComFila(object):
    """Imita a serial do Linux: uma fila grande, entregue em lotes de 4 KB."""

    LOTE = 4096

    def __init__(self, n_bytes):
        self.fila = bytearray(i % 251 for i in range(n_bytes))
        self.leituras = 0

    @property
    def in_waiting(self):
        return min(len(self.fila), self.LOTE)

    def read(self, n):
        self.leituras += 1
        pedaco, self.fila = bytes(self.fila[:n]), self.fila[n:]
        return pedaco


def test_a_backlog_bigger_than_one_batch_is_read_in_full():
    # 40 KB = ~36 scans de atraso. A leitura antiga levaria 10 quadros.
    ser = SerialComFila(40000)
    dados = ler_fila(ser)
    assert len(dados) == 40000
    assert ser.fila == bytearray()
    assert ser.leituras == 10


def test_the_bytes_come_out_in_order():
    ser = SerialComFila(10000)
    assert ler_fila(ser) == bytes(i % 251 for i in range(10000))


def test_nothing_waiting_returns_empty():
    assert ler_fila(SerialComFila(0)) == b""


def test_a_serial_that_never_empties_does_not_hang_the_loop():
    class Infinita(object):
        in_waiting = 4096

        def read(self, n):
            return b"\x00" * n

    assert len(ler_fila(Infinita(), max_lotes=5)) == 5 * 4096
