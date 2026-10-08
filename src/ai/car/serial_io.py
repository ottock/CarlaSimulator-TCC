"""Leitura da serial do LiDAR sem deixar fila para tras. Python 3.6-safe.

Achado nas corridas de 2026-10-07 (runs/Diag/lidar_x4_2026-10-07_2): o laco
travava no arranque (ate 9 s), o LiDAR seguia mandando dados, e eles se
acumulavam no buffer da serial. A leitura antiga pegava so ``in_waiting`` uma
vez por quadro -- e o buffer de leitura do Linux (N_TTY) entrega no maximo
~4 KB por vez, uns 3 scans. Resultado medido: 30-33 scans por meio segundo
saindo da fila contra os 5 que o sensor de fato produz, e o modelo recebendo o
LiDAR de SEGUNDOS atras -- o carro ainda no ponto de partida -- enquanto ja
andava. As tres batidas rapidas daquelas corridas aconteceram nessa janela.

Esvaziar a fila inteira a cada leitura faz o atraso, se houver, sumir em um
quadro so: o parser processa tudo e o laco fica com a volta mais nova.
"""


def ler_fila(ser, max_lotes=64):
    """Le TUDO que esta esperando na serial, nao so o primeiro lote.

    Args:
        ser: objeto com ``in_waiting`` e ``read(n)`` (pyserial).
        max_lotes: teto de leituras por chamada, para uma serial que nunca
            esvazia nao prender o laco. 64 lotes de 4 KB = 256 KB, mais de
            vinte segundos de LiDAR.

    Returns:
        ``bytes`` com tudo que foi lido (vazio se nao havia nada).
    """
    dados = bytearray()
    for _ in range(int(max_lotes)):
        n = ser.in_waiting
        if not n:
            break
        dados += ser.read(n)
    return bytes(dados)
