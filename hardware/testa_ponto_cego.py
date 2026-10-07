#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qual a distancia MINIMA que o COIN-D6 consegue medir?

Por que isto importa mais que tudo o que medimos ate agora. Nas 9 rodadas de
`runs/Diag/diag hz` a menor distancia que o sensor reportou foi 0,202 m -- o
MESMO valor nas nove, e o nosso filtro de software corta em 0,05 m. Entao o
piso nao e nosso: parece ser o ponto cego fisico do sensor.

Se for isso, a consequencia e grave e explica o carro bater:

    corredor da pista : 53 cm  -> parede a 26,5 cm com o carro centrado
    ponto cego        : 20 cm
    => basta o carro sair 6,3 cm do eixo para a parede PROXIMA desaparecer

E setor sem retorno le `max_range` = "LIVRE" (ver scan_to_sectors_m). Ou seja,
a parede que ele esta prestes a bater e entregue ao modelo como espaco aberto.
Pior: a linha de pilotagem corre 11,6 cm fora do eixo, logo na propria linha a
parede de referencia esta a 14,9 cm, DENTRO do ponto cego.

No simulador nao existe ponto cego: no treino as duas paredes sempre estiveram
la. Se o teste confirmar, o conserto e modelar o ponto cego no treino.

COMO USAR (nao precisa de pista, so uma parede qualquer):
    1. Rode o script.
    2. Durante a contagem, aproxime o carro devagar de uma parede, de uns 40 cm
       ate ENCOSTAR, e afaste. Repita umas 3 vezes.
    3. Ele imprime a menor distancia que o sensor reportou, e um histograma.

Se aparecer algo abaixo de 0,10 m, minha conclusao esta errada e o piso de
0,202 m vem de outro lugar -- o que tambem e util saber.

O ESC nem e tocado: este script nao mexe no carro.
"""
import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import numpy as np


def histograma(dists, limite=0.60, passo=0.05):
    """Conta quantas leituras cairam em cada faixa de 5 cm."""
    linhas = []
    arr = np.asarray(dists, dtype=np.float64)
    n = len(arr)
    faixa = 0.0
    while faixa < limite:
        c = int(((arr >= faixa) & (arr < faixa + passo)).sum())
        if c:
            pct = 100.0 * c / n
            linhas.append("  %.2f a %.2f m : %7d  %5.1f%%  %s"
                          % (faixa, faixa + passo, c, pct, "#" * int(pct / 2)))
        faixa += passo
    acima = int((arr >= limite).sum())
    if acima:
        linhas.append("  acima de %.2f m : %7d  %5.1f%%"
                      % (limite, acima, 100.0 * acima / n))
    return linhas


def main():
    p = argparse.ArgumentParser(description="Ponto cego do LiDAR")
    p.add_argument("--segundos", type=float, default=30.0)
    p.add_argument("--out", default="ponto_cego.txt")
    a = p.parse_args()

    sys.path.insert(0, os.path.join(_REPO, "hardware"))
    import jetson_runtime as jr

    print("")
    print("=" * 62)
    print("PONTO CEGO DO LIDAR -- o carro NAO anda")
    print("=" * 62)
    print("Subindo o LiDAR...")
    # Alcance FISICO (padrao 12 m). Passar o alcance do modelo aqui descartaria
    # justamente as leituras distantes que queremos ver no histograma.
    lidar = jr.SerialLidar()
    print("")
    print("Agora, por %.0f s: aproxime o carro de uma parede, de ~40 cm ate" % a.segundos)
    print("ENCOSTAR, e afaste. Repita umas 3 vezes, devagar.")
    print("")

    todas = []
    t0 = time.time()
    ultimo_aviso = 0.0
    try:
        while time.time() - t0 < a.segundos:
            pts = lidar.read_points()
            for _ang, d in pts:
                todas.append(d)
            # Um aviso por segundo, so para saber que esta vivo.
            if time.time() - ultimo_aviso >= 1.0:
                resta = a.segundos - (time.time() - t0)
                menor = min(todas) if todas else float("nan")
                print("  faltam %4.0f s   leituras %7d   menor ate agora %.3f m"
                      % (resta, len(todas), menor))
                ultimo_aviso = time.time()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\ninterrompido")
    finally:
        lidar.close()

    linhas = ["", "=" * 62, "RESULTADO", "=" * 62]
    if not todas:
        linhas.append("NENHUMA leitura -- o LiDAR nao entregou nada. Verifique a porta.")
    else:
        arr = np.asarray(todas)
        linhas.append("leituras: %d" % len(arr))
        linhas.append("MENOR distancia reportada: %.3f m" % arr.min())
        linhas.append("")
        linhas += histograma(arr)
        linhas.append("")
        if arr.min() < 0.10:
            linhas.append("O sensor LE abaixo de 10 cm. Minha conclusao estava errada:")
            linhas.append("o piso de 0,202 m nas rodadas vem de outro lugar.")
        elif arr.min() < 0.18:
            linhas.append("Ponto cego menor que eu estimei (%.3f m). Da para usar," % arr.min())
            linhas.append("mas ainda cega a parede proxima em parte do corredor.")
        else:
            linhas.append("CONFIRMADO: ponto cego de %.3f m." % arr.min())
            linhas.append("Num corredor de 53 cm, a parede proxima desaparece quando o")
            linhas.append("carro sai %.1f cm do eixo -- e o setor passa a ler 'livre'."
                          % ((0.265 - arr.min()) * 100))
            linhas.append("Na linha de pilotagem (11,6 cm fora do eixo) a parede de")
            linhas.append("referencia esta a 14,9 cm: dentro do ponto cego.")
            linhas.append("Conserto: modelar este ponto cego no treino.")
    texto = "\n".join(linhas)
    print(texto)
    try:
        import io
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write(texto + "\n")
        print("\nsalvo em %s" % a.out)
    except Exception as e:
        print("\nnao consegui salvar em %s: %s" % (a.out, e))


if __name__ == "__main__":
    main()
