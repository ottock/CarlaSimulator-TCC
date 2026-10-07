#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O que o LiDAR enxerga de perto? Radar ao vivo + menor distancia de PAREDE.

Por que existe. Nas rodadas de `runs/Diag/diag hz`, 60% dos quadros nao tinham
NENHUM ponto a menos de 30 cm (fora da carroceria), num corredor de 56 cm.

O QUE ELE ACHOU (2026-10-07). Um objeto a ~13 cm do sensor aparecia a 0,53 m, e
encostado parava em 0,200. O parser lia a distancia 4x maior: os 2 bits de baixo
de cada amostra sao flags (ver src/ai/car/coin_d6.py). Corrigido no parser, este
radar passa a mostrar distancias reais -- e a carroceria, que lia "0,202",
aparece a ~5 cm, que e tambem o alcance minimo do sensor.

O ERRO QUE A PRIMEIRA VERSAO TINHA. Ela tomava o minimo sobre TODOS os pontos.
Nas rodadas, 3.853 das 3.976 leituras abaixo de 0,215 m vinham do angulo 30-45
graus do sensor: a propria carroceria, a ~0,20 m atras do sensor. A versao
antiga leria o carro e anunciaria "ponto cego confirmado" nao importa o que o
operador fizesse. Agora o minimo e so de PAREDE, e a carroceria aparece em azul
no radar, separada, para se ver que ela nao conta.

O QUE ELE FAZ
  - liga SO o LiDAR. Nao abre camera, nao carrega engine, nao toca no PCA9685:
    servo e ESC nem sao criados. O carro nao tem como andar.
  - a cada volta do sensor imprime a menor distancia de parede DESSA volta, e
    marca quando bate o recorde do teste.
  - se houver tela, abre uma janela com o radar visto de cima (frente para cima)
    e um grafico do minimo no tempo. Sem tela (SSH sem -X), segue so no
    terminal. Em qualquer caso salva o ultimo quadro em ponto_cego.png.

COMO USAR
  1. Rode. Durante a contagem, encoste uma parede qualquer no carro PELA FRENTE
     OU PELOS LADOS -- a traseira e a carroceria (zona azul no radar) e e
     ignorada. Comece a ~40 cm e va ate ENCOSTAR. Repita umas 3 vezes, devagar.
  2. Bonus: ponha o carro na pista por alguns segundos e olhe o radar. Se o
     sensor enxerga as paredes, aparecem duas linhas de pontos a ~28 cm, uma de
     cada lado.
  3. `q` na janela (ou Ctrl+C) encerra antes do tempo.
"""
import argparse
import io
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import numpy as np

from ai.car import radar
from ai.car.lidar_frame import parse_arcs
from ai.car.scan_assembly import ScanAssembler

TITULO_JANELA = "LiDAR - radar (q para sair)"


def histograma(dists, limite=0.60, passo=0.05):
    """Conta quantas leituras de parede cairam em cada faixa de 5 cm."""
    linhas = []
    arr = np.asarray(dists, dtype=np.float64)
    n = len(arr)
    faixa = 0.0
    while faixa < limite - 1e-9:
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


# Alcance minimo do COIN-D6, medido a mao: encostado, o objeto parava em
# 0,050 m -- 0,040 m depois do desconto de 1 cm (LIDAR_DIST_OFFSET_M).
ALCANCE_MINIMO_M = 0.04


def veredito(menor):
    """Le o resultado SEM fingir saber o que o operador fez com o carro."""
    if menor is None or menor > 0.40:
        return ["Nenhuma parede chegou a menos de 40 cm FORA da carroceria.",
                "Encoste a parede pela frente ou pelos lados (fora da zona azul)."]
    if menor < ALCANCE_MINIMO_M + 0.02:
        return ["A parede chegou ao alcance minimo do sensor (%.3f m)." % menor,
                "Abaixo de ~5 cm o COIN-D6 nao mede: e o 'zero' dele."]
    return ["Mais perto que uma parede chegou: %.3f m." % menor,
            "Com o carro centrado na pista as paredes ficam a ~28 cm (anel vermelho)."]


def _abre_janela(sem_janela):
    """Janela so se houver tela. Uma falha aqui nunca derruba o teste."""
    if sem_janela:
        return False, "janela desligada (--sem-janela)"
    if not os.environ.get("DISPLAY"):
        return False, ("sem tela (DISPLAY vazio) -- rode no monitor do Jetson ou "
                       "com 'ssh -X'. Seguindo so no terminal.")
    try:
        import cv2
        cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_AUTOSIZE)
        return True, "janela aberta"
    except Exception as e:
        return False, "nao consegui abrir a janela (%s). Seguindo so no terminal." % e


def main():
    p = argparse.ArgumentParser(description="Radar do LiDAR e menor distancia de parede")
    p.add_argument("--segundos", type=float, default=30.0)
    p.add_argument("--raio", type=float, default=0.6,
                   help="alcance desenhado no radar, em metros")
    p.add_argument("--sem-janela", action="store_true",
                   help="nao abre janela; so terminal + png no fim")
    p.add_argument("--out", default="ponto_cego.txt")
    p.add_argument("--png", default="ponto_cego.png")
    a = p.parse_args()

    sys.path.insert(0, os.path.join(_REPO, "hardware"))
    import jetson_runtime as jr

    arcos = parse_arcs(jr.SELF_OCCLUSION_ARCS)
    off, inv = jr.LIDAR_OFFSET_DEG, jr.LIDAR_INVERT

    print("")
    print("=" * 62)
    print("RADAR DO LIDAR -- o carro NAO anda (so o LiDAR e ligado)")
    print("=" * 62)
    print("Subindo o LiDAR...")
    # Alcance FISICO (padrao 12 m): o radar quer ver tudo, nao so 1 m.
    lidar = jr.SerialLidar()
    janela, motivo = _abre_janela(a.sem_janela)
    print(motivo)
    print("")
    print("Encoste uma parede PELA FRENTE OU PELOS LADOS (a traseira e a carroceria,")
    print("que e ignorada), de ~40 cm ate ENCOSTAR. Umas 3 vezes, devagar.")
    print("")

    assembler = ScanAssembler()
    paredes_todas = []          # distancias de parede, o teste inteiro
    corpo_menor = None
    historico = []
    menor_teste = None
    recorde_impresso = None
    ultimo = None               # (parede, corpo, perto) da volta mais recente
    voltas = 0
    t0 = time.time()

    try:
        while time.time() - t0 < a.segundos:
            for scan in assembler.feed(lidar.read_points()):
                voltas += 1
                agora_t = time.time()
                parede_s, corpo_s = radar.separa(scan, arcos)
                parede = radar.para_carro(parede_s, off, inv)
                corpo = radar.para_carro(corpo_s, off, inv)
                paredes_todas.extend(d for _, d in parede)
                if corpo:
                    m = min(d for _, d in corpo)
                    corpo_menor = m if corpo_menor is None else min(corpo_menor, m)

                perto = radar.mais_proximo(parede)
                historico.append((agora_t, perto[1] if perto else None))
                if perto is not None and (menor_teste is None or perto[1] < menor_teste):
                    menor_teste = perto[1]
                ultimo = (parede, corpo, perto)
                restante = a.segundos - (agora_t - t0)

                # Recorde: linha permanente, mas so quando melhora 5 mm, para
                # nao virar uma cascata nos primeiros segundos.
                if (menor_teste is not None
                        and (recorde_impresso is None or menor_teste < recorde_impresso - 0.005)):
                    recorde_impresso = menor_teste
                    sys.stdout.write("\r%s\r  >> novo menor do teste: %.3f m  %+.0f graus (%s)\n"
                                     % (" " * 78, perto[1], radar.assinado(perto[0]),
                                        radar.lado(perto[0])))

                # A linha viva: a menor medida DESTA volta, sempre.
                if perto is not None:
                    viva = ("  agora %.3f m %+5.0f graus (%-8s) | menor do teste %.3f m | faltam %2.0f s"
                            % (perto[1], radar.assinado(perto[0]), radar.lado(perto[0]),
                               menor_teste, max(0.0, restante)))
                else:
                    viva = "  agora: nenhuma parede fora da carroceria | faltam %2.0f s" % max(0.0, restante)
                sys.stdout.write("\r" + viva.ljust(78))
                sys.stdout.flush()

                if janela:
                    try:
                        import cv2
                        img = radar.desenha(parede, corpo, perto, menor_teste, historico,
                                            restante, arcos_sensor=arcos,
                                            offset_deg=off, invert=inv, raio_m=a.raio)
                        cv2.imshow(TITULO_JANELA, img)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        janela = False
                        sys.stdout.write("\n  janela falhou (%s); seguindo so no terminal\n" % e)
            time.sleep(0.005)
    except KeyboardInterrupt:
        sys.stdout.write("\n  interrompido\n")
    finally:
        lidar.close()
        if janela:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
    sys.stdout.write("\n")

    # Ultimo quadro em arquivo: funciona mesmo sem tela, da para copiar via scp.
    if ultimo is not None:
        try:
            import cv2
            img = radar.desenha(ultimo[0], ultimo[1], ultimo[2], menor_teste, historico,
                                0.0, arcos_sensor=arcos, offset_deg=off, invert=inv,
                                raio_m=a.raio)
            cv2.imwrite(a.png, img)
        except Exception as e:
            print("nao consegui salvar %s: %s" % (a.png, e))

    linhas = ["", "=" * 62, "RESULTADO", "=" * 62]
    if not voltas:
        linhas.append("NENHUMA volta completa do LiDAR. Verifique a porta e a alimentacao.")
    else:
        linhas.append("voltas do sensor: %d   leituras de parede: %d" % (voltas, len(paredes_todas)))
        if corpo_menor is not None:
            linhas.append("carroceria: mais perto a %.3f m -- IGNORADA (o '0,202' das rodadas,"
                          " na escala errada)" % corpo_menor)
        linhas.append("MENOR distancia de PAREDE: %s"
                      % ("%.3f m" % menor_teste if menor_teste is not None else "nenhuma"))
        linhas.append("")
        if paredes_todas:
            linhas.append("histograma das leituras de parede:")
            linhas += histograma(paredes_todas)
            linhas.append("")
        linhas += veredito(menor_teste)
    texto = "\n".join(linhas)
    print(texto)
    try:
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write(texto + "\n")
        print("\nsalvo em %s e %s" % (a.out, a.png))
    except Exception as e:
        print("\nnao consegui salvar em %s: %s" % (a.out, e))


if __name__ == "__main__":
    main()
