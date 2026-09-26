#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mede a velocidade REAL do carro a um PWM fixo (medicao de bancada #3).

Anda em linha reta por X segundos ao mesmo PWM que o runtime usa, para voce medir
a distancia com uma trena e descobrir a velocidade de verdade.

Por que isto importa, sendo que o modelo nem comanda o acelerador: ele aprendeu a
estercar a UMA velocidade so. Nos 14.400 frames do `dataset_track_v1` a
distribuicao e quase uma delta (p5 1.69, p50 1.73, p95 1.80 m/s), o que na escala
1:12 da 0.144 m/s no carro. Esterco e velocidade nao sao independentes: a mesma
curva ao dobro da velocidade pede comando diferente. Se o ESC andar bem mais
rapido que isso, o carro CORTA as curvas -- falha lateral, dentro do escopo.

Nao usa camera nem LiDAR: e so servo + ESC. O tempo reportado e o medido com
`perf_counter` em volta da janela de PWM ligado, nao os `--seconds` nominais.

Para o carro ANDAR:
    1. ESC_ARMADO = True neste arquivo (nao e flag de CLI de proposito, para
       ninguem armar sem querer num comando copiado);
    2. --cruise-us a partir de 1600 (zona morta do ESC abaixo disso).

Uso:
    python3 hardware/teste_velocidade.py --seconds 5
    python3 hardware/teste_velocidade.py --seconds 15 --cruise-us 1610

    # depois das duas corridas, so a conta (nao toca no hardware):
    python3 hardware/teste_velocidade.py --calc 5:0.62 15:2.31
"""
import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai.car.control_map import (
    ESC_MIN_MOVE_US,
    ESC_NEUTRAL_US,
    STEER_CENTER_US,
    clamp_cruise_us,
)
from ai.car.speed_probe import (
    TARGET_CAR_SPEED_MS,
    average_speed_ms,
    speed_report,
    startup_loss_m,
    steady_speed_ms,
)

# Mesma trava do jetson_runtime.py: armar o ESC exige editar o arquivo.
ESC_ARMADO = False

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15
ESC_CHANNEL = 12

PERIOD_US = 1000000.0 / PWM_FREQ
STEPS = 4096


def _imprime_relatorio(rel, rotulo="distancia"):
    print("")
    print("=== RESULTADO ===")
    if rel["distancia_m"] is not None:
        print("  %-14s %.3f m" % (rotulo, rel["distancia_m"]))
    if rel["tempo_s"] is not None:
        print("  tempo          %.2f s" % rel["tempo_s"])
    print("  VELOCIDADE     %.3f m/s" % rel["velocidade_ms"])
    print("")
    print("  equivale no sim a %.2f m/s (x12)" % rel["sim_equivalente_ms"])
    print("  alvo            %.3f m/s  (p50 do dataset, 1.73 m/s no sim)"
          % rel["alvo_ms"])
    print("  faixa de treino %.3f a %.3f m/s  (p5-p95 do dataset)"
          % (rel["faixa_ms"][0], rel["faixa_ms"][1]))
    print("  razao           %.2fx do alvo" % rel["razao"])
    print("")
    if rel["veredito"] == "dentro":
        print("  -> DENTRO da faixa que o modelo viu no treino.")
    elif rel["veredito"] == "rapido":
        print("  -> RAPIDO demais: o modelo nunca estercou nesta velocidade.")
    else:
        print("  -> LENTO demais: o modelo nunca estercou nesta velocidade.")
    if rel["corta_curvas"]:
        print("")
        print("  ATENCAO: a %.2fx do alvo o esterco aprendido nao vale mais."
              % rel["razao"])
        print("  O carro tende a CORTAR as curvas, e isso e falha lateral, nao")
        print("  longitudinal. O conserto honesto seria recoletar com target_speed")
        print("  maior e retreinar -- nao ajustar ganho no runtime.")
    print("")


def _modo_calc(pares):
    """Duas corridas '<segundos>:<metros>' -> velocidade de regime."""
    medidas = []
    for texto in pares:
        try:
            t_txt, d_txt = texto.split(":")
            medidas.append((float(t_txt), float(d_txt)))
        except ValueError:
            raise SystemExit(
                "formato invalido: %r (esperado segundos:metros, ex. 5:0.62)" % texto)
    medidas.sort()
    if len(medidas) == 1:
        t, d = medidas[0]
        _imprime_relatorio(speed_report(average_speed_ms(d, t),
                                        distancia_m=d, tempo_s=t))
        print("  (media de UMA corrida: inclui a arrancada, entao subestima o")
        print("   regime. Rode outra, mais longa, e passe as duas para --calc)")
        print("")
        return

    (t1, d1), (t2, d2) = medidas[0], medidas[-1]
    v = steady_speed_ms(t1, d1, t2, d2)
    perda = startup_loss_m(v, t1, d1)
    print("")
    print("=== DUAS CORRIDAS (a arrancada se cancela) ===")
    print("  curta  %.2f s -> %.3f m   (media %.3f m/s)"
          % (t1, d1, average_speed_ms(d1, t1)))
    print("  longa  %.2f s -> %.3f m   (media %.3f m/s)"
          % (t2, d2, average_speed_ms(d2, t2)))
    print("  perda de arrancada: %.3f m  (= %.2f s parado equivalente)"
          % (perda, perda / v if v > 0 else 0.0))
    # O trecho usado e a DIFERENCA entre as duas corridas, nao uma medida sua --
    # rotulado para ninguem anotar 'distancia' e virar outro numero no relatorio.
    _imprime_relatorio(speed_report(v, distancia_m=d2 - d1, tempo_s=t2 - t1),
                       rotulo="trecho (d2-d1)")


def _pergunta_distancia():
    """Le a distancia do teclado; devolve None se nao der para perguntar."""
    prompt = "  distancia em metros (enter para pular): "
    leitor = input
    try:
        leitor = raw_input  # noqa: F821  -- Python 2/3.6 do Jetson
    except NameError:
        pass
    try:
        return float(leitor(prompt))
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def _modo_corrida(a):
    try:
        import Adafruit_PCA9685
    except ImportError:
        raise SystemExit("ERRO: Adafruit_PCA9685 nao instalado (rode isto no Jetson).")

    pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
    pwm.set_pwm_freq(PWM_FREQ)

    def set_us(channel, us):
        passos = int(us / (PERIOD_US / STEPS))
        pwm.set_pwm(channel, 0, max(0, min(4095, passos)))

    cruise = clamp_cruise_us(a.cruise_us)
    anda = ESC_ARMADO and cruise >= ESC_MIN_MOVE_US

    print("")
    print("=== Teste de velocidade ===")
    print("  ESC ARMADO     %s" % ESC_ARMADO)
    print("  cruise         %d us (pedido %d)" % (cruise, a.cruise_us))
    print("  servo          %d us (reto)" % a.servo_us)
    print("  duracao        %.1f s" % a.seconds)
    if not ESC_ARMADO:
        print("")
        print("  O carro NAO vai andar: ESC_ARMADO=False neste arquivo.")
    elif cruise < ESC_MIN_MOVE_US:
        print("")
        print("  O carro NAO vai andar: %d us esta na zona morta do ESC." % a.cruise_us)
    else:
        print("")
        print("  ESPACO LIVRE: no alvo (%.3f m/s) sao %.2f m, mas se o ESC for"
              % (TARGET_CAR_SPEED_MS, TARGET_CAR_SPEED_MS * a.seconds))
        print("  mais rapido a distancia cresce junto. Comece com --seconds 5.")
        print("  Marque o ponto de partida no chao ANTES de comecar.")
    print("")

    set_us(SERVO_CHANNEL, a.servo_us)
    set_us(ESC_CHANNEL, ESC_NEUTRAL_US)
    time.sleep(0.5)

    for n in range(3, 0, -1):
        print("  %d..." % n)
        time.sleep(1.0)

    t0 = time.perf_counter()
    try:
        print("  ANDANDO")
        set_us(ESC_CHANNEL, cruise if anda else ESC_NEUTRAL_US)
        while time.perf_counter() - t0 < a.seconds:
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("  interrompido")
    finally:
        set_us(ESC_CHANNEL, ESC_NEUTRAL_US)
        decorrido = time.perf_counter() - t0
        set_us(SERVO_CHANNEL, STEER_CENTER_US)

    print("  PAROU")
    print("")
    print("  tempo de PWM ligado: %.3f s  (use ESTE, nao os %.1f nominais)"
          % (decorrido, a.seconds))
    if not anda:
        print("  (o carro nao andou; nada para medir)")
        return

    print("")
    print("  Meca a distancia do ponto de partida ate onde ele parou.")
    dist = a.distancia if a.distancia is not None else _pergunta_distancia()
    if dist is None:
        print("")
        print("  Sem distancia. Quando medir, rode:")
        print("    python3 hardware/teste_velocidade.py --calc %.2f:<metros>" % decorrido)
        print("")
        return

    _imprime_relatorio(speed_report(average_speed_ms(dist, decorrido),
                                    distancia_m=dist, tempo_s=decorrido))
    print("  Para eliminar a arrancada, rode de novo com outra duracao e depois:")
    print("    python3 hardware/teste_velocidade.py --calc %.2f:%.2f <outra>:<metros>"
          % (decorrido, dist))
    print("")


def main():
    p = argparse.ArgumentParser(
        description="Mede a velocidade real do carro a um PWM fixo")
    p.add_argument("--seconds", type=float, default=5.0, help="duracao da corrida")
    p.add_argument("--cruise-us", type=int, default=ESC_MIN_MOVE_US,
                   help="PWM do ESC (a partir de %d; o runtime usou 1610)"
                        % ESC_MIN_MOVE_US)
    p.add_argument("--servo-us", type=int, default=STEER_CENTER_US,
                   help="PWM do servo; ajuste se o carro nao sair reto")
    p.add_argument("--distancia", type=float, default=None,
                   help="metros medidos, para nao esperar o prompt")
    p.add_argument("--calc", nargs="+", metavar="SEG:METROS", default=None,
                   help="so calcula, sem tocar no hardware; uma ou duas corridas")
    a = p.parse_args()
    if a.calc:
        _modo_calc(a.calc)
    else:
        _modo_corrida(a)


if __name__ == "__main__":
    main()
