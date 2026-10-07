#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pergunta ao modelo o que ele ve, com o carro PARADO numa posicao conhecida.

Por que existe: ate aqui todo diagnostico saiu de corridas inteiras, e corrida
mistura percepcao, controle, velocidade e sorte. Pior, uma metrica que eu usei
para culpar a percepcao ("esterca para longe da parede mais proxima") acabou se
mostrando sem sentido: o EXPERT tira 50,0% nela, com correlacao +0,006. Na
linha de pilotagem o carro corre colado na parede externa e esterca PARA a
curva, entao a metrica nao media erro nenhum.

Aqui nao ha inferencia: voce poe o carro num lugar, diz onde ele esta, e o
programa imprime o que o modelo responde. Se ele acerta parado, o problema e de
controle ou velocidade. Se erra parado, e percepcao -- e ai da para ver em qual
posicao.

O ESC fica em neutro o tempo todo: o carro nao anda.

Uso:
    python3 hardware/sonda_estatica.py --engine models/driving_oval_v3.engine \
        --config models/driving_oval_v3.json
"""
import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import numpy as np

from ai.car.config import car_max_range, load_model_config
from ai.car.control_map import STEER_CENTER_US, steer_to_us
from ai.car.image_crop import prepare_frame
from ai.shared.image_pipeline import preprocess

# Posicoes sugeridas. O nome e so rotulo; o que vale e voce por o carro ali.
POSICOES = [
    ("reta, centro, apontando para frente", "espera esterco perto de 0"),
    ("reta, colado na parede ESQUERDA", "espera esterco para a DIREITA (+)"),
    ("reta, colado na parede DIREITA", "espera esterco para a ESQUERDA (-)"),
    ("entrada da curva, no centro", "espera esterco no sentido da curva"),
    ("meio da curva, no centro", "espera esterco forte no sentido da curva"),
]


def main():
    p = argparse.ArgumentParser(description="O que o modelo ve, com o carro parado")
    p.add_argument("--engine", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--crop-frac", type=float, default=0.48)
    p.add_argument("--amostras", type=int, default=15,
                   help="quadros por posicao; imprime media e dispersao")
    p.add_argument("--flip-method", type=int, default=0)
    a = p.parse_args()

    from jetson_runtime import CsiCamera, SerialLidar, TensorRTEngine, Actuator
    from ai.car.scan_assembly import ScanAssembler
    from ai.car.lidar_frame import drop_self_occlusion, parse_arcs, rotate_angles
    from ai.shared.lidar_pipeline import apply_fov_mask, normalize_sectors_m, scan_to_sectors_m
    import jetson_runtime as jr

    cfg = load_model_config(a.config)
    max_range = car_max_range(cfg)
    cam = CsiCamera(flip_method=a.flip_method)
    lidar = SerialLidar(max_range_m=max_range)
    eng = TensorRTEngine(a.engine)
    atu = Actuator()
    atu.safe_state()                      # ESC em neutro: o carro NAO anda
    assembler = ScanAssembler()
    arcos = parse_arcs(jr.SELF_OCCLUSION_ARCS)
    vec = None

    print("")
    print("=== Sonda estatica ===")
    print("O carro NAO anda. Posicione, aperte Enter, leia a resposta.")
    print("Ctrl+C encerra.")
    print("")
    try:
        i = 0
        while True:
            rotulo, esperado = POSICOES[i % len(POSICOES)]
            i += 1
            print("--- %s" % rotulo)
            print("    (%s)" % esperado)
            try:
                raw_input("    posicione e aperte Enter... ")  # noqa: F821
            except NameError:
                input("    posicione e aperte Enter... ")

            steers, t0 = [], time.time()
            while len(steers) < a.amostras and time.time() - t0 < 10.0:
                for scan in assembler.feed(lidar.read_points()):
                    ang, dist = drop_self_occlusion(
                        [p0[0] for p0 in scan], [p0[1] for p0 in scan], arcos)
                    ang = rotate_angles(ang, jr.LIDAR_OFFSET_DEG, jr.LIDAR_INVERT)
                    sec = scan_to_sectors_m(ang, dist, cfg["n_sectors"], max_range)
                    vec = normalize_sectors_m(
                        apply_fov_mask(sec, cfg["fov_deg"], max_range), max_range)
                frame = cam.read()
                if frame is None or vec is None:
                    continue
                img = preprocess(prepare_frame(frame, a.crop_frac))
                steers.append(float(eng.infer(img, vec)[0]))

            if not steers:
                print("    SEM LEITURA (camera ou LiDAR nao entregaram)")
                continue
            s = np.array(steers)
            print("    steer = %+.3f   (desvio %.3f entre %d quadros)"
                  % (s.mean(), s.std(), len(s)))
            print("    servo = %d us   lado: %s" % (
                steer_to_us(s.mean()),
                "DIREITA" if s.mean() > 0.05 else ("ESQUERDA" if s.mean() < -0.05 else "reto")))
            print("")
    except KeyboardInterrupt:
        print("\nencerrado")
    finally:
        atu.safe_state()
        cam.close()
        lidar.close()


if __name__ == "__main__":
    main()
