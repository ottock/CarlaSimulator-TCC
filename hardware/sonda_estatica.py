#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pergunta ao modelo o que ele ve, com o carro PARADO numa posicao conhecida.

Por que existe: ate aqui todo diagnostico saiu de corridas inteiras, e corrida
mistura percepcao, controle, velocidade e sorte. Pior, uma metrica que usei para
culpar a percepcao ("esterca para longe da parede mais proxima") se mostrou sem
sentido: o EXPERT tira 50,0% nela, com correlacao +0,006. Na linha de pilotagem
o carro corre colado na parede externa e esterca PARA a curva.

Aqui nao ha inferencia: voce poe o carro num lugar, aperta uma tecla, e o
programa mede o que o modelo responde. Se ele acerta parado, o problema e de
controle ou velocidade. Se erra parado, e percepcao -- e da para ver em QUAL
posicao.

OPERADO AS CEGAS. Voce fica agachado na pista posicionando o carro, sem ver o
terminal. Por isso:
  - ha um MODO PREPARACAO antes de tudo, que confere os sensores e imprime a
    lista de posicoes na ordem. Leia isso ANTES de ir para a pista.
  - cada medicao e disparada por UMA tecla, sem Enter (seta direita, ou Enter,
    ou espaco -- o que a sua mao achar).
  - nada importante e impresso durante a medicao, porque voce nao estaria lendo.
    Tudo vai para a tabela final E para um arquivo.

O ESC fica em neutro o tempo todo: o carro nao anda.

Uso:
    python3 hardware/sonda_estatica.py --engine models/driving_oval_v3.engine \
        --config models/driving_oval_v3.json
"""
import argparse
import io
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import numpy as np

from ai.car.config import car_max_range, load_model_config
from ai.car.control_map import steer_to_us
from ai.car.image_crop import prepare_frame
from ai.car.teclas import LeitorDeTeclas, decodifica
from ai.shared.image_pipeline import preprocess

# Ordem fixa. Voce decora na preparacao e segue na pista sem olhar a tela.
#
# A coluna "esperado" NAO e intuicao -- a primeira versao desta lista era, e
# estava errada. Dizia "reta no centro: esperado perto de 0" quando o expert ali
# comanda +0.3 a +1.0, porque do EIXO a linha de pilotagem esta do lado de fora
# e o Pure Pursuit puxa para ela. Os numeros abaixo vem de avaliar o PROPRIO
# expert (lookahead 4.0, max_steer 27, linha estadio) nas poses, no oval_tcc.
#
# "treino" diz se a pose aparece no dataset. Isto importa mais que o esperado:
# na linha de pilotagem o rotulo e -0.73 em 70% da volta e NUNCA positivo; as
# poses colado na parede sao recuperacao, que e 0,25% dos quadros (36 de
# ~14.400). Errar numa pose FORA do treino nao acusa a percepcao -- acusa o
# dataset. Errar DENTRO do treino acusa a percepcao.
#
# A pista vira para a ESQUERDA, portanto a parede EXTERNA e a DIREITA do carro
# em toda a volta, inclusive nas retas. Verificado: a linha de pilotagem cai a
# direita do eixo em 48 de 48 pontos amostrados. Os rotulos abaixo dizem
# "DIREITA" em vez de "externa" porque, agachado na pista, "externa" numa reta
# nao quer dizer nada.
#
# faixa = (min, max) aceitavel, ou None quando o expert e ambiguo ali.
POSICOES = [
    ("1. RETA, no EIXO (centro da pista)",
     "+0.3 a +1.0 (DIREITA: a linha fica do lado de fora)", (0.05, 1.0), "nao"),
    ("2. RETA, colado na parede ESQUERDA",
     "+1.0 (DIREITA forte)", (0.3, 1.0), "nao"),
    ("3. RETA, colado na parede DIREITA",
     "-0.2 a -1.0 (ESQUERDA)", (-1.0, -0.05), "nao"),
    ("4. RETA, NA LINHA: flanco DIREITO a ~6 cm da parede DIREITA",
     "perto de 0 (a linha na reta e reta)", (-0.25, 0.25), "SIM"),
    ("5. CURVA, NA LINHA: flanco DIREITO a ~6 cm da parede DIREITA",
     "-0.73 (e o rotulo de 70% da volta)", (-1.0, -0.45), "SIM"),
    ("6. CURVA, colado na parede DIREITA (a externa), no meio",
     "-0.15 a -1.0 (ESQUERDA, para dentro)", (-1.0, -0.05), "nao"),
]


def _bip():
    """Apito do terminal. E a UNICA confirmacao que o operador recebe sem ver a
    tela: um bip quando a medicao terminou e ja pode mover o carro."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _mede(cam, lidar, eng, assembler, arcos, cfg, max_range, crop_frac, n, jr):
    """Roda o modelo por ``n`` quadros e devolve os esterços lidos."""
    from ai.car.lidar_frame import drop_self_occlusion, rotate_angles
    from ai.shared.lidar_pipeline import (apply_fov_mask, normalize_sectors_m,
                                          scan_to_sectors_m)
    steers, vec, t0 = [], None, time.time()
    while len(steers) < n and time.time() - t0 < 12.0:
        for scan in assembler.feed(lidar.read_points()):
            ang, dist = drop_self_occlusion([p[0] for p in scan],
                                            [p[1] for p in scan], arcos)
            ang = rotate_angles(ang, jr.LIDAR_OFFSET_DEG, jr.LIDAR_INVERT)
            sec = scan_to_sectors_m(ang, dist, cfg["n_sectors"], max_range)
            vec = normalize_sectors_m(apply_fov_mask(sec, cfg["fov_deg"], max_range),
                                      max_range)
        frame = cam.read()
        if frame is None or vec is None:
            continue
        steers.append(float(eng.infer(preprocess(prepare_frame(frame, crop_frac)),
                                      vec)[0]))
    return steers


def main():
    p = argparse.ArgumentParser(description="O que o modelo ve, com o carro parado")
    p.add_argument("--engine", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--crop-frac", type=float, default=0.48)
    p.add_argument("--amostras", type=int, default=20,
                   help="quadros por posicao; reporta media e dispersao")
    p.add_argument("--flip-method", type=int, default=0)
    p.add_argument("--out", default="sonda.txt", help="arquivo com a tabela final")
    a = p.parse_args()

    sys.path.insert(0, os.path.join(_REPO, "hardware"))
    import jetson_runtime as jr
    from ai.car.lidar_frame import parse_arcs
    from ai.car.scan_assembly import ScanAssembler

    cfg = load_model_config(a.config)
    max_range = car_max_range(cfg)

    # ---------------- MODO PREPARACAO ----------------
    print("")
    print("=" * 62)
    print("SONDA ESTATICA -- modo preparacao")
    print("=" * 62)
    print("Subindo camera, LiDAR e engine...")
    cam = jr.CsiCamera(flip_method=a.flip_method)
    # SEM max_range_m: o padrao e o alcance FISICO do sensor (12 m). Passar aqui
    # o max_range do modelo (1,0 m na escala do carro) faria o parser DESCARTAR
    # todo ponto alem de 1 m -- a parede do outro lado da pista sumiria, e com
    # menos de 50 pontos o ScanAssembler nunca emite scan. Daria "sensor nao
    # responde" com o sensor perfeito. O recorte na escala do modelo e feito
    # depois, em scan_to_sectors_m, igual ao DriveLoop.
    lidar = jr.SerialLidar()
    eng = jr.TrtEngine(a.engine)
    atu = jr.Pca9685Actuator()
    atu.safe_state()                       # ESC em neutro: o carro NAO anda
    assembler = ScanAssembler()
    arcos = parse_arcs(jr.SELF_OCCLUSION_ARCS)

    print("Conferindo que tudo responde (ate 12 s)...")
    teste = _mede(cam, lidar, eng, assembler, arcos, cfg, max_range,
                  a.crop_frac, 5, jr)
    if len(teste) < 5:
        print("")
        print("FALHOU: so %d leituras em 12 s. Camera ou LiDAR nao estao" % len(teste))
        print("entregando. Nao va para a pista assim -- conserte primeiro.")
        atu.safe_state(); cam.close(); lidar.close()
        return
    print("OK: %d leituras, esterco de teste %+.3f" % (len(teste), float(np.mean(teste))))
    print("ESC em NEUTRO -- o carro nao anda em nenhum momento.")
    print("")
    print("LEIA A ORDEM AGORA. Na pista voce nao vera a tela:")
    for rotulo, esperado, _faixa, treino in POSICOES:
        print("   %s" % rotulo)
        print("        esperado %-44s  no treino: %s" % (esperado, treino))
    print("")
    print("CONTROLES (uma tecla, sem Enter):")
    print("   seta DIREITA (ou Enter, ou espaco) -> mede e avanca")
    print("   seta ESQUERDA                      -> refaz a posicao anterior")
    print("   q                                  -> encerra e mostra a tabela")
    print("")
    print("Um BIP marca cada medicao concluida -- e a sua unica confirmacao")
    print("sem olhar a tela. So mova o carro depois de ouvir o bip.")
    print("")
    print("A tabela final tambem vai para %s." % a.out)
    print("Aperte uma tecla para comecar.")

    # ---------------- MEDICAO ----------------
    resultados = []
    i = 0
    try:
        with LeitorDeTeclas() as t:
            t.ler()
            _bip()                         # "ouvi, pode ir para a pista"
            while i < len(POSICOES):
                acao = decodifica(t.ler())
                if acao == "sair":
                    break
                if acao == "esquerda":
                    if resultados:
                        resultados.pop()
                        i -= 1
                    _bip()
                    continue
                if acao != "direita":
                    continue
                s = _mede(cam, lidar, eng, assembler, arcos, cfg, max_range,
                          a.crop_frac, a.amostras, jr)
                resultados.append(POSICOES[i] + (s,))
                i += 1
                # Primeiro o bip (medicao terminou), depois descarta o que ficou
                # na fila: quem nao ve a tela aperta de novo achando que nao
                # pegou, e essa tecla repetida mediria a posicao seguinte com o
                # carro ainda no lugar antigo.
                _bip()
                t.limpa()
    except KeyboardInterrupt:
        pass
    finally:
        atu.safe_state()
        cam.close()
        lidar.close()

    # ---------------- TABELA ----------------
    linhas = ["", "=" * 62, "RESULTADO", "=" * 62]
    for rotulo, esperado, faixa, treino, s in resultados:
        if not s:
            linhas.append("%-44s SEM LEITURA" % rotulo)
            continue
        arr = np.array(s)
        media = float(arr.mean())
        lado = ("DIREITA" if media > 0.05
                else ("ESQUERDA" if media < -0.05 else "reto"))
        if faixa is None:
            veredito = "(expert ambiguo aqui -- nao julga)"
        elif faixa[0] <= media <= faixa[1]:
            veredito = "OK"
        else:
            veredito = "FORA -- acusa %s" % ("a PERCEPCAO" if treino == "SIM"
                                             else "o DATASET (pose fora do treino)")
        linhas.append(rotulo)
        linhas.append("    steer %+.3f  (desvio %.3f em %d quadros)  servo %d us  -> %s"
                      % (media, arr.std(), len(arr), steer_to_us(media), lado))
        linhas.append("    esperado %s   [no treino: %s]" % (esperado, treino))
        linhas.append("    %s" % veredito)
        # Carro PARADO, cena identica: a saida tinha de repetir. Desvio alto
        # significa decisao no fio da navalha, nao medicao ruim.
        if arr.std() > 0.08:
            linhas.append("    ATENCAO: desvio %.3f com o carro parado -- a saida"
                          " oscila sobre a mesma cena" % arr.std())
    if not resultados:
        linhas.append("nenhuma posicao medida")
    texto = "\n".join(linhas)
    print(texto)
    try:
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write(texto + "\n")
        print("\nsalvo em %s" % a.out)
    except Exception as e:
        print("\nnao consegui salvar em %s: %s" % (a.out, e))


if __name__ == "__main__":
    main()
