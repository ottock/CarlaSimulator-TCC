#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Larga e para a corrida pelo controle 8BitDo: Start larga, Back para.

Por que existe. Ate 2026-10-13 cada corrida comecava com alguem digitando o
comando do jetson_runtime.py num terminal. Com o controle lido direto da USB
(docs/CONTROLE_8BITDO.md), quem opera o carro so precisa de dois botoes.

O QUE ELE FAZ
  - Start: lanca o hardware/jetson_runtime.py com o comando de sempre (o do
    topo da secao 2 de docs/ESTADO_IA.md). Cada corrida grava numa pasta nova
    em runs/. Com uma corrida em andamento, Start nao faz nada.
  - Back (o botao ao lado do Start): manda ao runtime o mesmo sinal do Ctrl+C.
    Ele sai do laco, centraliza o servo e poe o ESC em neutro.
  - Quando a corrida termina (Back, ou o tempo do runtime acabou), volta a
    esperar o Start.
  - Ctrl+C aqui, ou o adaptador sumir da USB: para a corrida e sai.
  - Este script NAO mexe no modelo, no esterco nem no ESC. Quem dirige e o
    jetson_runtime.py, sem nenhuma alteracao.

O QUE ELE NAO RESOLVE
  - O modelo e carregado de novo a cada corrida. Entre apertar Start e o carro
    andar passa o mesmo tempo de quando o comando e digitado a mao.
  - No fim da pista o carro para, mas o runtime continua ate o tempo dele
    acabar (--seconds, padrao 60). Aperte Back antes da proxima largada.
  - Se o controle desligar no meio da corrida, nao ha Back: o carro para
    sozinho no fim da pista, na parada frontal ou no fim do tempo. Fique ao
    lado da chave do carro.

SEGURANCA
  - O carro so anda se ESC_ARMADO = True no jetson_runtime.py, como sempre.
    Este script nao arma nem desarma nada.
  - Se o runtime nao responder ao Back, o script avisa na tela e NAO o mata:
    matar pularia a limpeza e o ESC ficaria no ultimo pulso. Desligue a chave.

COMO USAR
  cd ~/CarlaSimulator-TCC
  python3 hardware/largada_controle.py
  Controle ligado (chave traseira em 2.4G, luz do Home fixa). Ctrl+C sai.
  Qualquer flag que este script nao conhece vai direto para o runtime:
      python3 hardware/largada_controle.py --seconds 120

  Ler o adaptador sem sudo exige a regra de permissao da secao 7 de
  docs/CONTROLE_8BITDO.md. Com sudo funciona, mas a corrida roda como root.

PRIMEIRA VEZ: rodas no ar. Escrito em 2026-10-13 e ainda NAO rodado no Jetson;
a logica (ai/car/largada.py, ai/car/gamepad.py) foi testada no PC com um
controle e um runtime de mentira.
"""
import argparse
import datetime
import os
import signal
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai.car.gamepad import Controle, ControlePerdido, acha_adaptador
from ai.car.largada import Largada, comando_da_corrida, laco, pasta_da_corrida

_RUNTIME = os.path.join("hardware", "jetson_runtime.py")

_FALAS = {
    "largou": "=== CORRIDA LANCADA (o modelo ainda vai carregar). Back para. ===",
    "parando": "=== Back: parando a corrida... ===",
    "acabou": "=== corrida encerrada. Start larga outra. ===",
    "nao_parou": "!!! O RUNTIME NAO RESPONDEU. DESLIGUE A CHAVE DO CARRO. !!!",
}


def _sigint_padrao():
    # Lancado em segundo plano (nohup, &), o shell deixa o SIGINT ignorado e o
    # filho herda isso: o runtime nunca veria o "Ctrl+C" do Back.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Start no controle larga a corrida, Back para.",
        epilog="Qualquer outra flag vai direto para o jetson_runtime.py "
               "(ex.: --seconds 120).")
    p.add_argument("--engine", default="models/driving_grade_v5d1.engine",
                   help="Arquivo .engine do TensorRT (padrao: o v5d1)")
    p.add_argument("--config", default="models/driving_grade_v5d1.json",
                   help="Sidecar .json do modelo")
    p.add_argument("--cruise-us", type=int, default=1610,
                   help="PWM constante do ESC passado ao runtime (padrao 1610)")
    p.add_argument("--beco-dist", type=float, default=0.5,
                   help="Parada de fim de pista passada ao runtime (padrao 0.5)")
    a, extras = p.parse_known_args(argv)

    # Melhor descobrir agora do que depois de apertar Start.
    for caminho in (a.engine, a.config):
        if not os.path.exists(os.path.join(_REPO, caminho)):
            sys.exit("nao achei {0} em {1}".format(caminho, _REPO))

    try:
        controle = Controle(acha_adaptador())
        controle.abre()
    except ImportError:
        sys.exit("falta o pyusb: sudo apt install python3-usb")
    except ControlePerdido as erro:
        sys.exit(str(erro))
    except IOError as erro:
        sys.exit("nao consegui abrir o adaptador: {0}\n"
                 "Sem sudo, e preciso da regra de permissao "
                 "(docs/CONTROLE_8BITDO.md, secao 7).".format(erro))

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print("AVISO: rodando como root. A corrida tambem roda como root, o que")
        print("       nunca foi testado, e os logs em runs/ ficam com outro dono.")

    def inicia():
        out = pasta_da_corrida(a.engine, datetime.datetime.now())
        comando = comando_da_corrida(
            sys.executable, _RUNTIME, engine=a.engine, config=a.config, out=out,
            cruise_us=a.cruise_us, beco_dist=a.beco_dist, extras=extras)
        print("$ " + " ".join(comando))
        return subprocess.Popen(comando, cwd=_REPO, preexec_fn=_sigint_padrao)

    largada = Largada(inicia, avisa=lambda evento: print(_FALAS[evento]))
    print("Esperando Start. Back para a corrida. Ctrl+C sai.")
    motivo = laco(controle, largada)
    print("saindo: {0}".format(motivo))
    return 0 if motivo == "interrompido" else 1


if __name__ == "__main__":
    sys.exit(main())
