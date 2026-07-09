#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# launcher_camera_controle.py
# Fase 3 - abre a CAMERA e o CONTROLE ao mesmo tempo, cada um na sua janela.
# Projeto TCC Carro Autonomo RC - Instituto Maua.
#
# O que ele faz: dispara os DOIS scripts ja existentes como processos
# separados (nao junta os dois num so - isso seria a versao integrada).
# Um Ctrl+C aqui encerra os dois de forma limpa.
#
# Ambiente: Jetson Nano / JetPack 4.6.x / Python 3.6.
# Precisa de monitor HDMI plugado (ambos abrem janela grafica).
#
# Uso:
#   python3 launcher_camera_controle.py
#
# Se os nomes/caminhos dos scripts forem outros, ajuste CAMERA_SCRIPT
# e CONTROLE_SCRIPT abaixo.

import os
import sys
import time
import signal
import subprocess

# ----------------------------------------------------------------------
# CONFIGURACAO - ajuste os caminhos se necessario
# ----------------------------------------------------------------------
# Caminhos relativos a este launcher. Como o launcher fica na mesma pasta
# que os dois scripts ("Testes de Hardware"), basta o nome do arquivo.
# Se voce mover o launcher, troque por caminho absoluto.

CAMERA_SCRIPT   = "camera_teste.py"
CONTROLE_SCRIPT = "controle_teste.py"

# Pequena espera entre lancar um e outro (deixa a camera inicializar a
# pipeline GStreamer antes do controle subir). Ajustavel.
DELAY_ENTRE_LANCAMENTOS = 2.0


def caminho_do_script(nome):
    """Resolve o caminho do script relativo a pasta deste launcher."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, nome)


def lancar(nome):
    """Lanca um script Python como processo separado. Retorna o Popen."""
    caminho = caminho_do_script(nome)
    if not os.path.exists(caminho):
        print("ERRO: nao encontrei '{0}'".format(caminho))
        print("      Ajuste a constante no topo do launcher.")
        return None
    print("Lancando: {0}".format(nome))
    # Mesmo interpretador que roda o launcher, para usar o mesmo ambiente.
    return subprocess.Popen([sys.executable, caminho])


def encerrar(proc, nome):
    """Encerra um processo com cuidado: SIGINT (como Ctrl+C), depois forca."""
    if proc is None or proc.poll() is not None:
        return
    print("Encerrando {0}...".format(nome))
    try:
        # SIGINT imita Ctrl+C -> deixa o script rodar o finally dele
        # (o controle_teste.py centraliza o servo e neutraliza o ESC ai).
        proc.send_signal(signal.SIGINT)
        # Da um tempo pro encerramento limpo.
        for _ in range(30):           # ate ~3s
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    except Exception:
        pass
    # Se ainda estiver vivo, termina a forca.
    if proc.poll() is None:
        try:
            proc.terminate()
            time.sleep(0.5)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    print("=== Launcher: Camera + Controle ===")
    print("Ctrl+C encerra os dois. Rodas fora do chao se for armar o ESC.")
    print("")

    # Lanca a camera primeiro, depois o controle.
    cam = lancar(CAMERA_SCRIPT)
    if cam is None:
        sys.exit(1)

    time.sleep(DELAY_ENTRE_LANCAMENTOS)

    ctrl = lancar(CONTROLE_SCRIPT)
    if ctrl is None:
        # Se o controle nao subiu, nao deixa a camera orfa.
        encerrar(cam, "camera")
        sys.exit(1)

    print("")
    print("Os dois estao rodando. Clique na janela do CONTROLE para")
    print("dar foco ao teclado. Ctrl+C aqui encerra ambos.")
    print("")

    try:
        # Fica vivo enquanto os dois processos viverem.
        # Se um dos dois fechar sozinho, encerramos o outro tambem,
        # para nao deixar processo orfao segurando camera/I2C.
        while True:
            if cam.poll() is not None:
                print("A camera encerrou. Fechando o controle tambem.")
                break
            if ctrl.poll() is not None:
                print("O controle encerrou. Fechando a camera tambem.")
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nCtrl+C recebido.")
    finally:
        # Encerra o controle primeiro (pra ele centralizar o servo),
        # depois a camera.
        encerrar(ctrl, "controle")
        encerrar(cam, "camera")
        print("Tudo encerrado.")


if __name__ == "__main__":
    main()