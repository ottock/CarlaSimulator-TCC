#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Abre o programa do controle sozinho quando o Jetson liga (servico do systemd).

Por que existe. No dia da feira so o controle_8bitdo.py roda, e o carro nao tem
monitor nem teclado. O Rafael pediu em 2026-10-20 que, ao ligar o Jetson, o
programa ja esteja aberto e funcionando.

O que o servico garante, e por que:
  - Roda como o SEU usuario, nao root: o programa recusa root (com sudo a
    corrida nao achou as bibliotecas do CUDA em 2026-10-14).
  - Leva as variaveis do CUDA do terminal onde voce instalou: um servico nao
    le o .bashrc. Instale do mesmo terminal em que o programa funciona a mao.
  - Parar o servico e um Ctrl+C (SIGINT), nao o SIGTERM padrao: so o Ctrl+C faz
    o programa estacionar o carro antes de sair. Morto de outro jeito, o
    PCA9685 seguiria mandando o ultimo pulso. Vale tambem para desligar o
    Jetson pelo sistema.
  - Espera o daemon da camera (nvargus) e, se o programa cair (adaptador ou
    camera ainda nao apareceram no boot, controle perdido, erro), abre de novo
    5 s depois, sem desistir. Parado com Ctrl+C/systemctl stop, fica parado.

COMO USAR (no Jetson, SEM sudo -- ele pede a senha quando precisa)
  python3 hardware/servico_carro.py mostra            # so imprime o servico
  python3 hardware/servico_carro.py instala           # abre em todo boot
  python3 hardware/servico_carro.py instala --grava   # flags extras vao para
                                                      # o controle_8bitdo.py
  python3 hardware/servico_carro.py remove            # para e desfaz

  O instala so habilita: o programa abre no proximo boot (ou com
  sudo systemctl start carro-controle). Para ver o que ele imprime:
  journalctl -u carro-controle -f
"""
import argparse
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOME = "carro-controle.service"
DESTINO = "/etc/systemd/system/" + NOME
PROGRAMA = "hardware/controle_8bitdo.py"

# O que o programa precisa do terminal. LD_LIBRARY_PATH: as bibliotecas do CUDA
# (libcurand.so.10 sumiu com sudo). OPENBLAS_CORETYPE: o conserto comum do numpy
# no Jetson, se alguem o pos no .bashrc. PYTHONPATH: por precaucao.
VARIAVEIS = ("LD_LIBRARY_PATH", "OPENBLAS_CORETYPE", "PYTHONPATH")
CUDA_LIBS = "/usr/local/cuda/lib64"

_MODELO = """\
# Gerado por hardware/servico_carro.py. Nao edite aqui: rode o instala de novo.
[Unit]
Description=Carro TCC: controle 8BitDo (dirige a mao; Start larga a corrida)
Wants=nvargus-daemon.service
After=nvargus-daemon.service
# Tenta para sempre: no boot o adaptador ou a camera podem demorar a aparecer.
StartLimitIntervalSec=0

[Service]
Type=simple
User={usuario}
WorkingDirectory={repo}
{ambiente}
ExecStart={comando}
# Parar = Ctrl+C: o programa estaciona o carro (centro e neutro) antes de sair.
# Com o SIGTERM padrao ele morreria e o PCA9685 seguiria no ultimo pulso.
KillSignal=SIGINT
TimeoutStopSec=15
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def ambiente_do_terminal(environ):
    """As variaveis de ``VARIAVEIS`` que estao em ``environ``, e o CUDA sempre."""
    ambiente = {k: environ[k] for k in VARIAVEIS if environ.get(k)}
    ambiente.setdefault("LD_LIBRARY_PATH", CUDA_LIBS)
    return ambiente


def unidade(usuario, repo, python, ambiente, extras=()):
    """O texto do arquivo .service.

    Raises:
        ValueError: caminho com espaco ou ``%``, que o systemd leria errado.
    """
    for parte in [repo, python] + list(extras):
        if "%" in parte or any(c.isspace() for c in parte):
            raise ValueError("o servico nao aceita espaco nem %% em %r" % parte)
    linhas = ['Environment="PYTHONUNBUFFERED=1"']
    linhas += ['Environment="{0}={1}"'.format(k, ambiente[k]) for k in sorted(ambiente)]
    comando = " ".join([python, repo + "/" + PROGRAMA] + list(extras))
    return _MODELO.format(usuario=usuario, repo=repo, ambiente="\n".join(linhas),
                          comando=comando)


def _usuario():
    import pwd      # so existe no Linux

    return pwd.getpwuid(os.geteuid()).pw_name


def _python():
    return sys.executable or "/usr/bin/python3"


def _roda(comando, entrada=None):
    print("$ " + " ".join(comando))
    # O tee repete no terminal o que grava; o servico ja foi mostrado.
    subprocess.run(comando, input=entrada, check=True,
                   stdout=subprocess.DEVNULL if entrada is not None else None)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Abre o controle_8bitdo.py quando o Jetson liga.")
    p.add_argument("acao", choices=("mostra", "instala", "remove"))
    p.add_argument("extras", nargs=argparse.REMAINDER,
                   help="flags para o controle_8bitdo.py (ex.: --grava)")
    a = p.parse_args(argv)

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        sys.exit("Nao rode com sudo: o servico rodaria como root e sem as variaveis "
                 "do CUDA do seu terminal.\nRode como o usuario normal; o sudo e "
                 "pedido so para gravar o servico.")

    if a.acao == "remove":
        _roda(["sudo", "systemctl", "disable", "--now", NOME])
        _roda(["sudo", "rm", "-f", DESTINO])
        _roda(["sudo", "systemctl", "daemon-reload"])
        print("Removido: o programa nao abre mais sozinho.")
        return 0

    texto = unidade(_usuario(), _REPO, _python(), ambiente_do_terminal(os.environ),
                    extras=a.extras)
    if a.acao == "mostra":
        print(texto)
        return 0

    _roda(["sudo", "tee", DESTINO], entrada=texto.encode("utf-8"))
    _roda(["sudo", "systemctl", "daemon-reload"])
    _roda(["sudo", "systemctl", "enable", NOME])
    print("")
    print("Instalado. O programa abre sozinho no proximo boot.")
    print("  Abrir agora:      sudo systemctl start carro-controle")
    print("  Ver o que imprime: journalctl -u carro-controle -f")
    print("  Parar (estaciona): sudo systemctl stop carro-controle")
    print("  Desfazer:          python3 hardware/servico_carro.py remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
