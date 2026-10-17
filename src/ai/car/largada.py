"""Os argumentos e a pasta de log de uma corrida. Python 3.6-safe.

"Corrida" e o laco de direcao do ``hardware/jetson_runtime.py`` rodando o
modelo. Desde 2026-10-17 ela roda dentro do programa do controle
(``hardware/controle_8bitdo.py``, coordenado por ``ai.car.controle_unico``),
montada pelo parser do proprio runtime (``cria_parser``) -- assim os padroes
ficam num lugar so. Estes argumentos sao o que se entrega a esse parser.

Antes disso, a corrida era lancada como OUTRO programa, e este modulo tinha a
classe ``Largada`` que cuidava do processo (sinal de Ctrl+C, nunca matar,
carencia entre sinais). Saiu quando a corrida passou para dentro do programa:
parar ficou imediato e sem processo para esperar.
"""
import os


def argumentos_da_corrida(engine, config, out, cruise_us, beco_dist, extras=()):
    """Os argumentos do runtime, os mesmos do comando digitado a mao (ESTADO_IA.md §2).

    ``extras`` vai no fim, sem mexer: e por onde entra ``--seconds 120`` ou
    qualquer outro ajuste do runtime.
    """
    return ["--engine", engine, "--config", config, "--out", out,
            "--cruise-us", str(cruise_us), "--beco-dist", str(beco_dist)
            ] + list(extras)


def pasta_da_corrida(engine, quando):
    """``runs/<modelo>_<data>_<hora>``: uma pasta de log por corrida."""
    modelo = os.path.splitext(os.path.basename(engine))[0]
    return "runs/{0}_{1}".format(modelo, quando.strftime("%Y-%m-%d_%H%M%S"))
