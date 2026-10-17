"""Os argumentos e a pasta de log de uma corrida.

Desde 2026-10-17 a corrida roda dentro do programa do controle, montada pelo
parser do proprio jetson_runtime.py (``cria_parser``). Estes argumentos sao o
que o programa do controle entrega a esse parser: os mesmos do comando que se
digitava a mao (docs/ESTADO_IA.md, secao 2).
"""
import ast
import datetime
import io
import os

from ai.car.largada import argumentos_da_corrida, pasta_da_corrida

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME = os.path.join(_REPO, "hardware", "jetson_runtime.py")


def _flags_do_runtime():
    with io.open(_RUNTIME, encoding="utf-8") as fh:
        arvore = ast.parse(fh.read(), filename=_RUNTIME)
    flags = set()
    for no in ast.walk(arvore):
        if (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
                and no.func.attr == "add_argument"):
            for arg in no.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    return flags


def test_the_run_arguments_are_the_ones_typed_by_hand():
    assert argumentos_da_corrida(
        engine="models/driving_grade_v5d1.engine",
        config="models/driving_grade_v5d1.json",
        out="runs/x", cruise_us=1610, beco_dist=0.5) == [
            "--engine", "models/driving_grade_v5d1.engine",
            "--config", "models/driving_grade_v5d1.json",
            "--out", "runs/x", "--cruise-us", "1610", "--beco-dist", "0.5"]


def test_extra_flags_go_to_the_runtime_untouched():
    argumentos = argumentos_da_corrida(engine="e", config="c", out="o", cruise_us=1610,
                                       beco_dist=0.5, extras=["--seconds", "120"])
    assert argumentos[-2:] == ["--seconds", "120"]


def test_every_flag_in_the_run_arguments_exists_in_the_runtime():
    # Um nome de flag errado so apareceria no carro, ao abrir o programa.
    argumentos = argumentos_da_corrida(engine="e", config="c", out="o",
                                       cruise_us=1610, beco_dist=0.5)
    flags = {parte for parte in argumentos if parte.startswith("--")}
    assert flags and flags <= _flags_do_runtime()


def test_each_run_logs_to_a_folder_named_after_the_model_and_the_time():
    quando = datetime.datetime(2026, 10, 13, 14, 25, 1)
    assert pasta_da_corrida("models/driving_grade_v5d1.engine", quando) == \
        "runs/driving_grade_v5d1_2026-10-13_142501"
