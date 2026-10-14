"""O script de largada recusa rodar como root.

Medido em 2026-10-14: com sudo, o runtime lancado por ele parou no import do
pycuda (libcurand.so.10 nao encontrada) -- o sudo limpa as variaveis de ambiente
que apontam para o CUDA. Melhor recusar logo, dizendo por que, do que deixar o
operador apertar Start e ver a corrida morrer.
"""
import importlib.util
import os

import pytest

from ai.car.gamepad import ControlePerdido

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "hardware", "largada_controle.py")
# Qualquer arquivo que exista serve: o script so confere que a engine existe.
_ARGS = ["--engine", "models/driving_grade_v5d1.onnx"]


def _carrega():
    spec = importlib.util.spec_from_file_location("largada_controle", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adaptador_que_anota(abertos):
    def acha():
        abertos.append(True)
        raise ControlePerdido("adaptador de teste")
    return acha


def test_running_as_root_is_refused_before_touching_the_controller(monkeypatch):
    lc = _carrega()
    abertos = []
    monkeypatch.setattr(lc.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(lc, "acha_adaptador", _adaptador_que_anota(abertos))
    with pytest.raises(SystemExit) as saida:
        lc.main(_ARGS)
    assert "sudo" in str(saida.value.code)
    assert abertos == []


def test_a_normal_user_gets_past_the_check(monkeypatch):
    lc = _carrega()
    abertos = []
    monkeypatch.setattr(lc.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(lc, "acha_adaptador", _adaptador_que_anota(abertos))
    with pytest.raises(SystemExit):
        lc.main(_ARGS)
    assert abertos == [True]
