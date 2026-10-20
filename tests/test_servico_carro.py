"""O servico que abre o programa do controle quando o Jetson liga
(hardware/servico_carro.py).

No dia da feira so o controle_8bitdo.py roda, sem monitor nem teclado no carro
(pedido do Rafael em 2026-10-20). O que estes testes protegem e o que ja deu
errado ou daria errado sem ninguem olhando: o systemd para programas com
SIGTERM, e so o Ctrl+C (SIGINT) faz o programa estacionar o carro antes de
sair; e um servico nao le o .bashrc, entao as bibliotecas do CUDA somem como
sumiram com sudo em 2026-10-14.
"""
import importlib.util
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "hardware", "servico_carro.py")

REPO_JETSON = "/home/jetson/CarlaSimulator-TCC"
TERMINAL = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:", "HOME": "/home/jetson",
            "OPENBLAS_CORETYPE": "ARMV8", "PS1": "$ "}


@pytest.fixture
def sc():
    spec = importlib.util.spec_from_file_location("servico_carro", _SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _linhas(texto):
    return [l.strip() for l in texto.splitlines()]


def _unidade(sc, ambiente=None, extras=()):
    return sc.unidade("jetson", REPO_JETSON, "/usr/bin/python3",
                      sc.ambiente_do_terminal(TERMINAL if ambiente is None else ambiente),
                      extras=extras)


# --------------------------------------------------------------- a unidade

def test_it_runs_the_controller_program_as_the_user_from_the_repo(sc):
    linhas = _linhas(_unidade(sc))
    assert "User=jetson" in linhas
    assert "WorkingDirectory=" + REPO_JETSON in linhas
    assert ("ExecStart=/usr/bin/python3 " + REPO_JETSON + "/hardware/controle_8bitdo.py"
            in linhas)


def test_stopping_the_service_is_a_ctrl_c_so_the_car_is_parked(sc):
    linhas = _linhas(_unidade(sc))
    assert "KillSignal=SIGINT" in linhas
    # Tempo para estacionar e fechar camera e LiDAR antes do SIGKILL.
    prazo = [l for l in linhas if l.startswith("TimeoutStopSec=")]
    assert prazo and int(prazo[0].split("=")[1]) >= 10


def test_after_a_failure_it_starts_again_and_never_gives_up(sc):
    linhas = _linhas(_unidade(sc))
    assert "Restart=on-failure" in linhas
    assert "StartLimitIntervalSec=0" in linhas
    assert any(l.startswith("RestartSec=") for l in linhas)


def test_it_waits_for_the_camera_daemon(sc):
    linhas = _linhas(_unidade(sc))
    assert "After=nvargus-daemon.service" in linhas
    assert "Wants=nvargus-daemon.service" in linhas
    assert "WantedBy=multi-user.target" in linhas


def test_the_terminal_cuda_libraries_go_into_the_service(sc):
    linhas = _linhas(_unidade(sc))
    assert 'Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:"' in linhas
    assert 'Environment="OPENBLAS_CORETYPE=ARMV8"' in linhas
    # O print sai na hora no journal, nao em blocos.
    assert 'Environment="PYTHONUNBUFFERED=1"' in linhas
    # So o que importa para o programa; o resto do terminal fica de fora.
    assert not any("PS1" in l or "HOME=" in l for l in linhas)


def test_without_ld_library_path_in_the_terminal_the_cuda_folder_is_used(sc):
    linhas = _linhas(_unidade(sc, ambiente={}))
    assert 'Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"' in linhas


def test_extra_flags_go_to_the_controller_program(sc):
    linhas = _linhas(_unidade(sc, extras=["--grava", "--max-us", "1650"]))
    assert ("ExecStart=/usr/bin/python3 " + REPO_JETSON
            + "/hardware/controle_8bitdo.py --grava --max-us 1650") in linhas


@pytest.mark.parametrize("repo", ["/home/jetson/meu repo", "/home/jetson/100%"])
def test_a_repo_path_the_unit_cannot_hold_is_refused(sc, repo):
    with pytest.raises(ValueError):
        sc.unidade("jetson", repo, "/usr/bin/python3", {})


# ------------------------------------------------------------- o comando

class Sistema(object):
    """sudo/systemctl de mentira: guarda cada comando e o que foi para o stdin."""

    def __init__(self):
        self.comandos = []

    def __call__(self, comando, entrada=None):
        self.comandos.append((list(comando), entrada))


def _prepara(sc, monkeypatch, euid=1000):
    sistema = Sistema()
    monkeypatch.setattr(sc, "_roda", sistema)
    monkeypatch.setattr(sc.os, "geteuid", lambda: euid, raising=False)
    monkeypatch.setattr(sc, "_usuario", lambda: "jetson")
    monkeypatch.setattr(sc, "_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(sc, "_REPO", REPO_JETSON)
    monkeypatch.setattr(sc.os, "environ", dict(TERMINAL))
    return sistema


def test_mostra_prints_the_unit_and_touches_nothing(sc, monkeypatch, capsys):
    sistema = _prepara(sc, monkeypatch)
    assert sc.main(["mostra"]) == 0
    saida = capsys.readouterr().out
    assert "KillSignal=SIGINT" in saida and "User=jetson" in saida
    assert "NOPASSWD: /sbin/shutdown -h now" in saida
    assert sistema.comandos == []


def test_instala_writes_the_unit_and_enables_it_for_the_next_boot(sc, monkeypatch, capsys):
    sistema = _prepara(sc, monkeypatch)
    assert sc.main(["instala", "--grava"]) == 0
    destino = "/etc/systemd/system/carro-controle.service"
    escrita, recarga, habilita = sistema.comandos[:3]
    assert escrita[0] == ["sudo", "tee", destino]
    assert "controle_8bitdo.py --grava" in escrita[1].decode("utf-8")
    assert recarga[0] == ["sudo", "systemctl", "daemon-reload"]
    # So habilita: o carro nao comeca a ouvir o controle no meio da instalacao.
    assert habilita[0] == ["sudo", "systemctl", "enable", "carro-controle.service"]


def test_instala_lets_the_program_shut_the_jetson_down(sc, monkeypatch):
    # LB + RB por 3 s desliga o Jetson; o programa nao e root, entao o sudo
    # precisa deixar SO esse comando sem senha.
    sistema = _prepara(sc, monkeypatch)
    sc.main(["instala"])
    rascunho = "/etc/sudoers.d/carro-desligar.novo"
    assert [c for c, _ in sistema.comandos[3:]] == [
        ["sudo", "tee", rascunho],
        # Uma regra quebrada pode travar o sudo inteiro: confere antes de valer.
        # Enquanto o nome tem ponto, o sudo ignora o arquivo.
        ["sudo", "visudo", "-c", "-f", rascunho],
        ["sudo", "chmod", "0440", rascunho],
        ["sudo", "mv", rascunho, "/etc/sudoers.d/carro-desligar"]]
    assert sistema.comandos[3][1].decode("utf-8") == sc.regra_do_sudo("jetson")


def test_the_sudo_rule_allows_only_shutting_down_now(sc):
    regras = [l for l in sc.regra_do_sudo("jetson").splitlines()
              if l.strip() and not l.startswith("#")]
    assert regras == ["jetson ALL=(root) NOPASSWD: /sbin/shutdown -h now"]


@pytest.mark.parametrize("usuario", ["jet son", "root ALL=(ALL) NOPASSWD: ALL #", ""])
def test_a_user_name_that_could_change_the_sudo_rule_is_refused(sc, usuario):
    with pytest.raises(ValueError):
        sc.regra_do_sudo(usuario)


def test_remove_stops_disables_and_deletes(sc, monkeypatch):
    sistema = _prepara(sc, monkeypatch)
    assert sc.main(["remove"]) == 0
    comandos = [c for c, _ in sistema.comandos]
    assert comandos == [
        ["sudo", "systemctl", "disable", "--now", "carro-controle.service"],
        ["sudo", "rm", "-f", "/etc/systemd/system/carro-controle.service"],
        ["sudo", "rm", "-f", "/etc/sudoers.d/carro-desligar"],
        ["sudo", "systemctl", "daemon-reload"]]


def test_installing_with_sudo_is_refused(sc, monkeypatch):
    # Com sudo o usuario viraria root e o ambiente do CUDA se perderia.
    sistema = _prepara(sc, monkeypatch, euid=0)
    with pytest.raises(SystemExit) as saida:
        sc.main(["instala"])
    assert "sudo" in str(saida.value.code)
    assert sistema.comandos == []
