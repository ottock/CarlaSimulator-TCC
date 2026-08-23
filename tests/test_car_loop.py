"""O laco de controle do carro (Fase 6b), testado com dubles.

Este e o teste do ENVELOPE DE SEGURANCA: prova, sem Jetson, que o servo centraliza
quando falta dado e que o ESC nunca sai de neutro nesta fase.
"""
import numpy as np
import pytest

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US
from ai.car.loop import DriveLoop


class FakeCamera:
    def __init__(self, frame=None):
        self.frame = frame if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

    def read(self):
        return self.frame


class FakeLidar:
    """Emite pontos como a serial real: em pedaços, com o wrap fechando a volta.

    O ScanAssembler descarta o primeiro fragmento por construcao (comecamos a
    ouvir no meio de uma volta), entao sao precisos DOIS wraps para um scan sair.
    A 1a leitura traz fragmento + volta inteira; a 2a fecha essa volta.
    Distancia 0.5 m: dentro do max_range=1.0 do carro, entao gera retorno de
    verdade em vez de virar "livre" e mascarar o teste.
    """

    def __init__(self):
        frag = [(float(a), 0.5) for a in range(300, 360, 2)]
        rev = [(float(a), 0.5) for a in range(0, 360, 2)]
        self.batches = [frag + rev, list(rev), list(rev)]

    def read_points(self):
        return self.batches.pop(0) if self.batches else []


class FakeEngine:
    def __init__(self, out=(0.5, 0.4, 0.0)):
        self.out = out
        self.calls = 0
        self.last_img = None
        self.last_lidar = None

    def infer(self, img, lidar):
        self.calls += 1
        self.last_img = img
        self.last_lidar = lidar
        return self.out


class FakeActuator:
    def __init__(self):
        self.servo_history = []
        self.esc_history = []

    def set_servo_us(self, us):
        self.servo_history.append(us)

    def set_esc_us(self, us):
        self.esc_history.append(us)


class FakeLogger:
    def __init__(self):
        self.frames = []
        self.scans = []

    def log_frame(self, **kw):
        self.frames.append(kw)

    def log_scan(self, **kw):
        self.scans.append(kw)


def _loop(**over):
    kw = dict(camera=FakeCamera(), lidar=FakeLidar(), engine=FakeEngine(),
              actuator=FakeActuator(), logger=FakeLogger(),
              fov_deg=180.0, max_range=1.0, crop_frac=0.5)
    kw.update(over)
    return DriveLoop(**kw), kw


def test_servo_is_centred_before_the_first_complete_revolution():
    # Sem uma volta completa o vetor de setores teria buracos que a rede leria como
    # "livre". Inferir nesse estado seria dirigir com um mapa falso.
    lidar = FakeLidar()
    lidar.batches = [[(0.0, 2.0), (10.0, 2.0)]]      # meia volta, sem wrap
    eng = FakeEngine()
    loop, kw = _loop(lidar=lidar, engine=eng)
    tele = loop.step()
    assert kw["actuator"].servo_history == [STEER_CENTER_US]
    assert eng.calls == 0
    assert tele["has_scan"] is False


def test_servo_follows_the_model_once_a_revolution_arrives():
    loop, kw = _loop(engine=FakeEngine(out=(0.5, 0.4, 0.0)))
    loop.step()                                   # fragmento + 1a volta acumulada
    tele = loop.step()                            # o 2o wrap fecha e roda o modelo
    assert tele["has_scan"] is True
    assert tele["steer"] == pytest.approx(0.5)
    assert kw["actuator"].servo_history[-1] == 1600


def test_esc_is_always_neutral():
    # a trava desta fase: o carro nao anda, ponto
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert kw["actuator"].esc_history
    assert set(kw["actuator"].esc_history) == {ESC_NEUTRAL_US}


def test_a_stalled_frame_centres_the_servo():
    # 3 passos de proposito: no 2o ja existe scan e o servo SEGUE o modelo, entao
    # o unico motivo de centralizar no 3o e o watchdog. Com 2 passos este teste
    # passaria mesmo sem watchdog nenhum.
    clock = iter([100.0, 100.05, 100.6]).__next__
    loop, kw = _loop(clock=clock, engine=FakeEngine(out=(0.5, 0.4, 0.0)))
    loop.step()
    loop.step()
    assert kw["actuator"].servo_history[-1] == 1600      # seguindo o modelo
    tele = loop.step()                                    # gap 0.55 s > 0.25
    assert tele["stalled"] is True
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US


def test_missing_camera_frame_centres_the_servo():
    class Blind:
        def read(self):
            return None

    loop, kw = _loop(camera=Blind())
    loop.step()
    tele = loop.step()
    assert tele["has_scan"] is True                # o LiDAR esta ok...
    assert set(kw["actuator"].servo_history) == {STEER_CENTER_US}   # ...mas sem imagem


def test_the_lidar_vector_is_masked_and_normalised():
    loop, kw = _loop()
    loop.step()
    tele = loop.step()
    vec = tele["lidar_vec"]
    assert vec.shape == (72,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0
    # frente: retorno real a 0.5 m com max_range 1.0 -> 0.5
    assert vec[0] == pytest.approx(0.5, abs=1e-6)
    # traseira: cega pela mascara de FOV 180 -> "livre"
    assert np.allclose(vec[18:54], 1.0)


def test_the_engine_receives_the_same_masked_vector_and_a_preprocessed_frame():
    # The other tests only look at the telemetry dict. Telemetry and the engine's
    # actual input could silently diverge -- e.g. a refactor that logs the masked
    # vector but infers on the raw one, or that forwards the raw frame straight to
    # infer() instead of running it through prepare_frame/preprocess. All 8 other
    # tests would stay green in either case, and the network would train on one
    # distribution and drive on another. This test looks at what infer() was
    # actually called with, not what the loop claims it did.
    eng = FakeEngine(out=(0.5, 0.4, 0.0))
    loop, kw = _loop(engine=eng)
    loop.step()
    tele = loop.step()
    # Identity, not equality: a copy-and-diverge refactor (pass a fresh array with
    # the same values instead of the one actually used) would still equal but not
    # be the same object.
    assert eng.last_lidar is tele["lidar_vec"]
    # Proves the crop-and-preprocess chain actually ran -- the raw camera frame is
    # (720, 1280, 3); only prepare_frame + preprocess produce this CHW model shape.
    assert eng.last_img.shape == (3, 66, 200)


def test_every_step_is_logged():
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert len(kw["logger"].frames) == 2


def test_completed_revolutions_are_logged_raw():
    loop, kw = _loop()
    loop.step()
    loop.step()
    assert len(kw["logger"].scans) == 1
