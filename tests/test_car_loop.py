"""O laco de controle do carro (Fase 6b), testado com dubles.

Este e o teste do ENVELOPE DE SEGURANCA: prova, sem Jetson, que o servo centraliza
quando falta dado e que o ESC nunca sai de neutro nesta fase.
"""
import numpy as np
import pytest

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US, STEER_SPAN_US
from ai.car.loop import DriveLoop


class FakeCamera:
    """Devolve um quadro COM TEXTURA.

    Preto uniforme seria lido como lente tapada (ver sensor_health) -- e uma
    camera real nunca entrega isso. O dublê tem de representar o caso normal.
    """

    def __init__(self, frame=None):
        if frame is None:
            rng = np.random.default_rng(7)
            frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        self.frame = frame

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
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US - round(0.5 * STEER_SPAN_US)


def test_esc_is_neutral_by_default():
    # Sem cruise_us configurado o carro NAO anda. Andar tem de ser um ato
    # deliberado (passar a velocidade), nunca o comportamento padrao.
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
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US - round(0.5 * STEER_SPAN_US)      # seguindo o modelo
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


# ---------------------------------------------------------------------------
# Velocidade de cruzeiro constante (Fase 6b parte 2)
#
# Regra do Rafael: velocidade CONSTANTE onde nao ha aceleracao; se ela nao puder
# ser constante, a velocidade e ZERO. Estes testes provam as duas metades.
# ---------------------------------------------------------------------------

class NearLidar(FakeLidar):
    """Volta completa com um obstaculo colado na frente (0.1 m) e livre dos lados."""

    def __init__(self):
        FakeLidar.__init__(self)

        def d(a):
            return 0.1 if (a <= 10 or a >= 350) else 0.5

        frag = [(float(a), d(a)) for a in range(300, 360, 2)]
        rev = [(float(a), d(a)) for a in range(0, 360, 2)]
        self.batches = [frag + rev, list(rev), list(rev)]


def test_cruise_drives_the_esc_when_everything_is_healthy():
    loop, kw = _loop(cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    loop.step()
    assert kw["actuator"].esc_history[-1] == STEER_CENTER_US + round(0.5 * STEER_SPAN_US)


def test_cruise_is_zero_before_a_complete_revolution():
    # Mesma razao do servo: sem uma volta inteira o vetor tem buracos que a rede
    # leria como "livre". Andar com um mapa falso e pior do que ficar parado.
    lidar = FakeLidar()
    lidar.batches = [[(0.0, 2.0), (10.0, 2.0)]]
    loop, kw = _loop(lidar=lidar, cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    assert kw["actuator"].esc_history == [ESC_NEUTRAL_US]


def test_cruise_is_zero_when_the_front_is_blocked():
    # A parada de emergencia e um item de SEGURANCA, nao um comportamento
    # aprendido: a cabeca de freio do modelo esta inerte (dataset com 0% de
    # frenagem), entao parar nao pode depender dela.
    loop, kw = _loop(lidar=NearLidar(), cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    tele = loop.step()
    assert tele["blocked"] is True
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US


def test_a_wall_beside_the_car_does_not_stop_it():
    # Numa pista de 0.53 m as paredes laterais estao SEMPRE perto. Se a parada
    # olhasse o circulo inteiro, o carro nunca sairia do lugar.
    loop, kw = _loop(cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))          # FakeLidar: 0.5 m em todas as direcoes
    loop.step()
    tele = loop.step()
    assert tele["blocked"] is False
    assert kw["actuator"].esc_history[-1] == STEER_CENTER_US + round(0.5 * STEER_SPAN_US)


def test_cruise_is_zero_on_a_stalled_frame():
    clock = iter([100.0, 100.05, 100.6]).__next__
    loop, kw = _loop(clock=clock, cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    loop.step()
    assert kw["actuator"].esc_history[-1] == STEER_CENTER_US + round(0.5 * STEER_SPAN_US)
    tele = loop.step()
    assert tele["stalled"] is True
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US


# ---------------------------------------------------------------------------
# Frame do LiDAR: auto-oclusao e alinhamento (medidos no carro em 2026-09-12)
#
# As funcoes puras ja estao cobertas em test_car_lidar_frame.py. Estes testes
# provam a FIACAO: que o laco de fato as aplica antes de montar o vetor. Sem
# eles, um refactor poderia deixar as correcoes configuradas e nao usadas.
# ---------------------------------------------------------------------------

class OneBlipLidar(FakeLidar):
    """Volta completa a 0.9 m, com um unico retorno perto num angulo conhecido."""

    def __init__(self, blip_deg, dist=0.2):
        FakeLidar.__init__(self)

        def d(a):
            desvio = abs(((a - blip_deg + 180.0) % 360.0) - 180.0)
            return dist if desvio < 2.0 else 0.9

        frag = [(float(a), d(a)) for a in range(300, 360, 2)]
        rev = [(float(a), d(a)) for a in range(0, 360, 2)]
        self.batches = [frag + rev, list(rev), list(rev)]


def test_self_occluded_sectors_read_free_instead_of_a_wall():
    # A carroceria lia 0.25 m fixo em um terco do circulo. Mascarada, esses
    # setores tem de virar "livre" (1.0) -- o mesmo tratamento que o treino deu
    # aos setores fora do FOV.
    loop, kw = _loop(self_occlusion=[(0.0, 10.0)])
    loop.step()
    tele = loop.step()
    vec = tele["lidar_vec"]
    assert vec[0] == pytest.approx(1.0)      # setor 0 = 0-5 graus, ocluido
    assert vec[1] == pytest.approx(1.0)      # setor 1 = 5-10 graus, ocluido
    assert vec[2] == pytest.approx(0.5)      # 10-15 graus segue enxergando


def test_the_offset_moves_a_return_into_the_front_sector():
    # Objeto a 240 graus no sensor + offset 240 => ele passa a estar na FRENTE.
    loop, kw = _loop(lidar=OneBlipLidar(240.0), lidar_offset_deg=240.0)
    loop.step()
    tele = loop.step()
    vec = tele["lidar_vec"]
    assert vec[0] == pytest.approx(0.2, abs=1e-6)
    assert vec.argmin() == 0


def test_without_the_offset_the_same_return_is_not_in_front():
    loop, kw = _loop(lidar=OneBlipLidar(240.0))
    loop.step()
    tele = loop.step()
    # 240 graus cai na traseira, que o FOV de 180 mascara como "livre"
    assert tele["lidar_vec"][0] == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# Saude dos sensores (bancada, 2026-09-12): tapar camera ou LiDAR NAO parava o
# carro. As funcoes puras estao em test_car_sensor_health.py; aqui provamos que
# o laco as aplica e corta a tracao.
# ---------------------------------------------------------------------------

def test_a_covered_camera_stops_the_car():
    # Lente tapada devolve quadro, so que uniforme. Antes disso o modelo opinava
    # sobre uma imagem preta e o carro seguia andando.
    preto = np.zeros((720, 1280, 3), dtype=np.uint8)
    loop, kw = _loop(camera=FakeCamera(frame=preto), cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    tele = loop.step()
    assert tele["blind"] is True
    assert tele["steer"] == 0.0
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US


def test_a_lidar_that_stops_spinning_stops_the_car():
    # O vetor antigo continua em memoria: sem verificar frescor, o carro dirige
    # por um mapa congelado.
    class MudoDepois(FakeLidar):
        def read_points(self):
            return self.batches.pop(0) if self.batches else []

    clock = iter([100.0, 100.05, 100.10, 101.0]).__next__
    loop, kw = _loop(lidar=MudoDepois(), clock=clock, cruise_us=STEER_CENTER_US + round(0.5 * STEER_SPAN_US))
    loop.step()
    loop.step()
    loop.step()
    assert kw["actuator"].esc_history[-1] == STEER_CENTER_US + round(0.5 * STEER_SPAN_US)      # ainda fresco
    tele = loop.step()                                 # 0.9 s sem volta nova
    assert tele["stale_lidar"] is True
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US
    assert kw["actuator"].servo_history[-1] == STEER_CENTER_US


# ---------------------------------------------------------------------------
# Ganho de esterco no laco (2026-09-26)
# ---------------------------------------------------------------------------

def _drive_one(loop):
    """Roda os passos necessarios para a primeira inferencia sair."""
    loop.step()      # fragmento + 1a volta acumulada
    return loop.step()


def test_the_gain_reaches_the_servo():
    eng = FakeEngine(out=(0.30, 0.4, 0.0))
    sem, kw_sem = _loop(engine=eng)
    _drive_one(sem)
    us_sem = kw_sem["actuator"].servo_history[-1]

    com, kw_com = _loop(engine=FakeEngine(out=(0.30, 0.4, 0.0)), steer_gain=2.0)
    _drive_one(com)
    us_com = kw_com["actuator"].servo_history[-1]

    # 0.30 com ganho 2 tem de dar o mesmo us que 0.60 sem ganho.
    ref, kw_ref = _loop(engine=FakeEngine(out=(0.60, 0.4, 0.0)))
    _drive_one(ref)
    assert us_com == kw_ref["actuator"].servo_history[-1]
    assert us_com != us_sem


def test_the_log_keeps_the_models_raw_steer_not_the_amplified_one():
    # replay_car_log compara o steer GRAVADO com a saida do modelo para medir
    # fidelidade. Se o log guardasse o valor ja amplificado, a fidelidade
    # quebraria e a atribuicao de sensor perderia o sentido.
    loop, kw = _loop(engine=FakeEngine(out=(0.30, 0.4, 0.0)), steer_gain=2.0)
    tele = _drive_one(loop)
    assert tele["steer"] == pytest.approx(0.30)
    assert kw["logger"].frames[-1]["control"][0] == pytest.approx(0.30)


def test_the_applied_steer_is_logged_too_so_runs_stay_comparable():
    # Sem isto, duas corridas com ganhos diferentes ficam identicas no log.
    loop, kw = _loop(engine=FakeEngine(out=(0.30, 0.4, 0.0)), steer_gain=2.0)
    tele = _drive_one(loop)
    assert tele["steer_aplicado"] == pytest.approx(0.60)


def test_the_gain_does_not_move_the_wheel_when_the_model_did_not_run():
    # Sem volta completa do LiDAR o servo vai ao centro; o ganho nao pode
    # transformar isso em esterco nenhum.
    lidar = FakeLidar()
    lidar.batches = [[(0.0, 2.0), (10.0, 2.0)]]
    loop, kw = _loop(lidar=lidar, steer_gain=3.0)
    loop.step()
    assert kw["actuator"].servo_history == [STEER_CENTER_US]


def test_the_default_gain_is_neutral():
    loop, kw = _loop(engine=FakeEngine(out=(0.30, 0.4, 0.0)))
    tele = _drive_one(loop)
    assert tele["steer_aplicado"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Mediana causal do esterco no laco (2026-10-03)
# ---------------------------------------------------------------------------

class SequenceEngine:
    """Devolve um esterco diferente a cada chamada, para exercitar o filtro."""

    def __init__(self, steers):
        self.steers = list(steers)
        self.i = 0

    def infer(self, img, lidar):
        s = self.steers[min(self.i, len(self.steers) - 1)]
        self.i += 1
        return (s, 0.4, 0.0)


def _ate_inferencia(loop, eng, n):
    """Roda o laco ate o modelo ter sido chamado ``n`` vezes.

    O engine so roda a partir do 2o step (a 1a volta do LiDAR ainda esta
    sendo montada), entao contar steps na mao erra -- e erraria em silencio,
    testando um quadro diferente do pretendido.
    """
    tele = None
    for _ in range(n + 5):
        tele = loop.step()
        if eng.i >= n:
            return tele
    raise AssertionError('o modelo nao rodou %d vezes' % n)


def test_an_isolated_spike_never_reaches_the_servo():
    # O padrao medido no carro: dois quadros a +0.4, um sozinho em -0.88, volta.
    # Os dois primeiros passam direto (janela ainda enchendo); o pico e cortado.
    eng = SequenceEngine([0.4, 0.4, -0.88, 0.4])
    loop, kw = _loop(engine=eng, steer_median=3)
    tele = _ate_inferencia(loop, eng, 3)          # a 3a inferencia e o -0.88
    assert tele["steer"] == pytest.approx(-0.88)          # cru preservado no log
    assert tele["steer_aplicado"] == pytest.approx(0.4)   # o servo nao ve o pico


def test_the_filter_runs_before_the_gain():
    # Filtrar depois do ganho trabalharia sobre valores ja grampeados em +/-1 e
    # perderia a informacao que distingue um pico de uma curva.
    eng = SequenceEngine([0.4, 0.4, -0.88, 0.4])
    loop, kw = _loop(engine=eng, steer_median=3, steer_gain=2.0)
    tele = _ate_inferencia(loop, eng, 3)
    assert tele["steer_aplicado"] == pytest.approx(0.8)   # 0.4 filtrado, depois x2


def test_the_default_does_not_filter():
    eng = SequenceEngine([0.4, 0.4, -0.88])
    loop, kw = _loop(engine=eng)
    tele = _ate_inferencia(loop, eng, 3)
    assert tele["steer_aplicado"] == pytest.approx(-0.88)


def test_a_degraded_frame_clears_the_history():
    # Sem volta completa do LiDAR o modelo nao roda e o servo vai ao centro.
    # Guardar o historico atravessando isso faria a curva anterior reaparecer
    # quando o carro voltasse a enxergar, num lugar que pode ser outro.
    lidar = FakeLidar()
    lidar.batches = [[(0.0, 2.0), (10.0, 2.0)]]
    loop, kw = _loop(lidar=lidar, engine=SequenceEngine([0.9]), steer_median=3)
    loop.step()
    assert loop.steer_filter._buf == []


# ---------------------------------------------------------------------------
# Trava de arranque e LiDAR atrasado (2026-10-07)
#
# Nas corridas de runs/Diag/lidar_x4_2026-10-07_2 o laco travava no inicio, o
# LiDAR enchia a fila da serial, e o modelo recebia o LiDAR de SEGUNDOS atras
# enquanto o carro ja andava: 30-33 voltas por meio segundo saindo da fila
# contra as 5 que o sensor produz. Tres das quatro batidas foram no primeiro
# meio metro. O ESC agora espera o laco ficar saudavel E o LiDAR em dia.
# ---------------------------------------------------------------------------

CRUZEIRO = STEER_CENTER_US + round(0.5 * STEER_SPAN_US)


class GiraSempre(FakeLidar):
    """Uma volta nova por leitura; num passo escolhido, uma FILA de voltas."""

    def __init__(self, fila_no_passo=None, voltas_na_fila=4):
        FakeLidar.__init__(self)
        self.passo = 0
        self.fila_no_passo = fila_no_passo
        self.voltas_na_fila = voltas_na_fila
        self.frag = [(float(a), 0.5) for a in range(300, 360, 2)]
        self.rev = [(float(a), 0.5) for a in range(0, 360, 2)]

    def read_points(self):
        self.passo += 1
        if self.passo == 1:
            return self.frag + self.rev
        if self.passo == self.fila_no_passo:
            return self.rev * self.voltas_na_fila
        return list(self.rev)


def _relogio(passo=0.05):
    t = [100.0]

    def agora():
        t[0] += passo
        return t[0]
    return agora


def _roda(loop, n):
    return [loop.step() for _ in range(n)]


def test_the_esc_waits_for_a_steady_start():
    loop, kw = _loop(lidar=GiraSempre(), clock=_relogio(), cruise_us=CRUZEIRO,
                     arranque_s=0.5)
    teles = _roda(loop, 20)
    esc = kw["actuator"].esc_history
    primeiro_saudavel = next(t["t"] for t in teles if t["has_scan"])
    primeiro_andando = next(t["t"] for t, e in zip(teles, esc) if e == CRUZEIRO)
    assert primeiro_andando - primeiro_saudavel >= 0.5 - 1e-6
    assert all(e == ESC_NEUTRAL_US for t, e in zip(teles, esc)
               if t["t"] < primeiro_saudavel + 0.45)
    assert esc[-1] == CRUZEIRO


def test_the_servo_keeps_following_the_model_while_waiting():
    # Parado, o servo continua obedecendo: da para ver que o modelo esta vivo,
    # e o carro nao anda.
    loop, kw = _loop(lidar=GiraSempre(), clock=_relogio(), cruise_us=CRUZEIRO,
                     arranque_s=5.0)
    _roda(loop, 5)
    assert kw["actuator"].servo_history[-1] != STEER_CENTER_US
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US


def test_a_backlogged_read_is_flagged():
    loop, _ = _loop(lidar=GiraSempre(fila_no_passo=5), clock=_relogio())
    teles = _roda(loop, 6)
    assert teles[4]["voltas"] == 4
    assert teles[4]["lidar_atrasado"] is True
    assert teles[5]["lidar_atrasado"] is False


def test_a_backlog_before_release_restarts_the_wait():
    # Sem a fila, liberaria em ~100,60 s. A fila no passo 8 (100,40) zera a
    # contagem: o carro so pode andar meio segundo DEPOIS de o LiDAR estar em dia.
    loop, kw = _loop(lidar=GiraSempre(fila_no_passo=8), clock=_relogio(),
                     cruise_us=CRUZEIRO, arranque_s=0.5)
    teles = _roda(loop, 24)
    esc = kw["actuator"].esc_history
    t_fila = teles[7]["t"]
    primeiro_andando = next(t["t"] for t, e in zip(teles, esc) if e == CRUZEIRO)
    assert primeiro_andando - t_fila > 0.5 - 1e-6


def test_after_release_a_backlog_does_not_stop_the_car():
    # Depois de liberado, ler a fila inteira deixa o vetor com a volta MAIS
    # NOVA; parar por isso seria parar a toa. As checagens de sempre seguem.
    loop, kw = _loop(lidar=GiraSempre(fila_no_passo=20), clock=_relogio(),
                     cruise_us=CRUZEIRO, arranque_s=0.3)
    teles = _roda(loop, 21)
    assert teles[19]["lidar_atrasado"] is True
    assert teles[19]["liberado"] is True
    assert kw["actuator"].esc_history[19] == CRUZEIRO


def test_without_the_gate_the_old_behaviour_holds():
    loop, kw = _loop(lidar=GiraSempre(), clock=_relogio(), cruise_us=CRUZEIRO)
    _roda(loop, 2)
    assert kw["actuator"].esc_history[-1] == CRUZEIRO


# ---------------------------------------------------------------------------
# Fim de pista (2026-10-09). Uma pista aberta termina numa parede: o carro para
# quando nao sobra direcao livre a frente, e fica parado (trava).
# ---------------------------------------------------------------------------

class Beco(GiraSempre):
    """Volta a 0.5 m ate o passo ``ate``; dali em diante, parede a ``perto`` m
    em TODAS as direcoes; e, a partir de ``abre``, tudo livre de novo."""

    def __init__(self, ate=4, perto=0.3, abre=None):
        GiraSempre.__init__(self)
        self.ate, self.perto, self.abre = ate, perto, abre

    def read_points(self):
        pts = GiraSempre.read_points(self)
        if self.abre is not None and self.passo >= self.abre:
            return [(a, 2.0) for a, _ in pts]
        if self.passo >= self.ate:
            return [(a, self.perto) for a, _ in pts]
        return pts


def test_the_dead_end_is_off_by_default():
    # O MVP no oval roda sem a regra: nada muda se ninguem pedir.
    loop, kw = _loop(lidar=Beco(ate=3), clock=_relogio(), cruise_us=CRUZEIRO)
    teles = _roda(loop, 6)
    assert kw["actuator"].esc_history[-1] == CRUZEIRO
    assert teles[-1]["fim"] is False


def test_a_dead_end_stops_the_car():
    loop, kw = _loop(lidar=Beco(ate=4), clock=_relogio(), cruise_us=CRUZEIRO,
                     beco_dist_m=0.5)
    teles = _roda(loop, 7)
    esc = kw["actuator"].esc_history
    # A volta lida no passo 4 so fecha no wrap seguinte.
    i = next(k for k, t in enumerate(teles) if t["fim"])
    assert i == 4
    assert all(e == CRUZEIRO for e in esc[1:i])    # parede a 0.5 m: ainda ha saida
    assert all(e == ESC_NEUTRAL_US for e in esc[i:])


def test_the_dead_end_stop_holds_even_if_the_reading_clears():
    # Uma leitura que oscila para "livre" nao pode soltar o carro contra a parede.
    loop, kw = _loop(lidar=Beco(ate=4, abre=6), clock=_relogio(), cruise_us=CRUZEIRO,
                     beco_dist_m=0.5)
    teles = _roda(loop, 9)
    assert teles[-1]["fim"] is True
    assert kw["actuator"].esc_history[-1] == ESC_NEUTRAL_US


def test_the_servo_keeps_following_the_model_at_the_end():
    loop, kw = _loop(lidar=Beco(ate=4), clock=_relogio(), cruise_us=CRUZEIRO,
                     beco_dist_m=0.5)
    _roda(loop, 6)
    assert kw["actuator"].servo_history[-1] != STEER_CENTER_US


def test_the_end_is_logged_apart_from_the_emergency_stop():
    loop, kw = _loop(lidar=Beco(ate=4), clock=_relogio(), cruise_us=CRUZEIRO,
                     beco_dist_m=0.5)
    _roda(loop, 6)
    ultimo = kw["logger"].frames[-1]
    assert ultimo["fim"] is True
    assert ultimo["blocked"] is False             # 0.3 m nao e emergencia (0.25)
