"""Geometria das pistas custom (matematica pura, sem CARLA).

Garante que as pistas do estande FECHAM (o laco volta ao inicio) — em especial a
pista1, cujo residuo de fechamento escala com o `fator`: o `fechar_gap` precisa
distribuir esse residuo tambem em escala real (nao so a 1/12).
"""
import math

from core.carlaClient import track_builder as tb


def _centerline_closure(preset, fator):
    """Distancia entre o ULTIMO e o PRIMEIRO waypoint da linha de centro densa."""
    wps = tb.gerar_waypoints(preset, fator, 0.5)
    return math.hypot(wps[-1]["x"] - wps[0]["x"], wps[-1]["y"] - wps[0]["y"])


def test_stand_tracks_close_at_real_scale():
    # Em escala real (fator 12) as 3 pistas do estande fecham (< 15 cm).
    for preset in ("pista1", "pista2", "pista3"):
        gap = _centerline_closure(preset, 12.0)
        assert gap < 0.15, "%s nao fecha em escala real: gap=%.3f m" % (preset, gap)


def test_pista1_closes_both_scales():
    # pista1 fecha tanto a 1/12 quanto em escala real (o fix do limiar por fator).
    assert _centerline_closure("pista1", 1.0) < 0.05
    assert _centerline_closure("pista1", 12.0) < 0.15


def test_geometric_presets_close():
    for preset in ("oval", "quadrado", "octogono"):
        assert _centerline_closure(preset, 12.0) < 0.05


def test_curva_esquerda_vs_direita():
    # A mesma malha vira p/ lados opostos conforme o conector (esq +90, dir -90).
    c = tb._conectores(1.0)
    dh_esq = c["tcc_curva90"]["h_out"] - c["tcc_curva90"]["h_in"]
    dh_dir = c["tcc_curva90_r"]["h_out"] - c["tcc_curva90_r"]["h_in"]
    assert dh_esq > 0 and dh_dir < 0


def test_waypoints_tem_campos_e_densidade():
    wps = tb.gerar_waypoints("pista2", 12.0, 0.5)
    assert len(wps) > 50
    assert all({"x", "y", "heading", "s"} <= set(w) for w in wps)
    assert wps[-1]["s"] > wps[0]["s"]  # distancia acumulada cresce


# ---------------------------------------------------------------------------
# Pistas da grade e o fim das pistas abertas (2026-10-09)
# ---------------------------------------------------------------------------

def test_a_grid_code_builds_the_same_oval_as_the_preset():
    # oval_tcc comecando no lado longo: S S E S E S S E S E
    assert tb._preset("grade:SSESESSESE") == list(tb._OVAL_TCC)


def test_loops_are_closed_and_routes_are_open():
    assert tb.pista_fechada("oval_tcc", 12.0)
    assert tb.pista_fechada("grade:SESESESE", 12.0)        # o quadrado 3 x 3
    assert not tb.pista_fechada("grade:SSESS", 12.0)


def test_an_open_route_never_gets_its_closing_gap_spread():
    # Sem a checagem de rumo, um fim perto do inicio entortaria a pista inteira.
    wps = tb.gerar_waypoints("grade:SESESES", 12.0, 0.5)
    hw = tb.LARGURA_PISTA * 12.0 / 2
    # sete pecas: reta + curva + ... ; a primeira reta tem de seguir exatamente o eixo x
    assert all(abs(w["y"]) < 1e-9 for w in wps if w["s"] < 2 * hw)


def test_the_end_wall_spans_the_lane_right_at_the_end():
    hw = tb.LARGURA_PISTA * 12.0 / 2
    seq = tb._preset("grade:SSS")                  # 3 retas em linha, rumo +x
    fim = tb.parede_do_fim(seq, 12.0, espessura=0.1)
    assert [n for n, *_ in fim] == ["tcc_reta_branca", "tcc_reta_cinza"]
    comprimento = 3 * 2 * hw                        # cada reta real = 2 hw
    lb = tb.COMPRIMENTO_RETA_BRANCA * 12.0
    lc = tb.COMPRIMENTO_RETA_CINZA * 12.0
    for nome, x, y, alpha in fim:
        # giradas 90 graus, com a face da parede de perto no fim da faixa
        assert math.isclose(math.cos(alpha), 0.0, abs_tol=1e-9)
        assert math.isclose(x, comprimento + hw + 0.1, abs_tol=1e-9)
    # lado a lado, cobrindo exatamente a largura da faixa
    ys = sorted((y - L / 2, y + L / 2) for (n, x, y, a), L in zip(fim, (lb, lc)))
    assert math.isclose(ys[0][0], -hw, abs_tol=1e-9)
    assert math.isclose(ys[1][1], hw, abs_tol=1e-9)
    assert math.isclose(ys[0][1], ys[1][0], abs_tol=1e-9)


def test_loops_get_no_end_wall():
    assert tb.parede_do_fim("oval_tcc", 12.0) == []


def test_no_end_wall_when_the_route_already_occupies_the_next_cell():
    # Espiral S S E S E S E S: a ultima reta aponta para a casa da primeira, que ja
    # tem a parede lateral virada para ca.
    assert tb.parede_do_fim("grade:SESES", 12.0) != []
    assert tb.parede_do_fim("grade:SESESES", 12.0) != []     # dobra, mas a casa esta livre
    assert tb.parede_do_fim("grade:SSESESES", 12.0) == []
