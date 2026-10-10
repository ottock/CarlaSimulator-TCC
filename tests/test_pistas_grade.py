"""Catalogo das pistas da grade 4 x 3 (2026-10-09)."""
import pytest

from ai.pistas_grade import (enumera, espelho, formatos, pecas, respeita_a_regra,
                             reverso, separa, tipos, variantes_curva_final)
from core.carlaClient.track_builder import _OVAL_TCC


def test_the_grid_holds_three_loops_and_227_open_routes():
    lacos, abertos = enumera()
    assert len(lacos) == 3
    assert len(abertos) == 227


def test_two_closed_shapes_and_104_open_ones():
    fechados, abertos = formatos()
    assert len(fechados) == 2
    assert len(abertos) == 104


def test_the_ten_piece_loop_is_our_oval():
    # O oval do MVP: o gemeo monta as mesmas pecas, so que no sentido anti-horario.
    fechados, _ = formatos()
    oval = [c for c in fechados if len(c) == 10][0]
    alvo = pecas(espelho(oval))
    rotacoes = [alvo[i:] + alvo[:i] for i in range(len(alvo))]
    assert list(_OVAL_TCC) in rotacoes


def test_no_shape_has_two_curves_in_a_row():
    fechados, abertos = formatos(curva_final=True)
    assert all(respeita_a_regra(c, True) for c in fechados)
    assert all(respeita_a_regra(c, False) for c in abertos)


def test_open_routes_come_in_both_directions():
    _, abertos = formatos()
    assert all(reverso(c) in abertos for c in abertos)


def test_turn_direction_follows_the_right_hand_rule():
    # Descendo a coluna 0 e virando para a coluna 1: para quem desce, a coluna
    # da direita da grade fica a ESQUERDA.
    assert tipos([(0, 0), (1, 0), (1, 1)], False) == "SES"     # pontas sao retas
    assert tipos([(0, 0), (1, 0), (1, 1), (1, 2)], False) == "SESS"
    assert tipos([(0, 1), (1, 1), (1, 0), (1, -1)], False) == "SDSS"


def test_a_route_may_end_in_a_curve_only_after_a_straight():
    assert variantes_curva_final("SDSS") == ["SDSE", "SDSD"]
    assert variantes_curva_final("SSDS") == []                # penultima e curva


def test_a_real_straight_is_two_twin_pieces_of_53_cm():
    assert pecas("SES") == ["tcc_reta_branca", "tcc_reta_cinza", "tcc_curva90",
                            "tcc_reta_branca", "tcc_reta_cinza"]
    assert pecas("D") == ["tcc_curva90_r"]
    with pytest.raises(ValueError):
        pecas("SX")


def test_the_split_keeps_a_track_and_its_mirror_together():
    _, abertos = formatos()
    treino, teste = separa(abertos, frac_teste=0.2, semente=3)
    assert set(treino).isdisjoint(teste)
    assert sorted(treino + teste) == sorted(abertos)
    assert all(espelho(c) in teste for c in teste if espelho(c) in abertos)
    assert all(reverso(c) in teste for c in teste if reverso(c) in abertos)
    assert 0.1 < len(teste) / len(abertos) < 0.3


def test_the_split_is_deterministic():
    _, abertos = formatos()
    assert separa(abertos, semente=1) == separa(abertos, semente=1)


def test_groups_expand_and_the_test_tracks_stay_out_of_training():
    from ai.pistas_grade import expande_pistas
    treino, teste = expande_pistas("grade:treino", semente=0)
    assert set(treino).isdisjoint(teste)
    assert "grade:DSDSSDSDSS" in treino and "grade:DSDSDSDS" in treino
    todas, _ = expande_pistas("grade:todas")
    assert sorted(set(treino) | set(teste)) == sorted(todas)


def test_plain_presets_and_codes_pass_through_once():
    from ai.pistas_grade import expande_pistas
    pistas, _ = expande_pistas("oval_tcc, grade:SDSES,oval_tcc")
    assert pistas == ["oval_tcc", "grade:SDSES"]
