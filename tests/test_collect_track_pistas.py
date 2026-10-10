"""Quantos episodios cada pista recebe na coleta (2026-10-09)."""
import pytest

pytest.importorskip("carla")
from ai.collect_track import pista_e_episodios  # noqa: E402


def test_a_star_gives_one_track_its_own_episode_count():
    assert pista_e_episodios("oval_tcc*8", 2) == ("oval_tcc", 8)


def test_without_a_star_the_default_holds():
    assert pista_e_episodios("grade:SDSES", 2) == ("grade:SDSES", 2)


def test_nonsense_counts_are_refused():
    with pytest.raises(ValueError):
        pista_e_episodios("oval_tcc*0", 2)
    with pytest.raises(ValueError):
        pista_e_episodios("*3", 2)
