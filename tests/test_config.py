"""Tests for settings validation, focused on optional sections.

The camera block must be validated when present (so typos are caught) but not
required (track-only configs and the existing files have no camera block).
"""
from pathlib import Path

import pytest

from utils.config import _validate, _Optional, load_settings

_ROOT = Path(__file__).resolve().parents[1]


def test_required_key_still_enforced():
    with pytest.raises(ValueError):
        _validate({}, {"a": int})


def test_wrong_type_still_rejected():
    with pytest.raises(TypeError):
        _validate({"a": "nope"}, {"a": int})


def test_optional_section_absent_is_ok():
    _validate({"a": 1}, {"a": int, "cam": _Optional({"x": int})})


def test_optional_section_present_is_validated():
    schema = {"cam": _Optional({"x": int})}
    with pytest.raises(TypeError):
        _validate({"cam": {"x": "not-an-int"}}, schema)


def test_optional_leaf_absent_is_ok():
    _validate({"cam": {}}, {"cam": _Optional({"x": _Optional(int)})})


def test_existing_settings_files_still_load():
    load_settings(str(_ROOT / "settings" / "baseSettings.json"))
    load_settings(str(_ROOT / "settings" / "pistaTCC.json"))
