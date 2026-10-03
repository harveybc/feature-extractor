"""The port to predictor's current preprocessor API, declared as rules.

Owner's decision, 2026-09-14: feature-extractor is ported to
`run_preprocessing(self, target_plugin, config)`; it no longer waits on a decision and no
predictor commit is pinned. These rules pin the behaviour of the call site instead.
"""

from __future__ import annotations

import pytest

from app.preprocessing_api import PreprocessorApiError, run_preprocessing, takes_target_plugin


class CurrentApi:
    """predictor >= 9b7d611."""

    def __init__(self):
        self.seen = None

    def run_preprocessing(self, target_plugin, config):
        self.seen = (target_plugin, config)
        return {"x_train": [1], "feature_names": ["a"]}


class LegacyApi:
    """The shape feature-extractor called before the drift."""

    def __init__(self):
        self.seen = None

    def run_preprocessing(self, config):
        self.seen = config
        return {"x_train": [2], "feature_names": ["b"]}


class VariadicApi:
    def run_preprocessing(self, *args):
        return {"args": args}


class NoApi:
    pass


class Target:
    plugin_params = {"target_column": "CLOSE"}

    def set_params(self, **kwargs):
        self.params = kwargs


def test_the_current_api_receives_the_target_plugin_first():
    plugin, target, config = CurrentApi(), Target(), {"window_size": 8}
    result = run_preprocessing(plugin, config, target)
    assert plugin.seen == (target, config)
    assert result["x_train"] == [1]


def test_the_legacy_api_is_still_called_with_the_configuration_alone():
    plugin, config = LegacyApi(), {"window_size": 8}
    assert run_preprocessing(plugin, config, Target())["x_train"] == [2]
    assert plugin.seen == config


def test_a_missing_target_plugin_names_the_configuration_key():
    with pytest.raises(PreprocessorApiError, match="target_plugin"):
        run_preprocessing(CurrentApi(), {"window_size": 8})


def test_a_plugin_without_the_method_is_refused_by_name():
    with pytest.raises(PreprocessorApiError, match="NoApi has no run_preprocessing"):
        run_preprocessing(NoApi(), {})


def test_the_signature_check_answers_for_every_shape():
    assert takes_target_plugin(CurrentApi().run_preprocessing) is True
    assert takes_target_plugin(LegacyApi().run_preprocessing) is False
    assert takes_target_plugin(VariadicApi().run_preprocessing) is True


def test_the_predictor_plugin_this_repository_uses_declares_the_current_api():
    """Not a mock: the installed provider of `stl_preprocessor`, whichever it is."""
    entry_points = pytest.importorskip("importlib.metadata").entry_points
    matches = [e for e in entry_points(group="preprocessor.plugins") if e.name == "stl_preprocessor"]
    if not matches:
        pytest.skip("stl_preprocessor is not installed in this environment")
    plugin = matches[0].load()()
    assert takes_target_plugin(plugin.run_preprocessing) is True
