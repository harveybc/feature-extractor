"""FS17 of the progressive-selection subplan, written RED before any PS3-R code exists.

Subplan `FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md` section 10.4, FS17:

    An encoder without a decoder does not fail reconstruction; a pooled output does not satisfy the temporal
    contract; weights from a foreign corpus are not labelled TRAIN-only.

The missing mechanism is one module, `app.representation_card`, with `validate_card(card)` returning the card with
its derived states filled in, or raising `CardRefusal` whose message starts with a named code. Where the mechanism is
missing the test FAILS with a message beginning `MECHANISM_MISSING` -- never a skip.

One test runs against the installed code today without importing TensorFlow: it parses the `cnn` encoder plugin with
`ast` and asserts the plugin declares a temporal latent layout. It is red because the installed encoder declares no
layout at all: its two strided Conv1D layers emit a temporal output on a window_size/4 grid (288 -> 72 by default),
which is neither the recipe's 24-step common grid nor a declared adapter (its own docstring even calls the output "a
latent vector"). It turns green when a plugin declares `latent_layout = "temporal"`, no `Flatten` collapses time,
and any grid change is declared as an adapter. That is the AE control the shortlist
(`docs/PS3R_REPRESENTATION_SHORTLIST_2026_10_01.md`) needs before any family can be compared to it.

No model is built, no weights are loaded, nothing is trained. CPU only, standard library plus pytest.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

MECHANISM = "app.representation_card"
REPO = Path(__file__).resolve().parents[1]


def _mechanism(name):
    try:
        module = importlib.import_module(MECHANISM)
    except ImportError as trouble:
        pytest.fail(f"MECHANISM_MISSING: {MECHANISM} does not exist ({trouble}); FS17 and "
                    f"docs/PS3R_REPRESENTATION_SHORTLIST_2026_10_01.md section 0 specify it. This test turns green when "
                    f"`{MECHANISM}.{name}` exists.")
    attribute = getattr(module, name, None)
    if attribute is None:
        pytest.fail(f"MECHANISM_MISSING: {MECHANISM} has no `{name}`.")
    return attribute


def _card(**overrides):
    card = {
        "schema": "representation_candidate_card.v1",
        "candidate_id": "ts2vec-pilot-001",
        "family": "contrastive",
        "reference": {"paper_title": "TS2Vec: Towards Universal Representation of Time Series",
                      "paper_url": "https://arxiv.org/abs/2106.10466", "venue_year": "AAAI 2022",
                      "code_url": "https://github.com/zhihanyue/ts2vec",
                      "code_revision": "b0088e14a99706c05451316dc6db8d3da9351163", "licence": "MIT"},
        "corpus": {"kind": "TRAIN_ONLY", "pretrained_weights_source": None, "train_folds": ["fold_2015_2018"]},
        "latent": {"layout": "temporal", "time_steps": 24, "channels": 16, "grid_adapter": None},
        "conditioning_contract": "OPERATIONAL",
        "evaluation": {"reconstruction": {"state": "NOT_APPLICABLE"}, "probes": []},
    }
    card.update(overrides)
    return card


def test_fs17_encoder_without_decoder_does_not_fail_reconstruction():
    """A contrastive encoder has no decoder: reconstruction is NOT_APPLICABLE, a state, and the card is not refused."""
    validate = _mechanism("validate_card")
    card = validate(_card())
    assert card["evaluation"]["reconstruction"]["state"] == "NOT_APPLICABLE"
    assert card["admissibility"]["verdict"] != "NOT_ADMISSIBLE"
    assert "RECONSTRUCTION_FAILED" not in card["admissibility"]["reasons"]


def test_fs17_pooled_output_does_not_satisfy_the_temporal_contract():
    """A pooled latent can be declared, but its temporal_contract_state is VIOLATED_POOLED, never SATISFIED."""
    validate = _mechanism("validate_card")
    pooled = validate(_card(latent={"layout": "pooled", "time_steps": 1, "channels": 32, "grid_adapter": None}))
    assert pooled["temporal_contract_state"] == "VIOLATED_POOLED"
    temporal = validate(_card())
    assert temporal["temporal_contract_state"] == "SATISFIED"
    # a card that CLAIMS satisfaction for a pooled latent is refused by name
    CardRefusal = _mechanism("CardRefusal")
    with pytest.raises(CardRefusal) as trouble:
        validate(_card(latent={"layout": "pooled", "time_steps": 1, "channels": 32, "grid_adapter": None},
                       temporal_contract_state="SATISFIED"))
    assert "POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT" in str(trouble.value)


def test_fs17_foreign_corpus_weights_are_not_labelled_train_only():
    """Weights pretrained elsewhere make the corpus FOREIGN_PRETRAINED; a TRAIN_ONLY label over them is refused."""
    validate = _mechanism("validate_card")
    CardRefusal = _mechanism("CardRefusal")
    with pytest.raises(CardRefusal) as trouble:
        validate(_card(family="foundation_pretrained",
                       corpus={"kind": "TRAIN_ONLY", "pretrained_weights_source": "AutonLab/MOMENT-1-small",
                               "train_folds": ["fold_2015_2018"]}))
    assert "FOREIGN_CORPUS_WEIGHTS_NOT_TRAIN_ONLY" in str(trouble.value)
    foreign = validate(_card(family="foundation_pretrained",
                             corpus={"kind": "FOREIGN_PRETRAINED", "pretrained_weights_source": "AutonLab/MOMENT-1-small",
                                     "train_folds": []}))
    assert foreign["corpus"]["kind"] == "FOREIGN_PRETRAINED"
    assert foreign["admissibility"]["verdict"] in ("NOT_ADMISSIBLE", "ADMISSIBLE_AS_FOREIGN_COMPARATOR")


def test_fs17_installed_cnn_encoder_declares_temporal_layout():
    """RED TODAY against the installed code: the cnn encoder flattens time and declares no latent layout."""
    source = (REPO / "app" / "plugins" / "encoder_plugin_cnn.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    plugin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Plugin")
    declared = {}
    for node in plugin.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "plugin_params" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant):
                        declared[key.value] = getattr(value, "value", None)
    flattens = [node for node in ast.walk(plugin) if isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "Flatten"]
    strided = [node for node in ast.walk(plugin) if isinstance(node, ast.Call)
               and getattr(node.func, "id", getattr(node.func, "attr", None)) == "Conv1D"
               and any(k.arg == "strides" and isinstance(k.value, ast.Constant) and k.value.value != 1
                       for k in node.keywords)]
    assert declared.get("latent_layout") == "temporal" and not flattens and (
        not strided or declared.get("grid_adapter")), (
        "MECHANISM_MISSING: the installed cnn encoder declares plugin_params keys "
        f"{sorted(declared)} (no `latent_layout`, no `grid_adapter`), calls Flatten {len(flattens)} time(s) and "
        f"applies {len(strided)} strided Conv1D layer(s), so its output is temporal but on a grid of window_size/4 "
        "steps (288 -> 72 by default), not the 24-step common grid of the recipe, and nothing declares that. FS17 and "
        "the subplan (section 6.1: the base keeps the full window until fusion on a common grid) need an AE control "
        "that declares latent_layout='temporal' and either preserves the grid or declares and validates its adapter "
        "(FS04). This turns green when the plugin declares those keys and no Flatten collapses time.")
