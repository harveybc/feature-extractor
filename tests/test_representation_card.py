"""Further behaviour of app.representation_card beyond FS17 (standard library only)."""
import pytest

from app.representation_card import CardRefusal, validate_card


def card(**overrides):
    base = {"schema": "representation_candidate_card.v1", "candidate_id": "ae-control-001",
            "family": "autoencoder_control",
            "corpus": {"kind": "TRAIN_ONLY", "pretrained_weights_source": None, "train_folds": ["f1"]},
            "latent": {"layout": "temporal", "time_steps": 24, "channels": 16, "grid_adapter": None},
            "conditioning_contract": "OPERATIONAL",
            "evaluation": {"reconstruction": {"state": "MEASURED", "mae_z": 0.1, "mse_z": 0.02}, "probes": []}}
    base.update(overrides)
    return base


def test_patch_latent_needs_a_declared_adapter():
    patch = {"layout": "patch", "time_steps": 5, "channels": 32, "grid_adapter": None}
    with pytest.raises(CardRefusal, match="^GRID_ADAPTER_UNDECLARED"):
        validate_card(card(latent=patch))
    patch["grid_adapter"] = {"name": "patch_to_step", "validated_by": "tests/test_fs04.py::test_patch_adapter"}
    out = validate_card(card(latent=patch))
    assert out["temporal_contract_state"] == "ADAPTER_DECLARED"
    assert out["admissibility"]["verdict"] == "ADMISSIBLE_WITH_ADAPTER"


def test_temporal_latent_off_the_common_grid_is_not_satisfied():
    off = {"layout": "temporal", "time_steps": 72, "channels": 24, "grid_adapter": None}
    out = validate_card(card(latent=off))
    assert out["temporal_contract_state"] == "NOT_EVALUATED"
    with pytest.raises(CardRefusal, match="^GRID_MISMATCH_WITHOUT_ADAPTER"):
        validate_card(card(latent=off, temporal_contract_state="SATISFIED"))


def test_probe_rows_derive_and_check_delta_and_preservation():
    row = {"target": "Y_s", "horizon": 1, "fold": "f1", "seed": 0, "loss_trained": 0.8, "loss_random": 1.0,
           "loss_raw": 0.9, "naive": 1.1}
    out = validate_card(card(evaluation={"reconstruction": {"state": "NOT_APPLICABLE"}, "probes": [row]}))
    got = out["evaluation"]["probes"][0]
    assert got["delta_probe"] == pytest.approx(0.2) and got["preservation"] == pytest.approx(0.1)
    bad = dict(row, delta_probe=0.5)
    with pytest.raises(CardRefusal, match="^DELTA_PROBE_INCONSISTENT"):
        validate_card(card(evaluation={"reconstruction": {"state": "NOT_APPLICABLE"}, "probes": [bad]}))
    with pytest.raises(CardRefusal, match="^PROBE_TARGET_INVALID"):
        validate_card(card(evaluation={"reconstruction": {"state": "NOT_APPLICABLE"},
                                       "probes": [dict(row, target="self_forecast")]}))


def test_generative_is_secondary_and_failed_reconstruction_is_not_admissible():
    assert validate_card(card(family="generative_cvae"))["admissibility"]["verdict"] == "SECONDARY"
    failed = card(evaluation={"reconstruction": {"state": "FAILED"}, "probes": []})
    out = validate_card(failed)
    assert out["admissibility"]["verdict"] == "NOT_ADMISSIBLE"
    assert "RECONSTRUCTION_FAILED" in out["admissibility"]["reasons"]


def test_foreign_weights_become_comparator_only_after_a_clean_audit():
    corpus = {"kind": "FOREIGN_PRETRAINED", "pretrained_weights_source": "org/model", "train_folds": [],
              "contamination_audit": "DONE_NO_OVERLAP"}
    out = validate_card(card(family="foundation_pretrained", corpus=corpus))
    assert out["admissibility"]["verdict"] == "ADMISSIBLE_AS_FOREIGN_COMPARATOR"
    corpus["contamination_audit"] = "DONE_OVERLAP_FOUND"
    assert validate_card(card(family="foundation_pretrained", corpus=corpus))["admissibility"]["verdict"] \
        == "NOT_ADMISSIBLE"


def test_complete_mode_requires_every_schema_field():
    with pytest.raises(CardRefusal, match="^CARD_INCOMPLETE"):
        validate_card(card(), complete=True)
