"""Frozen selected-set training admission, without executing a GPU child."""

import hashlib

import pytest

from app import fs4_extractibility as X
from tools.i5p_train_selected import claim_for, frozen_members, terminal_is_valid


def test_freeze_and_claim_bind_the_selected_feature_set():
    body = {"winners": {"EURUSD": {"Y_s_1h": {"RAW": {"members": ["a", "b"]}}}}}
    freeze = {**body, "freeze_sha256": X.digest(body), "frozen_utc": "later"}
    assert frozen_members(freeze, "EURUSD", "Y_s_1h") == ("a", "b")
    claim = claim_for("a", "EURUSD", "corpus", "inner_2023", 0)
    assert claim["task_id"] == X.task_digest(claim)
    assert claim["task_id"] != claim_for("b", "EURUSD", "corpus", "inner_2023", 0)["task_id"]
    freeze["winners"]["EURUSD"]["Y_s_1h"]["RAW"]["members"].append("c")
    with pytest.raises(ValueError, match="FREEZE_DIGEST_MISMATCH"):
        frozen_members(freeze, "EURUSD", "Y_s_1h")


def test_retained_terminal_requires_real_weights_and_v2(tmp_path):
    claim = claim_for("a", "EURUSD", "corpus", "inner_2023", 0)
    weights = tmp_path / "chosen.weights.h5"
    weights.write_bytes(b"weights")
    result = {"status": "COMPLETE", "task_id": claim["task_id"], "arm": claim["arm"],
              "architecture": {"last_latent_lag_rows": 0},
              "artifacts": {"chosen_weights_file": str(weights),
                            "chosen_weights_file_sha256": hashlib.sha256(b"weights").hexdigest()}}
    import json
    terminal = tmp_path / "result.json"
    terminal.write_text(json.dumps(result))
    assert terminal_is_valid(terminal, claim)
    weights.write_bytes(b"changed")
    assert not terminal_is_valid(terminal, claim)
