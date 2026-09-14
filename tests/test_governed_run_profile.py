"""The governed wrapper declares this repository's inputs, outputs and metrics."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GOV = Path(os.environ.get("DATA_GOV_CHECKOUT") or ROOT.parent / "data-gov")

pytestmark = pytest.mark.skipif(
    not (DATA_GOV / "tools" / "governed_exec.py").is_file(), reason="data-gov checkout not available"
)


def _load():
    spec = importlib.util.spec_from_file_location("feature_extractor_governed_run", ROOT / "tools" / "governed_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["feature_extractor_governed_run"] = module
    spec.loader.exec_module(module)
    return module


def test_profile_builds_a_valid_spec(tmp_path, capsys):
    lake = tmp_path / "lake" / "phase_1"
    lake.mkdir(parents=True)
    for n in (4, 5, 6):
        (lake / f"normalized_d{n}.csv").write_text("DATE_TIME,typical_price\n2013-01-01 00:00:00,1\n")
    config = tmp_path / "phase_4_2" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "x_train_file": str(lake / "normalized_d4.csv"), "y_train_file": str(lake / "normalized_d4.csv"),
        "x_validation_file": str(lake / "normalized_d5.csv"), "y_validation_file": str(lake / "normalized_d5.csv"),
        "x_test_file": str(lake / "normalized_d6.csv"), "y_test_file": str(lake / "normalized_d6.csv"),
        "save_encoder": "examples/results/encoder.keras", "save_decoder": "examples/results/decoder.keras",
        "loss_plot_file": "examples/results/loss.png", "save_log": "examples/results/debug_out.json",
        "save_config": "examples/results/config_out.json", "epochs": 1,
    }))
    module = _load()
    assert module.main([
        "--load_config", str(config), "--experiment-key", "fext-001", "--lake", "predictor_examples",
        "--lake-root", str(tmp_path / "lake"), "--out-dir", str(tmp_path / "out"), "--print-spec",
    ]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["project"] == "feature-extractor"
    assert [d["role"] for d in spec["datasets"]] == ["x_train_file", "y_train_file", "x_validation_file",
                                                     "y_validation_file", "x_test_file", "y_test_file"]
    assert {d["resource"] for d in spec["datasets"]} == {f"phase_1/normalized_d{n}.csv" for n in (4, 5, 6)}
    assert spec["metrics"] == {"kind": "json_numbers", "path": "debug_out.json", "keys": module.METRIC_KEYS}
    assert "save_encoder" in spec["output_keys"] and spec["artifacts"]["encoder"] == "save_encoder"
    assert spec["tags"]["phase"] == "phase_4_2"
