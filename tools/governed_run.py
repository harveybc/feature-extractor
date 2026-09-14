#!/usr/bin/env python3
"""Governed feature-extractor run (data-gov Flow v3).

Declares what this repository consumes and produces; the protocol itself lives
in data-gov (`tools/governed_exec.py`): campaign before data, governed download
with hash confirmation, fresh output namespace, command on CPU, terminal
COMPLETED | FAILED | INCONCLUSIVE | REFUSED through a durable outbox, then
reconciliation. A result that is not reconciled is not governing.

Inputs are the six predictor-style keys (`x_train_file` ... `y_test_file`);
outputs (encoder, decoder, plots, logs, effective config) are redirected under
the output directory. Metrics come from the numeric leaves of `save_log`
restricted to the documented keys; a run whose log lacks them reports no
metric rather than inventing one.

The data-gov checkout is found through DATA_GOV_CHECKOUT or the sibling
directory `../data-gov`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ["final_training_mae_logged", "final_validation_mae_logged", "execution_time", "execution_time_seconds"]


def _governed_exec():
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT") or REPO_ROOT.parent / "data-gov").expanduser()
    path = checkout / "tools" / "governed_exec.py"
    if not path.is_file():
        raise SystemExit(f"governed_run: data-gov checkout not found at {checkout} (set DATA_GOV_CHECKOUT)")
    spec = importlib.util.spec_from_file_location("data_gov_governed_exec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_gov_governed_exec"] = module
    spec.loader.exec_module(module)
    return module


def _metrics(config: dict) -> dict:
    log = Path(str(config.get("save_log") or "debug_out.json")).name
    return {"kind": "json_numbers", "path": log, "keys": METRIC_KEYS}


PROFILE = {
    "project": "feature-extractor",
    "input_keys": ["x_train_file", "y_train_file", "x_validation_file", "y_validation_file",
                   "x_test_file", "y_test_file"],
    "output_keys": ["save_encoder", "save_decoder", "loss_plot_file", "save_log", "save_config", "results_file",
                    "output_file", "uncertainties_file", "model_plot_file", "predictions_plot_file",
                    "stl_plot_file", "wavelet_plot_file", "tapper_plot_file", "optimizer_output_file"],
    "command": [sys.executable, "app/main.py", "--load_config", "{config}"],
    "metrics": _metrics,
    "artifacts": {"encoder": "save_encoder", "decoder": "save_decoder", "debug_log": "save_log",
                  "effective_config": "save_config", "results": "results_file"},
    "tags": {},
}


def main(argv=None) -> int:
    return _governed_exec().consumer_main(PROFILE, argv, repo_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
