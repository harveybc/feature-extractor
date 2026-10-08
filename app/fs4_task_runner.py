"""Phase-4 extractibility task runner (order SATOSHI_FS4_EXECUTION_2026_10_07, step 1).

Contract:  stdin = the controller's claim JSON (tools/fs4_campaign.py claim; also $FS4_CLAIM_JSON or --claim-file,
           because crispdm-run gives its child /dev/null as stdin);
           stdout = exactly one JSON document;
           stderr = logs.
  exit 0   status COMPLETE, accepted by tools/fs4_campaign.py::_validate_result (mirrored below);
  exit 3   typed refusal, e.g. {"status": "NOT_AVAILABLE_FOR_TRAIN", "task_id": ..., "reason": ...}
           (no TRAIN observations, unknown feature, corpus digest mismatch); never a fabricated number;
  exit 4   TRAINED_ENCODER could not verify the physical GPU in process (no CPU fallback);
  exit 1   any other error.

Only the claimed arm and fold run.  Inputs are the governed TRAIN parquets pinned in
app.fs4_extractibility.CORPORA, resolved through --input ROLE=PATH (or FS4_INPUT_<ROLE>).
A terminal retained under --output-root/<task_id>/result.json is re-emitted on restart without
retraining (FS4-07).  `--coverage FEATURES.json` reports, per population/feature/fold, whether a
task can run, without training anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Dict, Optional

import numpy as np

from app import fs4_extractibility as X
from app import univariate_temporal as U
from app.univariate_temporal_pilot import _cgroup_peak_bytes, _git_commit

TASK_SCHEMA = "fs4.extractibility.task.v1"
RESULT_SCHEMA = "fs4.extractibility.result.v1"
CODE_FILES = ("app/fs4_task_runner.py", "app/fs4_extractibility.py", "app/univariate_temporal.py",
              "app/univariate_temporal_pilot.py", "app/npz_encoder_adapter.py")
EXIT_REFUSED, EXIT_GPU = 3, 4


def log(msg: str) -> None:
    print(f"[fs4_task_runner] {msg}", file=sys.stderr, flush=True)


def code_sha256() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for rel in CODE_FILES:
        with open(os.path.join(root, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def validate_terminal(result: dict, payload: dict) -> None:
    """Mirror of tools/fs4_campaign.py::_validate_result (predictor @b11de2ef)."""
    if result.get("status") != "COMPLETE" or result.get("task_id") != X.task_digest(payload):
        raise X.Refusal("RESULT_IDENTITY_OR_STATUS_MISMATCH")
    if result.get("seed") != payload["seed"]:
        raise X.Refusal("SEED_MISMATCH")
    for key in ("input_sha256", "code_sha256", "model_sha256", "rows_sha256", "mask_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise X.Refusal(f"INVALID_{key.upper()}")
    if type(result.get("population_n")) is not int or result["population_n"] < 1:
        raise X.Refusal("INVALID_POPULATION")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or "mae" not in metrics or "naive_mae" not in metrics:
        raise X.Refusal("MISSING_PAIRED_METRICS")
    import math
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in metrics.values()):
        raise X.Refusal("INVALID_METRIC")
    X.canonical(result)


def atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(X.canonical(obj))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- GPU verification
def _nvidia_smi_names() -> Dict[str, str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30, check=False)
    if out.returncode:
        raise X.Refusal("GPU_NOT_VERIFIED", f"nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
    names = {}
    for line in out.stdout.splitlines():
        if "," in line:
            uuid, name = line.split(",", 1)
            names[uuid.strip()] = name.strip()
    return names


def verify_gpu(expected_uuid: Optional[str]) -> dict:
    """In-process proof that TensorFlow sees exactly the requested physical GPU and places ops on it."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = expected_uuid or cvd
    if not uuid.startswith("GPU-") or cvd != uuid:
        raise X.Refusal("GPU_NOT_VERIFIED", f"CUDA_VISIBLE_DEVICES={cvd!r} must be the physical UUID {uuid!r}")
    names = _nvidia_smi_names()
    if uuid not in names:
        raise X.Refusal("GPU_NOT_VERIFIED", f"{uuid} not listed by nvidia-smi: {sorted(names)}")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise X.Refusal("GPU_NOT_VERIFIED", f"TensorFlow sees {len(gpus)} GPUs, expected exactly 1")
    details = tf.config.experimental.get_device_details(gpus[0])
    tf_name = str(details.get("device_name", ""))
    if tf_name.strip().lower() != names[uuid].strip().lower():
        raise X.Refusal("GPU_NOT_VERIFIED", f"TensorFlow device {tf_name!r} != nvidia-smi {names[uuid]!r} for {uuid}")
    with tf.device("/GPU:0"):
        c = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    if "GPU:0" not in c.device:
        raise X.Refusal("GPU_NOT_VERIFIED", f"probe op placed on {c.device}")
    return {"uuid": uuid, "nvidia_smi_name": names[uuid], "tensorflow_device_name": tf_name,
            "compute_capability": list(details.get("compute_capability", ())) or None,
            "probe_device": c.device}


def variable_device(model) -> str:
    v = model.weights[0]
    inner = getattr(v, "value", None)
    dev = getattr(inner, "device", None) or getattr(v, "device", None)
    return str(dev or "")


# --------------------------------------------------------------------------- task
def _inputs_from(args) -> Dict[str, str]:
    inputs = {}
    for key, value in os.environ.items():
        if key.startswith("FS4_INPUT_"):
            inputs[key[len("FS4_INPUT_"):].lower()] = value
    for item in args.input or []:
        if "=" not in item:
            raise X.Refusal("BAD_INPUT_ARG", item)
        role, path = item.split("=", 1)
        inputs[role.strip().lower()] = path
    return inputs


def _registry(args) -> Optional[dict]:
    if not args.corpus_registry:
        return None
    with open(args.corpus_registry) as f:
        return json.load(f)


def read_claim_text(stream, claim_file: Optional[str] = None) -> str:
    """Claim JSON from --claim-file, else stdin when it carries data, else $FS4_CLAIM_JSON.

    crispdm-run starts the job with a trailing `&`, so a non-interactive shell hands the job
    /dev/null as stdin; the environment variable survives the transient scope."""
    if claim_file:
        with open(claim_file) as f:
            return f.read()
    text = ""
    try:
        text = stream.read()
    except (OSError, ValueError):
        text = ""
    if text.strip():
        return text
    env = os.environ.get("FS4_CLAIM_JSON", "")
    if env.strip():
        return env
    raise X.Refusal("CLAIM_MISSING", "no claim on stdin, --claim-file or FS4_CLAIM_JSON")


def load_claim(stream, claim_file: Optional[str] = None) -> dict:
    try:
        claim = json.loads(read_claim_text(stream, claim_file))
    except json.JSONDecodeError as exc:
        raise X.Refusal("CLAIM_NOT_JSON", str(exc)) from exc
    for key in ("schema", "population_id", "identity", "feature_id", "fold_id", "arm", "seed", "task_id"):
        if key not in claim:
            raise X.Refusal("CLAIM_MISSING_FIELD", key)
    if claim["schema"] != TASK_SCHEMA:
        raise X.Refusal("CLAIM_SCHEMA", str(claim["schema"]))
    if claim["arm"] not in X.ARMS + X.V2_ARMS:
        raise X.Refusal("CLAIM_ARM", str(claim["arm"]))
    if type(claim["seed"]) is not int:
        raise X.Refusal("CLAIM_SEED", repr(claim["seed"]))
    if X.task_digest(claim) != claim["task_id"]:
        raise X.Refusal("TASK_IDENTITY_MISMATCH", "task_id is not the digest of the claim payload")
    return claim


def prepare(claim: dict, corpus: X.Corpus, hp: X.Hyper) -> dict:
    """Arm-independent material: grid, fold, origins, normalizer, scoring batch, hidden mask, naive."""
    if corpus.population_id != claim["population_id"]:
        raise X.Refusal("POPULATION_IDENTITY_MISMATCH", f"{corpus.population_id} != {claim['population_id']}")
    values, role, column_sha = corpus.feature(claim["feature_id"])
    ts_rows = corpus.timestamps()
    grid = X.to_grid(ts_rows, values)
    fold = X.fold_spec(claim["fold_id"], ts_rows)
    U.validate_fold(fold, grid.ts, int(ts_rows[-1]), hp.window)
    fit_o = X.origins(grid, fold.fit, hp)
    score_o = X.origins(grid, fold.val, hp)
    if score_o.size == 0 or fit_o.size == 0:
        raise X.Refusal("NO_TRAIN_OBSERVATIONS",
                        f"{claim['feature_id']} {fold.fold_id}: fit origins {int(fit_o.size)}, scoring origins {int(score_o.size)}")
    if score_o.size < hp.min_scoring_windows:
        raise X.Refusal("INSUFFICIENT_TRAIN_SCORING_WINDOWS",
                        f"{claim['feature_id']} {fold.fold_id}: {int(score_o.size)} scoring origins "
                        f"< {hp.min_scoring_windows} (observed TRAIN rows in the validation year)")
    if fit_o.size < hp.min_fit_windows:
        raise X.Refusal("INSUFFICIENT_TRAIN_FIT_WINDOWS",
                        f"{claim['feature_id']} {fold.fold_id}: {int(fit_o.size)} fit origins < {hp.min_fit_windows}")
    norm = U.Normalization.fit(grid.x, grid.observed, U.covered_indices(fit_o, hp.window))
    cal = U.calendar_features(grid.ts)
    score_b = U.make_windows(grid.ts, grid.x, grid.observed, score_o, hp.window, norm, calendar=cal)
    identity = {"population_id": claim["population_id"], "identity": claim["identity"],
                "feature_id": claim["feature_id"], "fold_id": claim["fold_id"], "seed": claim["seed"]}
    hidden = X.hidden_mask(score_b.observed_mask, X._seed_from({**identity, "purpose": "scoring_mask"}),
                           hp.hide_fraction)
    visible = (score_b.observed_mask[..., 0] > 0) & ~hidden
    naive = X.naive_prediction(score_b.signal, visible)
    hours = ((score_b.anchor_ts[:, None] - 3600 * (hp.window - 1 - np.arange(hp.window))[None, :]) // 3600) % 24
    input_sha = X.digest({"identity": claim["identity"], "population_id": claim["population_id"],
                          "feature_id": claim["feature_id"], "file_sha256": corpus.file_sha256,
                          "column_sha256": column_sha, "grid_seconds": X.GRID_SECONDS, "window": hp.window,
                          "fold": {"fold_id": fold.fold_id, "fit": list(fold.fit), "val": list(fold.val)}})
    return {"grid": grid, "fold": fold, "fit_origins": fit_o, "score": score_b, "hidden": hidden, "naive": naive,
            "hours": hours, "norm": norm, "cal": cal, "role": role, "column_sha256": column_sha,
            "identity": identity, "input_sha256": input_sha,
            "rows_sha256": U.row_ids_sha256(np.asarray(score_b.row_ids, np.int64)),
            "mask_sha256": X.mask_sha256(hidden)}


def run_task(claim: dict, inputs: Dict[str, str], output_root: Optional[str], hp: X.Hyper,
             registry: Optional[dict] = None, gpu_uuid: Optional[str] = None, allow_cpu_training: bool = False) -> dict:
    t_wall = time.perf_counter()
    task_id = claim["task_id"]
    terminal = os.path.join(output_root, task_id, "result.json") if output_root else None
    if terminal and os.path.isfile(terminal):
        with open(terminal) as f:
            retained = json.load(f)
        if retained.get("task_id") != task_id or retained.get("status") != "COMPLETE":
            raise X.Refusal("RETAINED_TERMINAL_IDENTITY_MISMATCH", terminal)
        log(f"RESTART_FROM_RETAINED_TERMINAL {terminal}; nothing recomputed")
        return retained
    corpus = X.Corpus(claim["identity"], inputs, registry)
    log(f"corpus {claim['identity']} verified: {sorted(corpus.file_sha256)}")
    P = prepare(claim, corpus, hp)
    score, hidden, fold = P["score"], P["hidden"], P["fold"]
    log(f"fold {fold.fold_id} fit={fold.fit} val={fold.val} fit_origins={P['fit_origins'].size} "
        f"scoring_rows={len(score)} hidden_points={int(hidden.sum())}")
    naive_scores = X.hidden_scores(score.signal, P["naive"], hidden, P["norm"].std)
    arm = claim["arm"]
    origin_covering = arm in X.V2_ARMS
    trained = arm in ("TRAINED_ENCODER", "TRAINED_ENCODER_V2")
    random_control = arm in ("RANDOM_ENCODER", "RANDOM_ENCODER_V2")
    model_sha, weights, training_report, device, gpu, vram_peak = None, None, None, "cpu", None, None
    artifacts = {}
    if arm == "RAW":
        corrupted = X.corrupt(score, hidden)
        scores = X.hidden_scores(score.signal, corrupted.signal, hidden, P["norm"].std, P["hours"])
        model_sha = X.digest({"arm": "RAW", "model": "corrupted input scored directly on hidden points",
                              "architecture_id": "identity"})
        weights = {"initial_weights_sha256": None, "final_weights_sha256": None, "chosen_weights_sha256": None,
                   "chosen_epoch": None, "updates": 0}
    else:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if trained and not allow_cpu_training:
            gpu = verify_gpu(gpu_uuid)
            log(f"GPU verified in process: {gpu}")
        import keras
        import tensorflow as tf
        keras.utils.set_random_seed(claim["seed"])
        builder = X.build_origin_covering_models if origin_covering else X.build_models
        encoder, decoder, training = builder(hp, calendar_dim=P["cal"].shape[1])
        X.seed_weights([encoder, decoder], claim["seed"])
        initial = X.weights_digest([encoder, decoder])
        device = variable_device(encoder)
        if trained and not allow_cpu_training and "GPU:0" not in device:
            raise X.Refusal("GPU_NOT_VERIFIED", f"model variables live on {device!r}, not on the GPU")
        if random_control:
            updates = int(training.optimizer.iterations.numpy())
            if updates != 0:
                raise X.Refusal("RANDOM_ENCODER_WAS_UPDATED", str(updates))
            chosen = X.weights_digest([encoder, decoder])
            if chosen != initial:
                raise X.Refusal("RANDOM_ENCODER_WEIGHTS_CHANGED")
            weights = {"initial_weights_sha256": initial, "final_weights_sha256": initial,
                       "chosen_weights_sha256": chosen, "chosen_epoch": None, "updates": 0}
            training_report = {"stop_reason": "NOT_TRAINED", "epochs_run": 0, "updates": 0}
        else:
            fit_o = P["fit_origins"]
            grid, norm, cal = P["grid"], P["norm"], P["cal"]
            n_tail = max(1, int(round(hp.es_tail_fraction * fit_o.size)))
            tail = fit_o[-n_tail:]
            tail_start = int(grid.ts[tail[0]])
            core = fit_o[grid.ts[fit_o] <= tail_start - hp.es_purge_hours * 3600]
            core = X.even_subset(core, hp.max_fit_windows)
            if core.size < hp.min_fit_windows:
                raise X.Refusal("INSUFFICIENT_TRAIN_FIT_WINDOWS", f"{core.size} fit origins after the purged tail")
            fit_b = U.make_windows(grid.ts, grid.x, grid.observed, core, hp.window, norm, calendar=cal)
            es_b = U.make_windows(grid.ts, grid.x, grid.observed, tail, hp.window, norm, calendar=cal)
            es_hidden = X.hidden_mask(es_b.observed_mask, X._seed_from({**P["identity"], "purpose": "es_mask"}),
                                      hp.hide_fraction)
            log(f"training on {len(fit_b)} fit windows; early-stopping tail {len(es_b)} windows "
                f"purged {hp.es_purge_hours}h; device {device}")
            training_report = X.fit_trained(training, encoder, decoder, fit_b, es_b, hp, claim["seed"], es_hidden, log)
            training_report.update({"n_fit_windows": len(fit_b), "n_es_windows": len(es_b),
                                    "es_tail_first_ts": int(es_b.anchor_ts[0]), "es_tail_last_ts": int(es_b.anchor_ts[-1]),
                                    "fit_last_ts": int(fit_b.anchor_ts[-1]), "es_mask_sha256": X.mask_sha256(es_hidden),
                                    "optimizer_device": str(getattr(getattr(training.optimizer.iterations, "value", None),
                                                                    "device", "") or "")})
            if training_report["updates"] < 1:
                raise X.Refusal("TRAINED_ENCODER_NO_UPDATES")
            weights = {"initial_weights_sha256": initial, "final_weights_sha256": training_report["final_weights_sha256"],
                       "chosen_weights_sha256": training_report["chosen_weights_sha256"],
                       "chosen_epoch": training_report["chosen_epoch"], "updates": training_report["updates"]}
        recon = X.reconstruct(training, X.corrupt(score, hidden))
        scores = X.hidden_scores(score.signal, recon, hidden, P["norm"].std, P["hours"])
        model_sha = weights["chosen_weights_sha256"]
        if output_root:
            wpath = os.path.join(output_root, task_id, "chosen.weights.h5")
            os.makedirs(os.path.dirname(wpath), exist_ok=True)
            training.save_weights(wpath)
            artifacts["chosen_weights_file"] = wpath
            artifacts["chosen_weights_file_sha256"] = U.sha256_file(wpath)
        if gpu is not None:
            try:
                vram_peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
            except Exception as exc:  # pragma: no cover - driver specific
                log(f"VRAM peak unavailable: {exc}")
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "schema": RESULT_SCHEMA, "status": "COMPLETE", "task_id": task_id, "seed": claim["seed"],
        "population_id": claim["population_id"], "identity": claim["identity"], "feature_id": claim["feature_id"],
        "fold_id": claim["fold_id"], "arm": arm,
        "input_sha256": P["input_sha256"], "code_sha256": code_sha256(), "model_sha256": model_sha,
        "rows_sha256": P["rows_sha256"], "mask_sha256": P["mask_sha256"], "population_n": int(len(score)),
        "metrics": {"mae": scores["mae"], "naive_mae": naive_scores["mae"], "mse": scores["mse"],
                    "naive_mse": naive_scores["mse"], "mae_orig": scores["mae_orig"],
                    "naive_mae_orig": naive_scores["mae_orig"], "hidden_points": scores["hidden_points"]},
        "by_hour_utc": scores.get("by_hour_utc"),
        "source": {"role": P["role"], "file_sha256": corpus.file_sha256, "column_sha256": P["column_sha256"],
                   "bar_seconds": corpus.bar_seconds, "grid_seconds": X.GRID_SECONDS,
                   "train_end_ts": int(corpus.timestamps()[-1])},
        "fold": {"fold_id": fold.fold_id, "fit": list(fold.fit), "val": list(fold.val),
                 "scoring_first_ts": int(score.anchor_ts[0]), "scoring_last_ts": int(score.anchor_ts[-1]),
                 "availability_rule": "window rows have timestamp <= origin; origin is an observed TRAIN decision row"},
        "normalization": {"mean": P["norm"].mean, "std": P["norm"].std, "n": P["norm"].n, "constant": P["norm"].constant,
                          "fitted_on": "fold fit range only"},
        "corruption": {"type": "hide_observed_points", "fraction": hp.hide_fraction, "derived_from": "feature/fold identity (arm excluded)"},
        "architecture": {"id": X.ARCHITECTURE_ID_V2 if origin_covering else X.ARCHITECTURE_ID,
                         "latent_shape": [hp.latent_steps, hp.latent_dim],
                         "last_latent_lag_rows": 0 if origin_covering else 3,
                         "calendar_context": list(U.CALENDAR_SPEC), "target_input": False},
        "hyper": hp.to_dict(), "weights": weights, "artifacts": artifacts,
        "training": {**(training_report or {"stop_reason": "NOT_APPLICABLE_RAW"}),
                     "chosen_epoch": weights["chosen_epoch"], "updates": weights["updates"]},
        "device": {"variables": device, "gpu": gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "cost": {"wall_s": time.perf_counter() - t_wall, "cpu_s": ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime,
                 "peak_ram_bytes": _cgroup_peak_bytes() or int(ru.ru_maxrss) * 1024,
                 "peak_rss_bytes": int(ru.ru_maxrss) * 1024, "peak_cgroup_bytes": _cgroup_peak_bytes(),
                 "peak_vram_bytes": vram_peak},
        "code_commit": _git_commit(), "host": platform.node(), "python": platform.python_version(),
    }
    validate_terminal(result, claim)
    if terminal:
        atomic_json(terminal, result)
    return result


# --------------------------------------------------------------------------- coverage
def coverage(features: Dict[str, list], inputs: Dict[str, str], folds, hp: X.Hyper, registry=None) -> dict:
    reg = X.CORPORA if registry is None else registry
    report = {"schema": "fs4.coverage.v1", "folds": list(folds), "populations": {}}
    for population, feats in features.items():
        identities = [k for k, v in reg.items() if v["population_id"] == population]
        if len(identities) != 1:
            report["populations"][population] = {"state": "NO_PINNED_CORPUS", "features_total": len(feats)}
            continue
        corpus = X.Corpus(identities[0], inputs, registry)
        ts_rows = corpus.timestamps()
        cols = corpus.columns()
        per, counts = {}, {"AVAILABLE": 0, "NOT_AVAILABLE_FOR_TRAIN": 0, "FEATURE_NOT_IN_CORPUS": 0, "REFUSED": 0}
        for f in feats:
            entry = {"folds": {}}
            if f not in cols:
                entry["state"] = "FEATURE_NOT_IN_CORPUS"
                counts["FEATURE_NOT_IN_CORPUS"] += 1
                per[f] = entry
                continue
            try:
                values, role, _ = corpus.feature(f)
                grid = X.to_grid(ts_rows, values)
            except X.Refusal as exc:
                entry["state"], entry["reason"] = "REFUSED", str(exc)
                counts["REFUSED"] += 1
                per[f] = entry
                continue
            entry["role"] = role
            ok = True
            for fold_id in folds:
                try:
                    fold = X.fold_spec(fold_id, ts_rows)
                    n_fit, n_score = int(X.origins(grid, fold.fit, hp).size), int(X.origins(grid, fold.val, hp).size)
                    code = ("NO_TRAIN_OBSERVATIONS" if n_fit == 0 or n_score == 0 else
                            "INSUFFICIENT_TRAIN_FIT_WINDOWS" if n_fit < hp.min_fit_windows else
                            "INSUFFICIENT_TRAIN_SCORING_WINDOWS" if n_score < hp.min_scoring_windows else None)
                    entry["folds"][fold_id] = {"state": "NOT_AVAILABLE_FOR_TRAIN" if code else "AVAILABLE",
                                               "code": code, "n_fit_origins": n_fit, "n_scoring_rows": n_score}
                except X.Refusal as exc:
                    entry["folds"][fold_id] = {"state": "NOT_AVAILABLE_FOR_TRAIN", "code": exc.code, "reason": str(exc)}
                ok = ok and entry["folds"][fold_id]["state"] == "AVAILABLE"
            entry["state"] = "AVAILABLE" if ok else "NOT_AVAILABLE_FOR_TRAIN"
            counts[entry["state"]] += 1
            per[f] = entry
        report["populations"][population] = {"identity": identities[0], "file_sha256": corpus.file_sha256,
                                             "features_total": len(feats), "counts": counts, "per_feature": per}
    return report


# --------------------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", metavar="ROLE=PATH",
                   help="governed input file by role (also FS4_INPUT_<ROLE> env); roles in app.fs4_extractibility.CORPORA")
    p.add_argument("--output-root", default=os.environ.get("FS4_OUTPUT_ROOT"),
                   help="durable local directory: <root>/<task_id>/result.json and chosen weights")
    p.add_argument("--gpu-uuid", default=os.environ.get("FS4_GPU_UUID"),
                   help="physical UUID the TRAINED_ENCODER arm must verify in process")
    p.add_argument("--claim-file", help="claim JSON file (alternative to stdin / FS4_CLAIM_JSON)")
    p.add_argument("--coverage", metavar="FEATURES_JSON", help='{"EURUSD": [feature_id, ...], ...}: report only')
    p.add_argument("--folds", default=",".join(X.FOLD_YEARS), help="coverage folds")
    p.add_argument("--corpus-registry", help="TESTS ONLY: JSON registry replacing the pinned CORPORA")
    p.add_argument("--allow-cpu-training", action="store_true", help="TESTS ONLY: skip the physical GPU proof")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--max-fit-windows", type=int)
    p.add_argument("--min-fit-windows", type=int)
    p.add_argument("--min-scoring-windows", type=int)
    return p.parse_args(argv)


def hyper_from(args) -> X.Hyper:
    kw = {k: v for k, v in (("max_epochs", args.max_epochs), ("patience", args.patience),
                            ("max_fit_windows", args.max_fit_windows), ("min_fit_windows", args.min_fit_windows),
                            ("min_scoring_windows", args.min_scoring_windows)) if v is not None}
    return X.Hyper(**kw)


def main(argv=None, stdin=None, stdout=None) -> int:
    args = parse(argv)
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    hp = hyper_from(args)
    try:
        inputs = _inputs_from(args)
        if args.coverage:
            with open(args.coverage) as f:
                feats = json.load(f)
            out = coverage(feats, inputs, [x for x in args.folds.split(",") if x], hp, _registry(args))
            print(X.canonical(out), file=stdout, flush=True)
            return 0
        claim = load_claim(stdin, args.claim_file)
    except X.Refusal as exc:
        print(X.canonical({"status": "REFUSED", "code": f"REFUSED_{exc.code}", "reason": str(exc)}), file=stdout, flush=True)
        log(f"REFUSED {exc}")
        print(f"REFUSED_{exc.code} {str(exc)[:160]}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    task_id = claim["task_id"]
    try:
        result = run_task(claim, inputs, args.output_root, hp, _registry(args), args.gpu_uuid, args.allow_cpu_training)
    except X.Refusal as exc:
        if X.is_not_available(exc.code):
            status, declared, rc = "NOT_AVAILABLE_FOR_TRAIN", exc.code, EXIT_REFUSED
        elif exc.code == "GPU_NOT_VERIFIED":
            status, declared, rc = "GPU_NOT_VERIFIED", "REFUSED_GPU_NOT_VERIFIED", EXIT_GPU
        else:
            status, declared, rc = "REFUSED", f"REFUSED_{exc.code}", EXIT_REFUSED
        print(X.canonical({"status": status, "code": declared, "task_id": task_id, "arm": claim["arm"],
                           "reason": str(exc)}), file=stdout, flush=True)
        log(f"{status} {exc}")
        print(f"{declared} {str(exc)[:160]}", file=sys.stderr, flush=True)  # last stderr line becomes fail --reason
        return rc
    except U.ContractError as exc:
        print(X.canonical({"status": "REFUSED", "code": f"REFUSED_{type(exc).__name__}", "task_id": task_id,
                           "reason": f"{type(exc).__name__}: {exc}"}), file=stdout, flush=True)
        log(f"REFUSED {type(exc).__name__}: {exc}")
        print(f"REFUSED_{type(exc).__name__} {str(exc)[:160]}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    print(X.canonical(result), file=stdout, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
