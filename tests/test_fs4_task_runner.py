"""Acceptance tests for app.fs4_task_runner (plan 2026-10-06 FS4-03/04/05/06/07/09). CPU, tiny synthetic corpus."""
import datetime as dt
import io
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import fs4_extractibility as X
from app import fs4_task_runner as R
from app import univariate_temporal as U

IDENTITY = "synthetic-train:v1"
ROLE = "synthetic_train"
FAST = ["--allow-cpu-training", "--max-epochs", "2", "--patience", "1", "--max-fit-windows", "256",
        "--min-fit-windows", "64", "--min-scoring-windows", "16"]


def _ts(y, m, d):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


def write_corpus(path, start=(2018, 1, 1), end=(2020, 3, 1), seed=0, perturb_after=None):
    rng = np.random.default_rng(seed)
    ts = np.arange(_ts(*start), _ts(*end), 3600, dtype=np.int64)
    ts = ts[((ts // 86400 + 3) % 7) != 6]  # no Sunday rows: grid gaps like a market calendar
    n = ts.size
    e = rng.normal(size=n)
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.9 * v[t - 1] + e[t]
    hour = (ts // 3600) % 24
    feat_a = (v + 0.5 * np.sin(2 * np.pi * hour / 24)).astype(np.float64)
    feat_a[rng.random(n) < 0.05] = np.nan
    clock = ((ts - ts[0]) // 3600).astype(np.float64)  # exact alignment probe: value == grid hour index
    feat_nan = np.full(n, np.nan)
    if perturb_after is not None:
        late = ts > perturb_after
        feat_a[late] = rng.normal(size=late.sum()) * 50
        clock[late] += 1e6
    table = pa.table({"t_decision_utc": pa.array(ts * 10 ** 9, pa.timestamp("ns", tz="UTC")),
                      "row_id": np.arange(n, dtype=np.int64), "feat_a": feat_a, "clock": clock,
                      "feat_nan": feat_nan, "Y_s": feat_a})
    pq.write_table(table, path)
    return path


def registry_for(path, population="SYN", bar=3600):
    return {IDENTITY: {"population_id": population, "bar_seconds": bar, "files": {ROLE: U.sha256_file(path)}}}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    path = write_corpus(str(d / "train.parquet"))
    reg = d / "registry.json"
    reg.write_text(json.dumps(registry_for(path)))
    return {"path": path, "registry": str(reg), "dir": d}


def claim_for(feature, arm, fold="inner_2019", seed=0, population="SYN"):
    payload = {"schema": R.TASK_SCHEMA, "population_id": population, "identity": IDENTITY,
               "feature_id": feature, "fold_id": fold, "arm": arm, "seed": seed}
    return {**payload, "task_id": X.task_digest(payload), "attempt": 1, "lease_until": 0}


def run(claim, corpus, extra=(), output_root=None, registry=None):
    out = io.StringIO()
    argv = ["--input", f"{ROLE}={corpus['path']}", "--corpus-registry", registry or corpus["registry"], *FAST, *extra]
    if output_root:
        argv += ["--output-root", str(output_root)]
    rc = R.main(argv, stdin=io.StringIO(json.dumps(claim)), stdout=out)
    text = out.getvalue()
    assert text.count("\n") == 1, "exactly one JSON document on stdout"
    return rc, json.loads(text)


# ---------------------------------------------------------------- FS4-09 three arms, FS4-05 controls
@pytest.fixture(scope="module")
def three_arms(corpus, tmp_path_factory):
    root = tmp_path_factory.mktemp("out")
    results = {}
    for arm in X.ARMS:
        rc, res = run(claim_for("feat_a", arm), corpus, output_root=root)
        assert rc == 0, res
        results[arm] = res
    return results, root


def test_three_arms_share_rows_mask_population_and_naive(three_arms):
    results, _ = three_arms
    raw, rnd, trn = (results[a] for a in X.ARMS)
    for key in ("rows_sha256", "mask_sha256", "population_n", "input_sha256"):
        assert raw[key] == rnd[key] == trn[key], key
    assert raw["metrics"]["naive_mae"] == rnd["metrics"]["naive_mae"] == trn["metrics"]["naive_mae"]
    assert raw["metrics"]["hidden_points"] == trn["metrics"]["hidden_points"] >= raw["population_n"]
    for res in results.values():
        R.validate_terminal(res, claim_for("feat_a", res["arm"]))
        assert res["metrics"]["mae"] > 0 and res["metrics"]["naive_mae"] > 0
    assert len({r["model_sha256"] for r in results.values()}) == 3


def test_random_control_runs_no_optimizer_and_shares_initial_weights(three_arms):
    results, _ = three_arms
    rnd, trn = results["RANDOM_ENCODER"], results["TRAINED_ENCODER"]
    assert rnd["weights"]["updates"] == 0 and rnd["training"]["stop_reason"] == "NOT_TRAINED"
    assert rnd["weights"]["initial_weights_sha256"] == rnd["weights"]["chosen_weights_sha256"] == rnd["model_sha256"]
    assert rnd["weights"]["initial_weights_sha256"] == trn["weights"]["initial_weights_sha256"]


def test_trained_records_update_counter_and_restored_checkpoint(three_arms):
    results, root = three_arms
    trn = results["TRAINED_ENCODER"]
    assert trn["weights"]["updates"] >= 1 and trn["training"]["restored_best_checkpoint"] is True
    assert trn["training"]["updates"] == trn["weights"]["updates"] and trn["training"]["chosen_epoch"] >= 0
    assert {"cpu_s", "wall_s", "peak_ram_bytes", "peak_vram_bytes"} <= set(trn["cost"])
    assert results["RAW"]["training"]["updates"] == 0 and results["RAW"]["training"]["chosen_epoch"] is None
    assert trn["weights"]["chosen_epoch"] == trn["training"]["chosen_epoch"] >= 0
    assert trn["weights"]["chosen_weights_sha256"] == trn["model_sha256"] != trn["weights"]["initial_weights_sha256"]
    assert trn["training"]["es_tail_last_ts"] <= trn["fold"]["fit"][1]
    assert trn["training"]["fit_last_ts"] <= trn["training"]["es_tail_first_ts"] - 168 * 3600  # purged tail
    assert trn["training"]["es_mask_sha256"] != trn["mask_sha256"]
    assert os.path.isfile(trn["artifacts"]["chosen_weights_file"])
    assert trn["architecture"]["latent_shape"] == [6, 8] and trn["architecture"]["target_input"] is False


# ---------------------------------------------------------------- FS4-03 future perturbation
def test_future_rows_after_the_fold_do_not_change_terminals(corpus, three_arms, tmp_path):
    results, _ = three_arms
    val_end = results["RAW"]["fold"]["val"][1]
    path = write_corpus(str(tmp_path / "future.parquet"), perturb_after=val_end)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps(registry_for(path)))
    alt = {"path": path, "registry": str(reg)}
    for arm in ("RAW", "RANDOM_ENCODER"):
        rc, res = run(claim_for("feat_a", arm), alt)
        assert rc == 0
        for key in ("rows_sha256", "mask_sha256", "population_n", "model_sha256"):
            assert res[key] == results[arm][key], (arm, key)
        assert res["metrics"] == results[arm]["metrics"]
    rc, res = run(claim_for("feat_a", "TRAINED_ENCODER"), alt)
    assert rc == 0 and res["rows_sha256"] == results["TRAINED_ENCODER"]["rows_sha256"]
    assert res["weights"]["initial_weights_sha256"] == results["TRAINED_ENCODER"]["weights"]["initial_weights_sha256"]
    assert res["metrics"]["naive_mae"] == results["TRAINED_ENCODER"]["metrics"]["naive_mae"]
    assert res["metrics"]["mae"] == pytest.approx(results["TRAINED_ENCODER"]["metrics"]["mae"], rel=1e-4)


def test_encoder_latent_is_causal_at_the_declared_reduction():
    enc, _, _ = X.build_models(X.Hyper())
    X.seed_weights([enc], 0)
    rng = np.random.default_rng(1)
    b = U.TemporalBatch.from_inputs({"signal": rng.normal(size=(3, 24, 1)).astype("f4"),
                                     "observed_mask": np.ones((3, 24, 1), "f4"),
                                     "delta_time": np.zeros((3, 24, 1), "f4"),
                                     "calendar": rng.normal(size=(3, 24, 6)).astype("f4")})
    z = enc.predict(b.as_inputs(), verbose=0)
    for t0 in (5, 13, 17):
        p = b.copy_arrays()
        p.signal[:, t0:] += 7.0
        p.calendar[:, t0:] = 0.0
        zp = enc.predict(p.as_inputs(), verbose=0)
        unchanged = [j for j in range(6) if 4 * j < t0]
        changed = [j for j in range(6) if 4 * j >= t0]
        np.testing.assert_allclose(zp[:, unchanged], z[:, unchanged], atol=1e-6)
        assert not np.allclose(zp[:, changed], z[:, changed])


# ---------------------------------------------------------------- exact alignment, availability <= origin
def test_windows_align_exactly_and_never_look_past_the_origin(corpus):
    c = X.Corpus(IDENTITY, {ROLE: corpus["path"]}, json.load(open(corpus["registry"])))
    claim = claim_for("clock", "RAW")
    hp = X.Hyper(min_fit_windows=64, min_scoring_windows=16)
    P = R.prepare(claim, c, hp)
    s, norm, grid = P["score"], P["norm"], P["grid"]
    hours = (s.signal[..., 0].astype(np.float64) * norm.std + norm.mean)
    vis = s.observed_mask[..., 0] > 0
    expect = ((s.anchor_ts[:, None] - grid.ts[0]) // 3600 - (23 - np.arange(24))[None, :]).astype(np.float64)
    np.testing.assert_allclose(hours[vis], expect[vis], atol=1e-3)
    assert vis[:, -1].all()  # the origin itself is observed
    assert np.all(hours[vis] <= ((s.anchor_ts - grid.ts[0]) // 3600)[:, None].repeat(24, 1)[vis] + 1e-3)
    assert np.all(grid.ts[P["fit_origins"]] <= P["fold"].fit[1]) and np.all(s.anchor_ts >= P["fold"].val[0])
    assert P["hidden"].sum(axis=1).min() >= 1 and not np.any(P["hidden"] & ~vis & ~P["hidden"])
    assert np.all(P["hidden"] <= (s.observed_mask[..., 0] > 0))


def test_naive_uses_last_visible_value_strictly_before_t():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    vis = np.array([[True, False, True, False]])
    np.testing.assert_array_equal(X.naive_prediction(x, vis), [[0.0, 1.0, 1.0, 3.0]])


# ---------------------------------------------------------------- FS4-07 restart from retained terminal
def test_restart_adopts_retained_terminal_without_recomputing(corpus, tmp_path):
    root = tmp_path / "out"
    claim = claim_for("feat_a", "RAW")
    rc, first = run(claim, corpus, output_root=root)
    assert rc == 0 and (root / claim["task_id"] / "result.json").is_file()
    gone = {"path": str(tmp_path / "missing.parquet"), "registry": corpus["registry"]}
    rc, again = run(claim, gone, output_root=root)  # the corpus is unreachable: only the terminal can answer
    assert rc == 0 and again == first
    rc, refused = run(claim, gone)
    assert rc == R.EXIT_REFUSED and refused["status"] == "REFUSED" and "CORPUS_FILE_MISSING" in refused["reason"]


# ---------------------------------------------------------------- FS4-06 refusals, never a fake zero
def test_refusals_are_typed(corpus, tmp_path):
    rc, res = run(claim_for("feat_nan", "RAW"), corpus)
    assert rc == R.EXIT_REFUSED and res["status"] == "NOT_AVAILABLE_FOR_TRAIN" and res["code"] == "NO_TRAIN_OBSERVATIONS"
    rc, res = run(claim_for("feat_nan", "TRAINED_ENCODER"), corpus)
    assert rc == R.EXIT_REFUSED and res["status"] == "NOT_AVAILABLE_FOR_TRAIN" and "metrics" not in res
    rc, res = run(claim_for("not_a_column", "RAW"), corpus)
    assert rc == R.EXIT_REFUSED and "FEATURE_NOT_IN_CORPUS" in res["reason"] and res["code"] == "REFUSED_FEATURE_NOT_IN_CORPUS"
    rc, res = run(claim_for("Y_s", "RAW"), corpus)
    assert rc == R.EXIT_REFUSED and "target" in res["reason"].lower()
    rc, res = run(claim_for("feat_a", "RAW", fold="inner_2023"), corpus)
    assert rc == R.EXIT_REFUSED and res["status"] == "NOT_AVAILABLE_FOR_TRAIN"
    bad = dict(claim_for("feat_a", "RAW"), task_id="0" * 64)
    rc, res = run(bad, corpus)
    assert rc == R.EXIT_REFUSED and "TASK_IDENTITY_MISMATCH" in res["reason"]
    reg = tmp_path / "wrong.json"
    reg.write_text(json.dumps({IDENTITY: {"population_id": "SYN", "bar_seconds": 3600, "files": {ROLE: "0" * 64}}}))
    rc, res = run(claim_for("feat_a", "RAW"), corpus, registry=str(reg))
    assert rc == R.EXIT_REFUSED and "CORPUS_DIGEST_MISMATCH" in res["reason"]
    rc, res = run(claim_for("feat_a", "RAW", population="EURUSD"), corpus)
    assert rc == R.EXIT_REFUSED and "POPULATION_IDENTITY_MISMATCH" in res["reason"]
    ok = {"status": "COMPLETE", "task_id": claim_for("feat_a", "RAW")["task_id"], "seed": 0, "population_n": 5,
          **{k: "a" * 64 for k in ("input_sha256", "code_sha256", "model_sha256", "rows_sha256", "mask_sha256")},
          "metrics": {"mae": 0.1, "naive_mae": 0.2}}
    R.validate_terminal(ok, claim_for("feat_a", "RAW"))
    with pytest.raises(X.Refusal):
        R.validate_terminal({**ok, "metrics": {"mae": float("nan"), "naive_mae": 0.2}}, claim_for("feat_a", "RAW"))
    with pytest.raises(X.Refusal):
        R.validate_terminal({**ok, "population_n": 0}, claim_for("feat_a", "RAW"))


def test_trained_arm_refuses_without_a_physical_gpu_proof(corpus):
    rc, res = run(claim_for("feat_a", "TRAINED_ENCODER"), corpus, extra=[])
    # FAST passes --allow-cpu-training; re-run without it on a CPU-only process
    out = io.StringIO()
    argv = ["--input", f"{ROLE}={corpus['path']}", "--corpus-registry", corpus["registry"], "--max-epochs", "1",
            "--min-fit-windows", "64", "--min-scoring-windows", "16"]
    rc = R.main(argv, stdin=io.StringIO(json.dumps(claim_for("feat_a", "TRAINED_ENCODER"))), stdout=out)
    res = json.loads(out.getvalue())
    assert rc == R.EXIT_GPU and res["status"] == "GPU_NOT_VERIFIED" and res["code"] == "REFUSED_GPU_NOT_VERIFIED"


# ---------------------------------------------------------------- coverage report
def test_coverage_reports_every_feature_and_fold(corpus, tmp_path):
    feats = tmp_path / "features.json"
    feats.write_text(json.dumps({"SYN": ["feat_a", "clock", "feat_nan", "absent"], "MARS": ["x"]}))
    out = io.StringIO()
    rc = R.main(["--input", f"{ROLE}={corpus['path']}", "--corpus-registry", corpus["registry"], "--coverage",
                 str(feats), "--folds", "inner_2019,inner_2023", "--min-fit-windows", "64", "--min-scoring-windows", "16"],
                stdout=out)
    assert rc == 0
    rep = json.loads(out.getvalue())
    syn = rep["populations"]["SYN"]
    assert syn["counts"] == {"AVAILABLE": 0, "NOT_AVAILABLE_FOR_TRAIN": 3, "FEATURE_NOT_IN_CORPUS": 1, "REFUSED": 0}
    assert syn["per_feature"]["feat_a"]["folds"]["inner_2019"]["state"] == "AVAILABLE"
    assert syn["per_feature"]["feat_a"]["folds"]["inner_2023"]["state"] == "NOT_AVAILABLE_FOR_TRAIN"
    assert syn["per_feature"]["feat_nan"]["folds"]["inner_2019"]["state"] == "NOT_AVAILABLE_FOR_TRAIN"
    assert rep["populations"]["MARS"]["state"] == "NO_PINNED_CORPUS"


def test_typed_refusal_code_is_the_last_stderr_line(corpus, capsys):
    rc, res = run(claim_for("feat_nan", "RAW"), corpus)
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == R.EXIT_REFUSED and err[-1].startswith("NO_TRAIN_OBSERVATIONS ")


def test_claim_may_come_from_env_when_stdin_is_empty(corpus, monkeypatch):
    claim = claim_for("feat_a", "RAW")
    monkeypatch.setenv("FS4_CLAIM_JSON", json.dumps(claim))
    out = io.StringIO()
    argv = ["--input", f"{ROLE}={corpus['path']}", "--corpus-registry", corpus["registry"], *FAST]
    rc = R.main(argv, stdin=io.StringIO(""), stdout=out)
    assert rc == 0 and json.loads(out.getvalue())["task_id"] == claim["task_id"]
