"""Metrics and pilot contract tests (lane D, 2026-10-03). Written before the implementation."""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest

from app import temporal_extractor_metrics as M


def test_effective_dimension_and_cka():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 1))
    rank1 = np.concatenate([a, 2 * a, -a], axis=1)
    assert M.effective_dimension(rank1)["participation_ratio"] == pytest.approx(1.0, abs=1e-6)
    iso = rng.normal(size=(5000, 4))
    assert M.effective_dimension(iso)["participation_ratio"] > 3.8
    z = rng.normal(size=(100, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert M.linear_cka(z, z @ q) == pytest.approx(1.0, abs=1e-6)
    assert M.linear_cka(z, rng.normal(size=(100, 8))) < 0.5


def test_reconstruction_not_applicable_without_decoder_and_scales():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 24, 1)).astype(np.float32)
    m = np.ones_like(x)
    r = M.reconstruction_metrics(x, None, m, mean=10.0, std=2.0, extreme_threshold_norm=1.5)
    assert r["status"] == "NOT_APPLICABLE"
    r = M.reconstruction_metrics(x, x + 0.5, m, mean=10.0, std=2.0, extreme_threshold_norm=1.5)
    assert r["status"] == "MEASURED"
    assert r["mae_norm"] == pytest.approx(0.5) and r["mae_orig"] == pytest.approx(1.0)
    assert r["mse_norm"] == pytest.approx(0.25) and r["mse_orig"] == pytest.approx(1.0)
    assert r["mae_rel_train_constant"] == pytest.approx(0.5 / np.mean(np.abs(x)), rel=1e-5)
    for k in ("acf_l1", "log_psd_l1", "mae_extremes", "dtw_mean"):
        assert k in r
    m2 = m.copy()
    m2[:, :12] = 0  # missing points are not scored
    r2 = M.reconstruction_metrics(x, np.where(m2 > 0, x, x + 9), m2, 0.0, 1.0, 1.5)
    assert r2["mae_norm"] == pytest.approx(0.0, abs=1e-7)


def test_probes_same_rows_naive_and_deltas():
    rng = np.random.default_rng(2)
    n, t, d = 300, 24, 4
    signal = rng.normal(size=(n, t, 1)).astype(np.float32)
    y = np.stack([signal[:, -1, 0] * 0.8 + 0.1 * rng.normal(size=n), rng.normal(size=n)], axis=1)
    y[5, 0] = np.nan
    yb = (signal[:, -1, 0] > 0).astype(np.int64)
    cal = np.zeros((n, t, 2), np.float32)
    reps = {"raw": signal, "random": rng.normal(size=(n, t, d)).astype(np.float32),
            "trained": np.concatenate([signal, rng.normal(size=(n, t, d - 1))], -1).astype(np.float32)}
    fit, val = np.arange(0, 200), np.arange(200, 300)
    rows = M.equal_probes(reps, cal, {"Y_s": y, "Y_b": yb}, fit, val)
    reg = [r for r in rows if r["target"] == "Y_s" and r["horizon_index"] == 0]
    assert {r["representation"] for r in reg} == {"raw", "random", "trained"}
    assert len({r["n_val"] for r in reg}) == 1  # same rows for all representations
    tr = next(r for r in reg if r["representation"] == "trained")
    assert tr["n_val"] == tr["naive_n_val"]
    assert tr["skill_vs_zero"] > 0.5
    cls = [r for r in rows if r["target"] == "Y_b"]
    assert all("log_loss" in r and "brier" in r and "prior_log_loss" in r for r in cls)
    deltas = M.probe_deltas(rows)
    k = ("Y_s", 0)
    assert deltas[k]["delta_probe_random_minus_trained"] > 0
    assert "preservation_raw_minus_trained" in deltas[k]


def test_target_is_probe_supervision_only():
    with pytest.raises(Exception):
        M.equal_probes({"trained": np.zeros((4, 3, 2), np.float32), "Y_s": np.zeros((4, 3, 1))},
                       np.zeros((4, 3, 1), np.float32), {"Y_s": np.zeros((4, 1))}, np.arange(2), np.arange(2, 4))


def test_pilot_end_to_end_on_synthetic_ps2_batch(tmp_path):
    from app import univariate_temporal_pilot as P
    bdir = tmp_path / "batch_001"
    P.write_synthetic_ps2_batch(str(bdir), n=420, features=("feat_a", "feat_b"), seed=0)
    out = tmp_path / "out"
    rc = P.main(["--batch_dir", str(bdir), "--out_dir", str(out), "--window", "24", "--latent_dim", "8",
                 "--filters", "4", "--dilations", "1,2", "--max_epochs", "2", "--patience", "1",
                 "--families", "identity,random,ae,dae", "--features", "feat_a"])
    assert rc == 0
    run = json.load(open(out / "run_manifest.json"))
    assert run["batch_manifest_sha256"] and run["peak_rss_bytes"] > 0
    assert run["features"] == ["feat_a"]
    rows = [json.loads(l) for l in open(out / "results.jsonl")]
    fams = {r["family"] for r in rows if r["kind"] == "fold_family"}
    assert fams == {"identity", "random", "ae", "dae"}
    assert any(r["kind"] == "feature_summary" and "stability" in r for r in rows)
    donors = [p for p in (out / "donors").rglob("donor_manifest.json")]
    assert donors and all(json.load(open(p))["train_scope"]["kind"] == "TRAIN_ONLY" for p in donors)
    # a tampered batch is refused before any fit
    man = json.load(open(bdir / "batch_manifest.json"))
    man["series"]["sha256"] = "0" * 64
    json.dump(man, open(bdir / "batch_manifest.json", "w"))
    with pytest.raises(Exception):
        P.main(["--batch_dir", str(bdir), "--out_dir", str(tmp_path / "o2"), "--window", "24"])
