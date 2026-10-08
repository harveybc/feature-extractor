"""Train origin-covering FS4 donors for one frozen validation feature set.

Each feature is a separate process. The FS4 runner verifies governed corpus bytes,
physical GPU identity, task digest and any retained terminal before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from app import fs4_extractibility as X
from app.fs4_task_runner import TASK_SCHEMA


def frozen_members(freeze: dict, population: str, target: str) -> tuple[str, ...]:
    expected = freeze.get("freeze_sha256")
    actual = X.digest({k: v for k, v in freeze.items() if k not in ("freeze_sha256", "frozen_utc")})
    if expected != actual:
        raise ValueError("FREEZE_DIGEST_MISMATCH")
    winner = freeze["winners"][population][target]["RAW"]
    members = winner["members"]
    if not members or len(members) != len(set(members)):
        raise ValueError("FROZEN_MEMBERS_INVALID")
    return tuple(members)


def claim_for(feature: str, population: str, identity: str, fold: str, seed: int) -> dict:
    claim = {"schema": TASK_SCHEMA, "population_id": population, "identity": identity,
             "feature_id": feature, "fold_id": fold, "arm": "TRAINED_ENCODER_V2", "seed": seed}
    claim["task_id"] = X.task_digest(claim)
    return claim


def terminal_is_valid(path: Path, claim: dict) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    weights = result.get("artifacts", {}).get("chosen_weights_file")
    digest = result.get("artifacts", {}).get("chosen_weights_file_sha256")
    if not weights or not digest or not Path(weights).is_file():
        return False
    h = hashlib.sha256()
    with open(weights, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return (result.get("status") == "COMPLETE" and result.get("task_id") == claim["task_id"]
            and result.get("arm") == claim["arm"] and result.get("architecture", {}).get("last_latent_lag_rows") == 0
            and h.hexdigest() == digest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--fold", default="inner_2023")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args(argv)
    freeze = json.loads(Path(args.freeze).read_text())
    members = frozen_members(freeze, args.population, args.target)
    registry = X.CORPORA.get(args.identity)
    if not registry or registry["population_id"] != args.population or type(args.seed) is not int:
        raise ValueError("CORPUS_OR_SEED_INVALID")
    roles = dict(item.split("=", 1) for item in args.input)
    if set(roles) != set(registry["files"]):
        raise ValueError("CORPUS_ROLE_MISMATCH")
    root = Path(args.output_root)
    for feature in members:
        claim = claim_for(feature, args.population, args.identity, args.fold, args.seed)
        terminal = root / claim["task_id"] / "result.json"
        if terminal_is_valid(terminal, claim):
            print(json.dumps({"feature_id": feature, "status": "RETAINED", "task_id": claim["task_id"]}), flush=True)
            continue
        cmd = [sys.executable, "-m", "app.fs4_task_runner", "--output-root", str(root),
               "--gpu-uuid", args.gpu_uuid]
        for role, path in sorted(roles.items()):
            cmd.extend(["--input", f"{role}={path}"])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_uuid}
        proc = subprocess.run(cmd, input=X.canonical(claim), text=True, capture_output=True,
                              check=False, env=env)
        if proc.returncode != 0 or not terminal_is_valid(terminal, claim):
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"DONOR_FAILED: {feature}, exit {proc.returncode}")
        result = json.loads(terminal.read_text())
        print(json.dumps({"feature_id": feature, "status": "COMPLETE", "task_id": claim["task_id"],
                          "mae": result["metrics"]["mae"], "naive_mae": result["metrics"]["naive_mae"],
                          "cost": result["cost"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
