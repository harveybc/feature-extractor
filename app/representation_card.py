"""Representation candidate cards (PS3-R, subplan section 6, FS17).

``validate_card(card)`` returns a deep copy of the card with its DERIVED states
filled in, or raises :class:`CardRefusal` whose message starts with a named code.
It enforces the rules of ``representation_candidate_card.v1`` (predictor
``docs/contracts/representation_candidate_card.v1.schema.json`` at 2de7ea2a)
that a JSON schema cannot derive on its own:

* reconstruction ``NOT_APPLICABLE`` is a STATE, not a failure: an encoder
  without a decoder is never refused, and ``RECONSTRUCTION_FAILED`` never
  appears in its reasons;
* a ``pooled`` latent never satisfies the temporal contract
  (``VIOLATED_POOLED``); a card CLAIMING ``SATISFIED`` for it is refused
  (``POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT``); a ``patch`` latent needs a
  declared adapter (``ADAPTER_DECLARED``), never ``SATISFIED``;
* weights from a foreign corpus are never ``TRAIN_ONLY``
  (``FOREIGN_CORPUS_WEIGHTS_NOT_TRAIN_ONLY``); a ``FOREIGN_PRETRAINED`` card
  is at most ``ADMISSIBLE_AS_FOREIGN_COMPARATOR`` and only after a
  contamination audit found no overlap;
* an ``OPERATIONAL`` encoder may not be conditioned on future targets; a
  generative family is at most ``SECONDARY``;
* probe rows carry ``delta_probe = loss_random - loss_trained`` and
  ``preservation = loss_raw - loss_trained`` and must agree with them.

Pure standard library: no model is built, nothing is trained. Partial cards
(missing optional bookkeeping fields such as ``cost`` or ``limitations``) are
accepted by ``validate_card``; ``validate_card(card, complete=True)`` also
requires every field the schema requires.
"""
from __future__ import annotations

import copy
import math

SCHEMA = "representation_candidate_card.v1"
REQUIRED = ("schema", "candidate_id", "produced_at", "family", "architecture", "objective", "reference",
            "comparator", "corpus", "latent", "temporal_contract_state", "conditioning_contract",
            "evaluation", "cost", "admissibility", "limitations")
FAMILIES = {"autoencoder_control", "denoising_autoencoder", "contrastive", "masked_reconstruction",
            "latent_prediction", "foundation_pretrained", "generative_cvae", "generative_vae_gan",
            "supervised"}
GENERATIVE = {"generative_cvae", "generative_vae_gan"}
RECONSTRUCTION_STATES = {"MEASURED", "NOT_APPLICABLE", "FAILED", "NOT_EVALUATED"}
TEMPORAL_STATES = {"SATISFIED", "VIOLATED_POOLED", "ADAPTER_DECLARED", "NOT_EVALUATED"}
COMMON_GRID_STEPS = 24
PROBE_TOLERANCE = 1e-6


class CardRefusal(ValueError):
    """Raised with a message that starts with a named refusal code."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _require(condition, code, detail):
    if not condition:
        raise CardRefusal(code, detail)


def _temporal_state(latent, claimed, reasons):
    layout = latent.get("layout")
    _require(layout in ("temporal", "patch", "pooled"), "LATENT_LAYOUT_UNDECLARED",
             f"latent.layout must be temporal, patch or pooled, got {layout!r}")
    adapter = latent.get("grid_adapter")
    if layout == "pooled":
        _require(claimed in (None, "VIOLATED_POOLED", "NOT_EVALUATED"),
                 "POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT",
                 f"a pooled latent cannot claim {claimed!r}; it is VIOLATED_POOLED")
        reasons.append("POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT")
        return "VIOLATED_POOLED"
    if layout == "patch" or (adapter is not None):
        _require(isinstance(adapter, dict) and adapter.get("name") and adapter.get("validated_by"),
                 "GRID_ADAPTER_UNDECLARED",
                 "a patch latent, or any grid change, needs a declared adapter {name, validated_by}")
        _require(claimed in (None, "ADAPTER_DECLARED", "NOT_EVALUATED"), "ADAPTER_STATE_MISCLAIMED",
                 f"a latent behind an adapter cannot claim {claimed!r}")
        reasons.append("GRID_ADAPTER_DECLARED")
        return "ADAPTER_DECLARED"
    steps = latent.get("time_steps")
    if steps != COMMON_GRID_STEPS:
        _require(claimed != "SATISFIED", "GRID_MISMATCH_WITHOUT_ADAPTER",
                 f"a temporal latent of {steps} steps is not the {COMMON_GRID_STEPS}-step common grid")
        reasons.append("GRID_MISMATCH_WITHOUT_ADAPTER")
        return "NOT_EVALUATED"
    _require(claimed in (None, "SATISFIED", "NOT_EVALUATED"), "TEMPORAL_STATE_MISCLAIMED",
             f"a temporal {steps}-step latent cannot claim {claimed!r}")
    return "SATISFIED"


def _corpus(corpus, reasons):
    kind = corpus.get("kind")
    source = corpus.get("pretrained_weights_source")
    _require(kind in ("TRAIN_ONLY", "FOREIGN_PRETRAINED", "MIXED"), "CORPUS_KIND_UNDECLARED",
             f"corpus.kind must be TRAIN_ONLY, FOREIGN_PRETRAINED or MIXED, got {kind!r}")
    if kind == "TRAIN_ONLY":
        _require(source is None, "FOREIGN_CORPUS_WEIGHTS_NOT_TRAIN_ONLY",
                 f"weights pretrained on {source!r} cannot be labelled TRAIN_ONLY")
        _require(bool(corpus.get("train_folds")), "TRAIN_FOLDS_MISSING",
                 "a TRAIN_ONLY corpus names the TRAIN folds it was fitted on")
        corpus.setdefault("contamination_audit", "NOT_REQUIRED")
        return
    _require(isinstance(source, str) and source, "PRETRAINED_SOURCE_MISSING",
             f"a {kind} corpus names its pretrained weights source")
    corpus.setdefault("contamination_audit", "REQUIRED_NOT_DONE")
    reasons.append("FOREIGN_PRETRAINED_WEIGHTS")
    if corpus["contamination_audit"] != "DONE_NO_OVERLAP":
        reasons.append("CONTAMINATION_AUDIT_" + corpus["contamination_audit"])


def _probes(probes):
    for row in probes:
        for key in ("loss_trained", "loss_random", "naive"):
            _require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]), "PROBE_LOSS_INVALID",
                     f"probe {key} must be a finite number")
        _require(row.get("target") in ("Y_s", "Y_l", "Y_b", "J_policy"), "PROBE_TARGET_INVALID",
                 f"probe target {row.get('target')!r}; a self-forecast of the input is refused (FS02)")
        delta = row["loss_random"] - row["loss_trained"]
        if "delta_probe" in row:
            _require(abs(row["delta_probe"] - delta) <= PROBE_TOLERANCE * max(1.0, abs(delta)),
                     "DELTA_PROBE_INCONSISTENT", "delta_probe must equal loss_random - loss_trained")
        row["delta_probe"] = delta
        raw = row.get("loss_raw")
        preservation = None if raw is None else raw - row["loss_trained"]
        if row.get("preservation") is not None and preservation is not None:
            _require(abs(row["preservation"] - preservation) <= PROBE_TOLERANCE * max(1.0, abs(preservation)),
                     "PRESERVATION_INCONSISTENT", "preservation must equal loss_raw - loss_trained")
        row["preservation"] = preservation


def validate_card(card, *, complete=False):
    """Return the card with derived states, or raise CardRefusal naming the rule."""
    _require(isinstance(card, dict), "CARD_NOT_AN_OBJECT", "a card is a JSON object")
    out = copy.deepcopy(card)
    _require(out.get("schema") == SCHEMA, "SCHEMA_MISMATCH", f"schema must be {SCHEMA}")
    if complete:
        missing = [k for k in REQUIRED if k not in out]
        _require(not missing, "CARD_INCOMPLETE", f"missing required fields {missing}")
    family = out.get("family")
    _require(family in FAMILIES, "FAMILY_UNKNOWN", f"family {family!r}")
    reasons = []
    evaluation = out.setdefault("evaluation", {})
    reconstruction = evaluation.setdefault("reconstruction", {"state": "NOT_EVALUATED"})
    state = reconstruction.get("state")
    _require(state in RECONSTRUCTION_STATES, "RECONSTRUCTION_STATE_INVALID", f"state {state!r}")
    if state == "MEASURED":
        _require(all(isinstance(reconstruction.get(k), (int, float)) for k in ("mae_z", "mse_z")),
                 "RECONSTRUCTION_MEASURE_MISSING", "MEASURED reconstruction carries mae_z and mse_z")
    if state == "FAILED":
        reasons.append("RECONSTRUCTION_FAILED")
    if state == "NOT_APPLICABLE":
        reasons.append("RECONSTRUCTION_NOT_APPLICABLE_NO_DECODER")
    out["temporal_contract_state"] = _temporal_state(out.get("latent") or {}, out.get("temporal_contract_state"),
                                                     reasons)
    _corpus(out.setdefault("corpus", {}), reasons)
    conditioning = out.get("conditioning_contract")
    _require(conditioning in ("OPERATIONAL", "SYNTHETIC_OFFLINE"), "CONDITIONING_UNDECLARED",
             f"conditioning_contract {conditioning!r}")
    if family in GENERATIVE and conditioning == "OPERATIONAL" and out.get("conditioned_on_future_targets"):
        raise CardRefusal("OPERATIONAL_ENCODER_CONDITIONED_ON_FUTURE",
                          "an OPERATIONAL encoder may not be conditioned on future targets (FS03)")
    _probes(evaluation.setdefault("probes", []))

    kind = out["corpus"]["kind"]
    tstate = out["temporal_contract_state"]
    if kind != "TRAIN_ONLY":
        verdict = ("ADMISSIBLE_AS_FOREIGN_COMPARATOR"
                   if out["corpus"]["contamination_audit"] == "DONE_NO_OVERLAP" and tstate != "VIOLATED_POOLED"
                   else "NOT_ADMISSIBLE")
    elif tstate == "VIOLATED_POOLED" or state == "FAILED":
        verdict = "NOT_ADMISSIBLE"
    elif family in GENERATIVE:
        verdict = "SECONDARY"
    elif tstate == "ADAPTER_DECLARED":
        verdict = "ADMISSIBLE_WITH_ADAPTER"
    elif tstate == "NOT_EVALUATED":
        verdict = "NOT_EVALUATED"
    else:
        verdict = "ADMISSIBLE"
    previous = out.get("admissibility") or {}
    order = previous.get("pilot_order")
    out["admissibility"] = {"verdict": verdict, "reasons": sorted(set(reasons) | set(previous.get("reasons", []))
                                                                   - ({"RECONSTRUCTION_FAILED"}
                                                                      if state == "NOT_APPLICABLE" else set())),
                            "pilot_order": order}
    return out
