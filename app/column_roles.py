"""Which columns are features, which are targets, and which are only metadata.

P1 of `docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

An ISO timestamp reached a float tensor because the loader treated every column of a file as
a feature. Removing that column from a fixture hides the defect; declaring the roles removes
it. The contract lives in the configuration, beside the file keys it describes:

    "column_roles": {"time": "DATE_TIME",
                     "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
                     "targets": ["CLOSE"],
                     "metadata": ["available_time"],
                     "allow_target_as_feature": true}

Rules, all refusals rather than guesses:

* a model sees the declared features, in the **declared** order — a permutation of the file
  cannot change what the model reads;
* a column the file has and the contract does not mention is a refusal, by name: it may be a
  new feature, a leak, or a timestamp, and the configuration has to say which;
* a declared column the file lacks is a refusal, by name;
* a target inside the feature list is allowed only when the contract says so;
* a feature that is not numeric is refused **before** it reaches a tensor;
* a run with no contract is refused, unless it declares the legacy migration explicitly. The
  old behaviour is kept exactly as it was, and named, so that a heuristic selection can never
  quietly fold a target into the inputs again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

LEGACY = "LEGACY_ALL_COLUMNS_ARE_FEATURES"


class ColumnRoleError(ValueError):
    """The columns of a file and the declared roles do not agree."""


@dataclass
class RolePlan:
    features: list
    targets: list = field(default_factory=list)
    time: str | None = None
    metadata: list = field(default_factory=list)
    target_is_feature: bool = False
    migration: str | None = None
    contract: dict | None = None

    def as_record(self) -> dict:
        """What the receipt carries, so a result says which columns produced it."""
        body = {"features": list(self.features), "targets": list(self.targets),
                "time": self.time, "metadata": list(self.metadata),
                "target_is_feature": self.target_is_feature, "migration": self.migration}
        body["contract_sha256"] = hashlib.sha256(
            json.dumps(self.contract if self.contract is not None else body,
                       sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        return body


def _names(contract: dict, role: str) -> list:
    """The declared names for one role, refusing anything that is not a list of names.

    A bare string is the usual accident (`"features": "OPEN"` iterates into characters), and a
    non-string entry cannot be a column. Both are refused here rather than producing a
    confusing "the file does not carry declared columns: ['O', 'P', 'E', 'N']" later.
    """
    value = contract.get(role)
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ColumnRoleError(
            f"column_roles: {role!r} must be a list of column names, got {value!r}")
    names = list(value)
    bad = [name for name in names if not isinstance(name, str) or not name.strip()]
    if bad:
        raise ColumnRoleError(f"column_roles: {role!r} carries entries that are not column "
                              f"names: {bad}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ColumnRoleError(
            f"column_roles: {role!r} declares {repeated} more than once. A repeated "
            "declaration is refused rather than deduplicated: which of the two intents was "
            "meant cannot be recovered from the file")
    return names


def _refuse_contradictions(features, targets, metadata, time, contract) -> None:
    """A column has one role, except the target/feature overlap the contract may declare.

    Metadata is never model input, so a column declared metadata *and* feature contradicts
    itself; it reached the model before only because it happened to be numeric. The time
    column is the run's clock and cannot also be an input or a label.
    """
    clash = [name for name in features if name in metadata]
    if clash:
        raise ColumnRoleError(
            f"contradictory roles for {clash}: declared both a feature and metadata. Metadata "
            "never reaches the model; a value that must reach it is a feature under its own "
            "name, including a calendar feature derived on purpose")
    clash = [name for name in targets if name in metadata]
    if clash:
        raise ColumnRoleError(
            f"contradictory roles for {clash}: declared both a target and metadata. A target "
            "is what the run predicts and metadata is what it never reads")
    if time is not None:
        where = [role for role, names in (("features", features), ("targets", targets),
                                          ("metadata", metadata)) if time in names]
        if where:
            raise ColumnRoleError(
                f"contradictory roles for {time!r}: it is the time column and is also declared "
                f"in {where}. Availability or event time does not become model input by being "
                "declared twice; derive an explicit calendar feature under its own name")


def _opt_in(contract: dict) -> bool:
    """Whether the contract *declares* the target/feature overlap.

    Only a literal boolean counts: `"true"`, `1` or `["yes"]` are configuration accidents that
    a truthiness test would silently read as a decision.
    """
    if "allow_target_as_feature" not in contract:
        return False
    value = contract["allow_target_as_feature"]
    if not isinstance(value, bool):
        raise ColumnRoleError(
            f"column_roles: allow_target_as_feature must be a literal true or false, got "
            f"{value!r}. A truthy value is not a declaration")
    return value


def resolve(config: dict, columns) -> RolePlan:
    """Build the plan for these file columns, or refuse with the column that is wrong."""
    columns = list(columns)
    contract = (config or {}).get("column_roles")
    if not contract:
        migration = (config or {}).get("column_roles_migration")
        if migration == LEGACY:
            return RolePlan(features=list(columns), migration=LEGACY, contract={"migration": LEGACY})
        raise ColumnRoleError(
            "this run declares no column_roles: say which columns are features, targets, time "
            f"and metadata, or declare column_roles_migration: {LEGACY!r} to keep the old "
            "behaviour deliberately")

    features = _names(contract, "features")
    targets = _names(contract, "targets")
    metadata = _names(contract, "metadata")
    time = contract.get("time")
    if time is not None and (not isinstance(time, str) or not time.strip()):
        raise ColumnRoleError(
            f"column_roles: 'time' must be a single column name, got {time!r}")
    if not features:
        raise ColumnRoleError("column_roles declares no feature")

    _refuse_contradictions(features, targets, metadata, time, contract)

    declared = [*features, *targets, *metadata] + ([time] if time else [])
    seen, ordered = set(), []
    for name in declared:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    missing = [name for name in ordered if name not in columns]
    if missing:
        raise ColumnRoleError(f"the file does not carry declared columns: {missing}")
    undeclared = [name for name in columns if name not in seen]
    if undeclared:
        raise ColumnRoleError(
            f"the file carries columns this run does not declare: {undeclared}. Declare them as "
            "features, targets or metadata; an undeclared column is not silently dropped and "
            "not silently used")

    declared_opt_in = _opt_in(contract)  # validated even with no overlap: a malformed
                                         # declaration is a defect wherever it appears
    overlap = [name for name in targets if name in features]
    if overlap and not declared_opt_in:
        raise ColumnRoleError(
            f"a target is also declared a feature: {overlap}. This is allowed only when the "
            "contract says so on purpose: set allow_target_as_feature to true. Its absence is "
            "not permission, and the declaration alone does not establish that the values fed "
            "to the model are known at the decision time of their own target")
    return RolePlan(features=features, targets=targets, time=time, metadata=metadata,
                    target_is_feature=bool(overlap), contract=contract)


def select_features(frame, plan: RolePlan):
    """The feature frame, in the declared order, with every column checked numeric."""
    import pandas as pd

    missing = [name for name in plan.features if name not in frame.columns]
    if missing:
        raise ColumnRoleError(f"the frame does not carry declared features: {missing}")
    selected = frame.loc[:, list(plan.features)]
    if plan.migration == LEGACY:
        return selected
    bad = [name for name in selected.columns
           if not pd.api.types.is_numeric_dtype(selected[name])]
    if bad:
        raise ColumnRoleError(
            f"declared features that are not numeric: {bad}. A timestamp or a label cannot be "
            "fed to a model as a number; declare it as metadata or convert it on purpose")
    return selected


def record_of(config: dict, columns) -> dict:
    """Convenience for callers that only want the receipt entry."""
    return resolve(config, columns).as_record()
