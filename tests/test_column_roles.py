"""What a run is allowed to feed a model: the declared columns, in the declared order.

P1 of `docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

The incident behind this: an ISO timestamp reached a float tensor because the loader took
"every column" as a feature. Removing that column from a fixture hides the problem; declaring
the roles removes it. These rules are written before the implementation.

The contract lives in the configuration, next to the file keys it describes:

    "column_roles": {"time": "DATE_TIME",
                     "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
                     "targets": ["CLOSE"],
                     "metadata": ["available_time"]}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("column_roles_under_test",
                                              REPO / "app" / "column_roles.py")
roles = importlib.util.module_from_spec(spec)
sys.modules["column_roles_under_test"] = roles
spec.loader.exec_module(roles)

pd = pytest.importorskip("pandas")

#: CLOSE is both a feature and the target here, so the contract has to say so out loud
#: (R1: absence is not permission). `WITHOUT_OPT_IN` is the same contract with that
#: declaration removed, which must now be refused.
CONTRACT = {"time": "DATE_TIME", "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "targets": ["CLOSE"], "metadata": ["available_time"],
            "allow_target_as_feature": True}
WITHOUT_OPT_IN = {key: value for key, value in CONTRACT.items()
                  if key != "allow_target_as_feature"}


def frame(columns):
    return pd.DataFrame({name: ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
                         if name in ("DATE_TIME", "available_time") else [1.0, 2.0]
                         for name in columns})


def test_only_declared_features_reach_the_model_in_the_declared_order():
    data = frame(["DATE_TIME", "CLOSE", "OPEN", "available_time", "HIGH", "LOW"])
    plan = roles.resolve({"column_roles": CONTRACT}, list(data.columns))
    assert plan.features == ["OPEN", "HIGH", "LOW", "CLOSE"]
    selected = roles.select_features(data, plan)
    assert list(selected.columns) == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert "DATE_TIME" not in selected.columns and "available_time" not in selected.columns


def test_a_permutation_of_the_file_does_not_change_the_feature_order():
    first = roles.select_features(frame(["OPEN", "HIGH", "LOW", "CLOSE", "DATE_TIME",
                                         "available_time"]),
                                  roles.resolve({"column_roles": CONTRACT},
                                                ["OPEN", "HIGH", "LOW", "CLOSE", "DATE_TIME",
                                                 "available_time"]))
    second = roles.select_features(frame(["CLOSE", "LOW", "HIGH", "OPEN", "available_time",
                                          "DATE_TIME"]),
                                   roles.resolve({"column_roles": CONTRACT},
                                                 ["CLOSE", "LOW", "HIGH", "OPEN",
                                                  "available_time", "DATE_TIME"]))
    assert list(first.columns) == list(second.columns) == ["OPEN", "HIGH", "LOW", "CLOSE"]


@pytest.mark.parametrize("values", [
    ["2024-01-01 00:00:00", "2024-01-01 04:00:00"],   # temporal metadata as text
    [1704067200, 1704081600],                          # and as a number
])
def test_temporal_metadata_never_becomes_a_feature(values):
    data = pd.DataFrame({"DATE_TIME": values, "OPEN": [1.0, 2.0], "HIGH": [1.0, 2.0],
                         "LOW": [1.0, 2.0], "CLOSE": [1.0, 2.0],
                         "available_time": values})
    plan = roles.resolve({"column_roles": CONTRACT}, list(data.columns))
    selected = roles.select_features(data, plan)
    assert list(selected.columns) == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert plan.metadata == ["available_time"] and plan.time == "DATE_TIME"


def test_an_undeclared_column_is_refused_by_name():
    with pytest.raises(roles.ColumnRoleError, match="surprise"):
        roles.resolve({"column_roles": CONTRACT},
                      ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time", "surprise"])


def test_a_declared_column_that_the_file_lacks_is_refused_by_name():
    with pytest.raises(roles.ColumnRoleError, match="LOW"):
        roles.resolve({"column_roles": CONTRACT},
                      ["DATE_TIME", "OPEN", "HIGH", "CLOSE", "available_time"])


COLUMNS = ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"]


def test_a_target_inside_the_feature_list_must_be_declared_on_purpose():
    """R1: absence is not permission.

    Musashi's first counterexample: `{"features": ["x", "y"], "targets": ["y"]}` was accepted
    and `y` reached the model. The default was True, so a contract that never mentions the
    overlap granted it. The opt-in is now required, and it must be a literal boolean.
    """
    with pytest.raises(roles.ColumnRoleError, match="CLOSE"):
        roles.resolve({"column_roles": dict(WITHOUT_OPT_IN)}, COLUMNS)

    allowed = roles.resolve({"column_roles": dict(CONTRACT)}, COLUMNS)
    assert allowed.target_is_feature is True, "the overlap must be visible, not incidental"
    assert "CLOSE" in allowed.features

    with pytest.raises(roles.ColumnRoleError, match="target"):
        roles.resolve({"column_roles": dict(CONTRACT, allow_target_as_feature=False)}, COLUMNS)


@pytest.mark.parametrize("value", ["true", "yes", 1, [True], {"ok": True}])
def test_only_a_literal_boolean_grants_the_overlap(value):
    """A truthy string or a 1 is a configuration accident, not a declared decision."""
    with pytest.raises(roles.ColumnRoleError, match="allow_target_as_feature"):
        roles.resolve({"column_roles": dict(CONTRACT, allow_target_as_feature=value)}, COLUMNS)


def test_a_column_cannot_be_metadata_and_a_feature_at_once():
    """R1, Musashi's second counterexample: `available_time` was declared both and reached
    the model because it happened to be numeric. Contradictory roles are refused by name."""
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"],
                "targets": ["CLOSE"], "metadata": ["available_time"],
                "allow_target_as_feature": False}
    with pytest.raises(roles.ColumnRoleError, match="available_time"):
        roles.resolve({"column_roles": contract}, ["DATE_TIME", "OPEN", "CLOSE",
                                                   "available_time"])


def test_the_time_column_cannot_also_be_a_feature_or_metadata():
    for contract in ({"time": "DATE_TIME", "features": ["DATE_TIME", "OPEN"]},
                     {"time": "DATE_TIME", "features": ["OPEN"],
                      "metadata": ["DATE_TIME"]}):
        with pytest.raises(roles.ColumnRoleError, match="DATE_TIME"):
            roles.resolve({"column_roles": contract}, ["DATE_TIME", "OPEN"])


def test_a_target_cannot_be_declared_metadata():
    """Metadata is never model input and a target is an output: the roles contradict."""
    contract = {"features": ["OPEN"], "targets": ["CLOSE"], "metadata": ["CLOSE"]}
    with pytest.raises(roles.ColumnRoleError, match="CLOSE"):
        roles.resolve({"column_roles": contract}, ["OPEN", "CLOSE"])


def test_a_repeated_declaration_is_refused_rather_than_deduplicated():
    """Silently collapsing `["OPEN", "OPEN"]` hides which of two intents was meant."""
    contract = {"features": ["OPEN", "OPEN"], "targets": ["CLOSE"], "metadata": []}
    with pytest.raises(roles.ColumnRoleError, match="OPEN"):
        roles.resolve({"column_roles": contract}, ["OPEN", "CLOSE"])


@pytest.mark.parametrize("contract", [
    {"features": "OPEN"},                                   # a string is not a list of names
    {"features": ["OPEN", None]},                           # a name that is not a name
    {"features": ["OPEN", 3]},
    {"features": ["OPEN"], "metadata": "available_time"},
    {"features": ["OPEN"], "time": ["DATE_TIME"]},
])
def test_a_malformed_role_list_is_refused(contract):
    with pytest.raises(roles.ColumnRoleError):
        roles.resolve({"column_roles": contract}, ["DATE_TIME", "OPEN", "available_time"])


def test_a_run_without_a_contract_is_refused_unless_the_migration_is_declared():
    columns = ["DATE_TIME", "OPEN", "CLOSE"]
    with pytest.raises(roles.ColumnRoleError, match="column_roles"):
        roles.resolve({}, columns)
    legacy = roles.resolve({"column_roles_migration": "LEGACY_ALL_COLUMNS_ARE_FEATURES"}, columns)
    assert legacy.migration == "LEGACY_ALL_COLUMNS_ARE_FEATURES"
    assert legacy.features == ["DATE_TIME", "OPEN", "CLOSE"], (
        "the legacy behaviour is kept exactly as it was, and named")


def test_the_plan_is_recorded_for_the_receipt():
    plan = roles.resolve({"column_roles": CONTRACT},
                         ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    record = plan.as_record()
    assert record["features"] == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert record["targets"] == ["CLOSE"] and record["time"] == "DATE_TIME"
    assert record["metadata"] == ["available_time"]
    assert len(record["contract_sha256"]) == 64


def test_changing_the_contract_changes_its_identity():
    first = roles.resolve({"column_roles": CONTRACT},
                          ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    changed = dict(CONTRACT, features=["OPEN", "HIGH", "LOW"])
    second = roles.resolve({"column_roles": changed},
                           ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    assert first.as_record()["contract_sha256"] != second.as_record()["contract_sha256"]


def test_a_non_numeric_feature_is_refused_before_it_reaches_a_tensor():
    """The incident itself: a timestamp declared as a feature must be stopped here."""
    contract = dict(CONTRACT, features=["OPEN", "HIGH", "LOW", "CLOSE", "available_time"],
                    metadata=[])
    data = frame(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    plan = roles.resolve({"column_roles": contract},
                         ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    with pytest.raises(roles.ColumnRoleError, match="available_time"):
        roles.select_features(data, plan)
