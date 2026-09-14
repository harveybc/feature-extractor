"""The contract, exercised through the loader feature-extractor actually runs.

R1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`.
This consumer had **no** column-role contract at all: `app.data_handler.load_csv` reads every
column as text and then coerces it with `fillna(0)`, so a timestamp taken for a feature became
a column of zeros and the run continued. These rules drive that function with files on disk.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from app.column_roles import ColumnRoleError
from app.data_handler import load_csv

CONTRACT = {"time": "DATE_TIME", "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "targets": ["CLOSE"], "metadata": ["available_time"],
            "allow_target_as_feature": True}


def csv_at(tmp_path, columns, rows=4):
    stamps = [f"2024-01-0{i + 1} 00:00:00" for i in range(rows)]
    body = {column: (stamps if column in ("DATE_TIME", "available_time")
                     else [float(i + 1) for i in range(rows)]) for column in columns}
    path = tmp_path / "series.csv"
    pd.DataFrame(body).to_csv(path, index=False)
    return str(path)


def test_the_declared_features_reach_the_run_in_the_declared_order(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "CLOSE", "OPEN", "available_time", "HIGH", "LOW"])
    config = dict(column_roles=CONTRACT)
    data = load_csv(path, headers=True, config=config)
    assert list(data.columns) == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert data.index.name == "DATE_TIME", "the time column stays the index, never a feature"
    record = config["column_roles_applied"]["input_file"]
    assert record["features"] == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert len(record["contract_sha256"]) == 64


def test_an_undeclared_column_stops_the_load(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time",
                             "SURPRISE"])
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_csv(path, headers=True, config=dict(column_roles=CONTRACT))


def test_the_target_overlap_needs_the_declaration(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    without = {key: value for key, value in CONTRACT.items()
               if key != "allow_target_as_feature"}
    with pytest.raises(ColumnRoleError, match="CLOSE"):
        load_csv(path, headers=True, config=dict(column_roles=without))


def test_a_contradictory_metadata_feature_stops_the_load(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"],
                "metadata": ["available_time"]}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_csv(path, headers=True, config=dict(column_roles=contract))


def test_a_timestamp_declared_as_a_feature_is_refused_not_zeroed(tmp_path):
    """The defect this port exists to close: `fillna(0)` turned a timestamp into zeros."""
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"], "metadata": []}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_csv(path, headers=True, config=dict(column_roles=contract))


def test_a_genuinely_missing_value_is_still_filled_not_refused(tmp_path):
    """An empty cell is missing data, not a non-numeric feature: the old behaviour stands."""
    path = tmp_path / "gap.csv"
    pd.DataFrame({"DATE_TIME": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                  "OPEN": [1.0, None]}).to_csv(path, index=False)
    contract = {"time": "DATE_TIME", "features": ["OPEN"], "metadata": []}
    data = load_csv(str(path), headers=True, config=dict(column_roles=contract))
    assert list(data["OPEN"]) == [1.0, 0.0]


def test_a_run_without_a_contract_is_refused(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN"])
    with pytest.raises(ColumnRoleError, match="column_roles"):
        load_csv(path, headers=True, config={})


def test_a_caller_that_passes_no_config_keeps_the_previous_behaviour(tmp_path):
    """The port is additive: existing callers are not broken into a refusal by this change.

    They also gain nothing — a run that wants the guarantee has to pass its contract.
    """
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    data = load_csv(path, headers=True)
    assert list(data.columns) == ["OPEN", "available_time"]


def test_the_declared_legacy_migration_still_loads_everything(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "CLOSE"])
    config = {"column_roles_migration": "LEGACY_ALL_COLUMNS_ARE_FEATURES"}
    data = load_csv(path, headers=True, config=config)
    assert list(data.columns) == ["OPEN", "CLOSE"]
    assert config["column_roles_applied"]["input_file"]["migration"] == (
        "LEGACY_ALL_COLUMNS_ARE_FEATURES")
