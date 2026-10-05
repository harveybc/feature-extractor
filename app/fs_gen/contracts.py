"""Input contract for the FS-GEN generator (SYNTHETIC_OFFLINE, work plan §6.2 / FS03).

The generator conditions ONLY on information available at t: known time encodings, the observed
mask, delta time and past values of the same feature. The future target and economic-calendar
events are refused as conditions. All rows must lie inside TRAIN (<= train_end_ts).
"""
from __future__ import annotations

import numpy as np

from app import univariate_temporal as U

CONTRACT = "SYNTHETIC_OFFLINE"
ALLOWED_CONDITIONS = tuple(U.INPUT_NAMES)  # signal, observed_mask, delta_time, calendar


def check_generator_conditions(mapping) -> None:
    """Refuse target-like names and anything outside the four declared conditions."""
    U.check_no_target(mapping)
    extra = set(mapping) - set(ALLOWED_CONDITIONS)
    if extra:
        raise U.ContractError(f"generator conditions must be a subset of {ALLOWED_CONDITIONS}; extra={sorted(extra)}")


def check_calendar_columns(names) -> None:
    """Calendar is limited to time encodings and session/holiday columns published at t (I11)."""
    for n in names:
        U.check_known_calendar_name(n)


def assert_train_only(ts, train_end_ts: int) -> None:
    ts = np.asarray(ts, np.int64)
    if ts.size and int(ts.max()) > int(train_end_ts):
        raise U.FoldScopeError(f"{int((ts > train_end_ts).sum())} rows after train_end_ts: TEST is never read")
