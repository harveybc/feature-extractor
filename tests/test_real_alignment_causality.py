"""The extractor's alignment, and the conditioning vector it decides.

R2 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "...preprocessing/window construction and extractor alignment. Check output
     identity/timestamps, not merely shape or unequal numeric values."

`app/preprocessing_api.align_timestamps` exists because a real misalignment was observed on
2026-09-14: predictor's preprocessor trims the trailing windows that have no future to predict
but does not trim the parallel date vector, so `phase_4_2_small` produced 24,913 train windows
against 25,057 dates — a difference of exactly the largest horizon, 144.

What that function decides is which timestamp belongs to which window, and every conditioning
vector in `calculate_datetime_features` is built from its answer. So the rules below check
**identity** — which timestamp ends up against which window — rather than lengths.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from app.preprocessing_api import PreprocessorApiError, align_timestamps
from app.data_processor import calculate_datetime_features

HORIZON = 144


def dates(rows):
    return pd.date_range("2024-01-01", periods=rows, freq="h")


def test_the_kept_timestamps_are_the_FIRST_ones_not_the_last():
    """The windows that survive are the oldest ones; their dates are the oldest ones too.

    Trimming from the wrong end would keep the same COUNT and pass any length check, while
    shifting every window against a timestamp 144 hours away from its own.
    """
    stamps = dates(25057)
    kept = align_timestamps(stamps, 24913, split="train")
    assert len(kept) == 24913
    assert kept[0] == stamps[0], "the first window keeps the first timestamp"
    assert kept[-1] == stamps[24912], "and the last kept window keeps ITS own timestamp"
    assert (kept == stamps[:24913]).all()


def test_the_dropped_tail_is_exactly_the_largest_horizon():
    """The observed case, as a rule: 25,057 - 24,913 = 144."""
    stamps = dates(25057)
    kept = align_timestamps(stamps, 25057 - HORIZON, split="train")
    assert len(stamps) - len(kept) == HORIZON


def test_an_aligned_vector_is_returned_untouched():
    stamps = dates(1000)
    kept = align_timestamps(stamps, 1000, split="validation")
    assert kept is stamps


def test_too_few_dates_is_refused_rather_than_padded():
    """There is no alignment that does not invent a timestamp, so it refuses."""
    with pytest.raises(PreprocessorApiError, match="invent"):
        align_timestamps(dates(900), 1000, split="train")


def test_a_missing_date_vector_stays_missing():
    assert align_timestamps(None, 10, split="train") is None


def test_the_conditioning_vector_follows_the_window_it_belongs_to():
    """Identity, end to end: the features of window i must be the features of ITS hour.

    `calculate_datetime_features` turns each timestamp into ten cyclical values. If alignment
    ever trimmed from the wrong end, these would be the features of a different hour and
    nothing about the shapes would complain.
    """
    stamps = dates(500)
    kept = align_timestamps(stamps, 500 - HORIZON, split="train")
    produced = calculate_datetime_features(pd.DatetimeIndex(kept))
    expected = calculate_datetime_features(pd.DatetimeIndex(stamps[:500 - HORIZON]))
    np.testing.assert_array_equal(produced, expected)
    # and the first row really is hour zero of 2024-01-01, not hour 144
    assert produced[0][0] == pytest.approx(np.sin(0.0))
    assert produced[0][1] == pytest.approx(np.cos(0.0))


def test_trimming_from_the_wrong_end_would_be_caught_by_that_same_check():
    """Check on the check: the mutation keeps the count and breaks the identity."""
    stamps = dates(500)
    wrong_end = stamps[HORIZON:]          # same length as the correct answer
    right = align_timestamps(stamps, 500 - HORIZON, split="train")
    assert len(wrong_end) == len(right), "the mutation is invisible to any length check"
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            calculate_datetime_features(pd.DatetimeIndex(wrong_end)),
            calculate_datetime_features(pd.DatetimeIndex(right)))


def test_extending_the_series_does_not_change_the_earlier_alignment():
    """Append-only: more data later cannot move a timestamp already assigned to a window."""
    short = align_timestamps(dates(500), 500 - HORIZON, split="train")
    longer = align_timestamps(dates(900), 900 - HORIZON, split="train")
    assert (np.asarray(longer)[:len(short)] == np.asarray(short)).all()
