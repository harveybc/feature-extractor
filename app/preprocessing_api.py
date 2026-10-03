"""How this application calls a preprocessor plugin, across the two APIs that exist.

`preprocessor.plugins` is a shared entry-point group. The plugin this repository uses,
predictor's `stl_preprocessor`, changed its signature in predictor `9b7d611` (2026-02-18):

    run_preprocessing(self, config)                   # before
    run_preprocessing(self, target_plugin, config)    # since

Both shapes are in the wild across the sibling repositories, so the call site inspects the
signature instead of assuming one. The target plugin is loaded from predictor's
`target.plugins` group and named by `target_plugin` in the configuration; when the plugin
requires one and none was loaded, the refusal says which key to set rather than failing
inside the preprocessor with a TypeError.
"""

from __future__ import annotations

import inspect


class PreprocessorApiError(RuntimeError):
    """The preprocessor plugin cannot be called with what this application has."""


def takes_target_plugin(method) -> bool:
    """True when `run_preprocessing` expects a target plugin before the configuration."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # a builtin or a C callable: assume the current API
        return True
    names = [name for name in signature.parameters if name != "self"]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()):
        return True
    return len(names) >= 2 or "target_plugin" in names


def run_preprocessing(preprocessor_plugin, config, target_plugin=None):
    """Call the plugin under whichever API it declares, or refuse with a usable reason."""
    method = getattr(preprocessor_plugin, "run_preprocessing", None)
    if method is None:
        raise PreprocessorApiError(
            f"{type(preprocessor_plugin).__name__} has no run_preprocessing method")
    if not takes_target_plugin(method):
        return method(config)
    if target_plugin is None:
        raise PreprocessorApiError(
            f"{type(preprocessor_plugin).__name__}.run_preprocessing requires a target plugin; "
            "set 'target_plugin' in the configuration to a name registered in the "
            "'target.plugins' entry-point group (for example 'default_target')")
    return method(target_plugin, config)


def align_timestamps(dates, n_samples, *, split: str):
    """Trim a preprocessor's date vector to the windows that survived target alignment.

    predictor's `stl_preprocessor` aligns the sliding windows to the targets with
    `windows[:target_length]` (`_align_sliding_windows_with_targets`), which drops the
    trailing windows that have no future to predict — `max(predicted_horizons)` of them —
    but it does not trim the parallel `*_dates` vectors. Observed on 2026-09-14 with
    `phase_4_2_small`: X_train 24,913 windows against 25,057 dates, X_val 6,013 against
    6,157; the difference is exactly the largest horizon, 144, on both splits.

    So the first `n_samples` dates are the ones that belong to the kept windows. Anything
    else is refused rather than trimmed from the other end: a silent misalignment here
    would shift every conditioning vector against its window.
    """
    if dates is None:
        return None
    length = len(dates)
    if length == n_samples:
        return dates
    if length > n_samples:
        return dates[:n_samples]
    raise PreprocessorApiError(
        f"the preprocessor returned {length} dates for {n_samples} {split} windows; "
        "there is no alignment that does not invent timestamps")
