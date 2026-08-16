# AGENTS.md — feature-extractor

Instructions for AI coding agents working in this repository.
Human-facing documentation is in [`README.md`](README.md).

## Project overview

feature-extractor trains autoencoders (Keras/TensorFlow) on sliding windows of
preprocessed financial time-series CSVs, evaluates reconstruction quality, and
saves the encoder and decoder as model files. Downstream repositories load the
saved encoder as a learned feature extractor. Encoders and decoders are plugins
selected by name; the training/prediction pipeline is driven by one flat JSON
config.

It does not engineer domain features or labels (feature-eng), does not normalize
or split datasets (preprocessor), and does not train predictive models or serve
predictions (predictor, prediction_provider).

**Read this before planning any work here:** the CLI pipeline cannot run from
this repository alone, and the test suite does not pass. Both are documented
below with the exact observed behavior.

## Agent quickstart (install → run → show the user results)

Verified on Python 3.12.13 with TensorFlow 2.21.0 / Keras 3.15.0, pandas, numpy.

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow pandas numpy h5py scipy keras-multi-head
pip install -e .
```

`requirements.txt` pins `tensorflow-gpu`, a legacy package name that no longer
resolves on modern TensorFlow — install `tensorflow` instead. `pip install -e .`
is required: encoder and decoder plugins are resolved through the
`feature_extractor.encoders` / `feature_extractor.decoders` entry-point groups.

Use a dedicated virtualenv: this package installs a generic top-level `app`
package shared by name with sibling repositories.

If you have a GPU that is busy with other work, prefix commands with
`CUDA_VISIBLE_DEVICES=` to force CPU (all timings below are CPU).

### 2. Smoke test

The pytest suite is broken — do not use it as a health signal:

```bash
python -m pytest tests -q --continue-on-collection-errors
# observed: 10 failed, 2 passed, 6 errors in ~4s
```

The collection errors and `TypeError`/`AttributeError` failures come from test
modules written against an older plugin API. Use these instead as the real
smoke check (both verified, exit 0):

```bash
python -c "import app.plugins.encoder_plugin_cnn, app.plugins.decoder_plugin_cnn, app.plugins.encoder_plugin_vae"
python -m app.main --help
```

### 3. Representative run

The CLI (`python -m app.main`, or the `feature_extractor` console script) is
real, but **its pipeline cannot complete from this repository alone**. Verified:

```bash
python -m app.main --load_config examples/config/phase_4_2/phase_4_2_small.json
# loads the vae_small encoder and decoder, then:
#   Loading Plugin ..stl_preprocessor
#   Failed to load or initialize Preprocessor Plugin:
#     Plugin stl_preprocessor not found in group preprocessor.plugins.
# exit code 1
```

`app/main.py` loads its windowing/decomposition stage from the **external**
`preprocessor.plugins` entry-point group (default name `stl_preprocessor`),
which is provided by the predictor package, not by this one. Without predictor
installed alongside, every full pipeline run exits 1 at that point. Training
runs from these configs are also long (`"epochs": 1000` with early stopping over
25 200 training steps) and were not executed here.

So the representative run is the library path — build an encoder plugin and
encode windows of a bundled dataset directly:

```python
# quickstart.py — run from the repository root
import numpy as np, pandas as pd, pathlib
from app.plugins.encoder_plugin_cnn import Plugin as Encoder

WINDOW, LATENT, N = 32, 8, 64
df = pd.read_csv("examples/data/phase_3/normalized_d3.csv").head(WINDOW + N).drop(columns=["DATE_TIME"])
windows = np.stack([df.values[i:i + WINDOW] for i in range(N)]).astype("float32")

encoder = Encoder()
encoder.configure_size(WINDOW, LATENT, windows.shape[2], True, config={"window_size": WINDOW})
encoded = encoder.encode(windows)          # untrained weights — this checks wiring, not quality

out = pathlib.Path("quickstart_out"); out.mkdir(exist_ok=True)
pd.DataFrame(encoded.reshape(len(encoded), -1)).to_csv(out / "encoded_sample.csv", index=False)
print(f"windows {windows.shape} -> encoded {encoded.shape}; wrote {out / 'encoded_sample.csv'}")
```

```bash
CUDA_VISIBLE_DEVICES= python quickstart.py
```

Observed: `windows (64, 32, 44) -> encoded (64, 8, 64)`, a few seconds, writes
`quickstart_out/encoded_sample.csv` (64 rows × 512 flattened columns). Note that
`configure_size(input_shape, interface_size, num_channels, use_sliding_windows,
config=...)` requires the `config` dict — it reads `window_size` from it and
raises `AttributeError` if it is `None`.

### 4. Analytics

No analytics or visualization step runs standalone. The plot outputs named in
the configs (`loss_plot_file`, `predictions_plot_file`, `stl_plot_file`,
`wavelet_plot_file`, `tapper_plot_file`) are produced by the full training
pipeline, which needs the external preprocessor plugin. Committed examples of
those artifacts are under `examples/results/phase_4_1/`, `phase_4_2/`,
`phase_4_3/` — read them rather than regenerating them.

### 5. Final message to the user

> The encoder ran. Output is `quickstart_out/encoded_sample.csv` — 64 encoded
> windows, each a flattened 8 × 64 representation of 32 timesteps × 44 features
> read from `examples/data/phase_3/normalized_d3.csv`. There is no UI. The
> weights are untrained, so this proves the plugin wiring and shapes, not
> reconstruction quality.
>
> Full training through `python -m app.main --load_config
> examples/config/phase_4_2/phase_4_2_small.json` additionally needs the
> predictor package installed (it supplies the `stl_preprocessor` plugin) and
> writes models, metrics and plots under `examples/results/phase_4_2/`;
> committed examples of those artifacts are already there.
>
> Suggested first analysis: plot the first encoded channel (column `0` of
> `encoded_sample.csv`) against the raw `CLOSE` column of the same 64 windows
> from `examples/data/phase_3/normalized_d3.csv`. With untrained weights they
> should look unrelated — rerun the same plot after training an encoder and the
> difference tells you what the encoder learned to track.

## Build, test and lint commands

```bash
pip install -e .                                  # required: registers encoder/decoder plugins
python -m app.main --help                         # CLI reference (exits 0)
feature_extractor --help                          # console script installed by setup.py
python -m pytest tests -q --continue-on-collection-errors   # 10 failed, 2 passed, 6 errors
sh feature-extractor.sh <args>                    # sets PYTHONPATH=./ and runs app/main.py
```

CI: `.github/workflows/python-package.yml` runs `flake8` (first pass fails only
on E9/F63/F7/F82; second pass is `--exit-zero --max-line-length=127`) and then
bare `pytest`, on push and pull requests to `master`, across Python 3.8–3.11.
Given the local test state above, expect that job to be red; its actual status
on GitHub was not checked. No formatter is configured.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | CLI entry point: merge config → load encoder/decoder/preprocessor plugins → run pipeline |
| `app/config.py`, `app/cli.py`, `app/config_merger.py`, `app/config_handler.py` | Defaults, flags, merge order, local/remote config |
| `app/autoencoder_manager.py`, `app/autoencoder_helper.py` | Model assembly and training loop |
| `app/data_processor.py` | Pipeline: preprocessing call, training, prediction, evaluation |
| `app/reconstruction.py`, `app/data_handler.py` | Reconstruction and CSV IO |
| `app/plugins/` | Encoder/decoder plugins: `ann`, `cnn`, `lstm`, `transformer`, `vae`, `vae_small` (plus unregistered `*_working.py` spares) |
| `examples/config/` | Phase 3.2 / 4.1 / 4.2 JSON configs |
| `examples/data/phase_3/` | Committed normalized `base_d1..d6` / `normalized_d1..d6` CSVs |
| `examples/results/` | Committed model artifacts, metrics and plots from historical runs |
| `examples/scripts/` | Shell drivers for the phase configs |
| `tests/` | pytest suite (broken, see Smoke test) |

## Conventions and constraints

- **Config-driven**: defaults in `app/config.py` → `--load_config` JSON → CLI
  flags → unknown `--flags` (merged too, so any key such as `--epochs` is
  settable from the CLI even though it has no declared argument). The merge runs
  again after plugins load so their `plugin_params` defaults participate.
- **Plugin architecture**: encoders and decoders come from the
  `feature_extractor.encoders` / `feature_extractor.decoders` groups declared in
  `setup.py`; the preprocessing stage comes from the external
  `preprocessor.plugins` group. A plugin implements `plugin_params`,
  `set_params`, `configure_size`, `train`, `encode`/`decode`, `save`, `load`.
- **Broken entry points**: `rnn` and `cnn_signed` are registered in `setup.py`
  but the modules do not exist; selecting them fails with `ModuleNotFoundError`.
  Do not "fix" them by pointing at the `*_working.py` files without checking
  what those actually implement.
- **Data contract**: input CSVs have a `DATE_TIME` column plus numeric feature
  columns, already normalized by the preprocessor repository; the phase_3
  samples have 44 feature columns. `use_normalization_json` points at the
  preprocessor's persisted normalization parameters so encoders train in the
  same value space as downstream consumers.
- **Artifacts are a contract**: predictor's phase 3/4 configs reference saved
  encoders through their `feature_extractor_file` key. Keep artifact paths and
  file names stable once published.
- No credentials belong in configs; `--username`/`--password` exist only for the
  optional remote config/log endpoints.

## Do not touch

- `examples/results/` — committed model files (`*.keras`, `*.h5`), metrics and
  plots from historical runs; other repositories reference these paths.
- `examples/data/phase_3/` — committed input fixtures (~180 k rows total).
- `feature_extractor.egg-info/`, `__pycache__/`, `.pytest_cache/` — generated.
- `quickstart_out/` — generated by the example above; not part of the repository.
- Sibling repositories (predictor, preprocessor, feature-eng). In particular, do
  not vendor a copy of `stl_preprocessor` here to work around the missing
  plugin group — that fork would silently diverge from predictor's.
