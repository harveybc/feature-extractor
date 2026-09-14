# feature-extractor

Autoencoder trainer for financial time-series windows. `feature-extractor`
trains encoder/decoder pairs (Keras/TensorFlow) on sliding windows of
preprocessed CSV data, evaluates reconstruction quality, and saves the trained
encoder and decoder models. The saved encoder artifacts are used as learned
feature extractors by downstream training phases: the
[predictor](https://github.com/harveybc/predictor) phase 3/4 configurations
load them through their `feature_extractor_file` setting, and the same encoder
workflow produces learned-representation inputs for the owner's private
data-engineering pipeline.

## Status

**Publication update (2026-09-14):** this branch includes the
[governed-run adapter](tools/governed_run.py), profile test and agent setup
guide. The adapter delegates to [data-gov](https://github.com/harveybc/data-gov);
set `DATA_GOV_CHECKOUT` when the checkouts are not siblings. A passing profile
test is not a completed autoencoder training run. The external preprocessor
dependency and incomplete artifact coverage remain integration limitations.

This autoencoder implementation is an antecedent of the
[doctoral proposal on modular temporal representations](https://github.com/harveybc/predictor/blob/master/docs/propuesta_doctoral_representaciones_temporales_modulares.pdf),
not an implementation of all its proposed experiments. Reconstruction quality
alone does not establish downstream forecasting or RL performance.
See the [repository map](https://github.com/harveybc/predictor/blob/master/docs/RESEARCH_STACK.md)
for the separation between feature engineering, learned representations and training.

**Active component** of the harveybc trading stack (package
`feature-extractor` 0.1.0). Maintained as the representation-learning stage
between preprocessing and predictor training.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md`, inspect the documented preprocessor dependency, and run the
> import/CLI checks first. Attempt a bounded CPU encoding example only when
> those dependencies are satisfied. Report actual outputs or the exact blocker;
> do not describe a configured plugin as a completed training run.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

## Role and non-responsibilities

`feature-extractor` trains and evaluates autoencoders and exports encoder /
decoder model artifacts. It does **not**:

- engineer domain features or labels — that is
  [feature-eng](https://github.com/harveybc/feature-eng);
- normalize or split datasets — that is
  [preprocessor](https://github.com/harveybc/preprocessor);
- train predictive models or serve predictions — that is
  [predictor](https://github.com/harveybc/predictor) and
  [prediction_provider](https://github.com/harveybc/prediction_provider);
- generate synthetic data — that is
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen).

## Architecture

[`app/main.py`](app/main.py) merges configuration (defaults in
[`app/config.py`](app/config.py), optional JSON config via `--load_config`,
CLI flags), loads one encoder plugin and one decoder plugin by name, loads a
windowing/decomposition preprocessing plugin, and runs the autoencoder
training/evaluation pipeline in
[`app/autoencoder_manager.py`](app/autoencoder_manager.py) and
[`app/data_processor.py`](app/data_processor.py).

Encoder/decoder plugins are discovered through the entry-point groups
`feature_extractor.encoders` and `feature_extractor.decoders` declared in
[`setup.py`](setup.py):

| Name | Encoder | Decoder | Status |
|---|---|---|---|
| `ann` (also `default`) | [`app/plugins/encoder_plugin_ann.py`](app/plugins/encoder_plugin_ann.py) | [`app/plugins/decoder_plugin_ann.py`](app/plugins/decoder_plugin_ann.py) | working |
| `cnn` | [`app/plugins/encoder_plugin_cnn.py`](app/plugins/encoder_plugin_cnn.py) | [`app/plugins/decoder_plugin_cnn.py`](app/plugins/decoder_plugin_cnn.py) | working |
| `lstm` | [`app/plugins/encoder_plugin_lstm.py`](app/plugins/encoder_plugin_lstm.py) | [`app/plugins/decoder_plugin_lstm.py`](app/plugins/decoder_plugin_lstm.py) | working |
| `transformer` | [`app/plugins/encoder_plugin_transformer.py`](app/plugins/encoder_plugin_transformer.py) | [`app/plugins/decoder_plugin_transformer.py`](app/plugins/decoder_plugin_transformer.py) | working |
| `vae` | [`app/plugins/encoder_plugin_vae.py`](app/plugins/encoder_plugin_vae.py) | [`app/plugins/decoder_plugin_vae.py`](app/plugins/decoder_plugin_vae.py) | working (conditional VAE with configurable target features) |
| `vae_small` | [`app/plugins/encoder_plugin_vae_small.py`](app/plugins/encoder_plugin_vae_small.py) | [`app/plugins/decoder_plugin_vae_small.py`](app/plugins/decoder_plugin_vae_small.py) | working |
| `rnn` | — | — | **broken**: registered in `setup.py` but the modules do not exist |
| `cnn_signed` | — | — | **broken**: registered in `setup.py` but the modules do not exist |

The windowing/decomposition step is loaded from the **external**
`preprocessor.plugins` entry-point group (default plugin name
`stl_preprocessor`), which the installed
[predictor](https://github.com/harveybc/predictor) package provides; running
the pipeline therefore requires that package to be installed alongside this
one.

## Requirements

- Python 3 (no `python_requires` pin in [`setup.py`](setup.py); verified below
  under Python 3.12.13).
- TensorFlow/Keras plus the packages listed in
  [`requirements.txt`](requirements.txt) (`tensorflow-gpu`, `numpy`, `pandas`,
  `h5py`, `scipy`, `keras-multi-head`, ...). GPU use is optional; CUDA setup
  notes are in [`README_CUDA.md`](README_CUDA.md).

## Installation

Unverified (not executed in a clean environment for this README):

```bash
git clone https://github.com/harveybc/feature-extractor.git
cd feature-extractor
pip install -r requirements.txt
pip install -e .
# plus the predictor package, which provides the preprocessor.plugins group
```

Verified in the maintainer environment (Python 3.12.13, TensorFlow with GPU,
2026-08-10):

- `python -c "import app.plugins.encoder_plugin_vae, app.plugins.decoder_plugin_vae, app.plugins.encoder_plugin_cnn"`
  → `fe plugin imports OK`.
- `python -m app.main --help` → prints the full CLI usage.
- `python -c "import app.plugins.encoder_plugin_rnn"` →
  `ModuleNotFoundError` (see Limitations).

## Quickstart

The repository owns phase-3/4 example configurations and normalized sample
datasets:

```bash
# Train the phase-3.2 CNN autoencoder on repo-owned normalized data
python -m app.main --load_config examples/config/phase_3_2/phase_3_2_cnn_1h_config.json
```

The example config
[`examples/config/phase_3_2/phase_3_2_cnn_1h_config.json`](examples/config/phase_3_2/phase_3_2_cnn_1h_config.json)
reads training/validation/test CSVs from
[`examples/data/phase_3/`](examples/data/phase_3) and writes models, metrics
and plots under `examples/results/`. Full training is long-running and was not
executed for this README; `--help` and plugin imports were verified as listed
above.

Without the predictor package installed alongside, the pipeline stops before
training (verified): after loading the encoder and decoder plugins it prints
`Plugin stl_preprocessor not found in group preprocessor.plugins` and exits 1. Additional configs cover `phase_3_2_daily`, `phase_4_1` and `phase_4_2`
under [`examples/config/`](examples/config), with matching driver scripts in
[`examples/scripts/`](examples/scripts).

Key flags: `--encoder_plugin`/`--decoder_plugin` select plugins;
`--save_encoder`/`--save_decoder` and `--load_encoder`/`--load_decoder` manage
model artifacts; `--window_size`, `--epochs`, `--batch_size`,
`--incremental_search` and `--threshold_error` control training;
`--use_normalization_json` points at the preprocessor's persisted
normalization parameters so encoders are trained in the same value space as
downstream consumers.

This is a standalone training tool; it has no distributed/DOIN runtime role.
Its `.keras`/`.h5` artifacts are plain files consumed by other repositories.

## Tests

```bash
python -m pytest tests -q --continue-on-collection-errors
```

Observed result (Python 3.12.13, TensorFlow 2.21.0): `10 failed, 2 passed,
6 errors` — part of the suite under [`tests/`](tests) fails to collect, and most
of what does collect is written against an older plugin API. Treat the test
suite as needing repair; the import and `--help` checks above are the current
smoke validation.

## Artifacts and reproducibility

- Trained encoder/decoder models are written to the paths given by
  `--save_encoder`/`--save_decoder` (committed examples live under
  `examples/results/phase_*/`, e.g.
  [`examples/results/phase_4_2/phase_4_2_cnn_small_encoder_model.keras`](examples/results/phase_4_2/phase_4_2_cnn_small_encoder_model.keras)).
- Reconstruction metrics, prediction CSVs and diagnostic plots are written
  next to the models as configured.
- The effective configuration can be persisted with `--save_config` and
  replayed with `--load_config` for reproducible runs.
- Downstream, predictor phase 3/4 configs reference these encoders via their
  `feature_extractor_file` key — keep artifact paths stable once published.

## Safety and security

- No credentials are stored in this repository. Optional remote config/log
  endpoints take `--username`/`--password` as CLI arguments — do not embed
  secrets in committed config files.
- Example datasets are historical market data for research; nothing here is
  financial advice.

## Limitations

- The `rnn` and `cnn_signed` encoder/decoder entry points in
  [`setup.py`](setup.py) point at modules that do not exist
  (`app/plugins/encoder_plugin_rnn.py`, `app/plugins/encoder_plugin_cnn_signed.py`
  and their decoder counterparts); selecting them fails with
  `ModuleNotFoundError` (verified). The `*_working.py` variants in
  [`app/plugins/`](app/plugins) are unregistered spares.
- Running the pipeline requires an external package that provides the
  `preprocessor.plugins` entry-point group (predictor); the group name is also
  claimed by other repositories in the stack, so co-installations can shadow
  each other.
- The test suite has collection errors (see Tests).
- `requirements.txt` pins `tensorflow-gpu`, which is a legacy package name on
  modern TensorFlow installs.

## Related repositories

- [preprocessor](https://github.com/harveybc/preprocessor) — produces the
  normalized datasets and normalization JSON consumed here.
- [feature-eng](https://github.com/harveybc/feature-eng) — engineered
  features/labels upstream of normalization.
- [predictor](https://github.com/harveybc/predictor) — loads the saved
  encoders (`feature_extractor_file`) for phase 3/4 model training and
  provides the runtime preprocessing plugin group.
- [prediction_provider](https://github.com/harveybc/prediction_provider) —
  serving layer for trained models.

## License

[MIT](LICENSE.txt).
