# PS3-R shortlist: representation families beyond the AE control, with official references pinned

Lane C of `SATOSHI_PROGRESSIVE_SELECTION_AND_MODULAR_CONTINUATION_2026_09_30` (predictor master `ac125db9`), subplan
`FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md` section 6 and FS17. Written 2026-10-01.

**Status: SHORTLIST. NO_NEW_MEASUREMENT. No winner is picked.** Nothing was trained, downloaded as weights, or run.
Every reference below is pinned by the commit `git ls-remote` returned on 2026-10-01; every comparator number names
its paper table and dataset and is labelled **RE-DERIVED** (read by this lane from the paper's PDF text in this
session, `pdftotext` of the arXiv PDF) or **TRANSCRIBED** (copied from an earlier reading, not re-derived here).
A transcribed number is admissible only with that label. No number here is a measurement of ours.

Consumers read the card contract `docs/contracts/representation_candidate_card.v1.schema.json` on predictor branch
`satoshi/c-contracts-20261001`; this document is the human-readable shortlist behind the first cards.

## 0. The two axes and the contract every family must meet

Encoder architecture (Conv1D/TCN, recurrent, Transformer) and training objective (reconstruction/denoising,
contrastive, masked, latent prediction, supervised) are separate axes. Every candidate is evaluated under one
contract:

| rule | source | consequence for a card |
|---|---|---|
| latent keeps `(batch, time, channels)` on the common 24-step hourly grid of the default recipe; a patch or pooled output needs a **declared and validated adapter** or is compared in its native recipe outside the assembly | subplan 6.1; orders section 2 (branch `(B,24,1) -> (B,24,16)`) | `latent.layout` and `temporal_contract_state`; `pooled` can never be `SATISFIED` |
| **TRAIN-only**: weights fitted on our TRAIN folds only; weights pretrained on a foreign corpus are `FOREIGN_PRETRAINED`, never labelled TRAIN-only, and need a contamination audit before any comparison | subplan 6.1, FS17 | `corpus.kind` with the if/then rule in the schema |
| no decoder is demanded of a family that has none: reconstruction is `NOT_APPLICABLE`, which is a state, not a failure | subplan 6.1, 6.3, FS17 | `evaluation.reconstruction.state` |
| the validation criterion without a decoder is the probe battery of subplan section 7 (below), on `Y_s`/`Y_l`/`Y_b`, never the self-forecast of the input | FS02, FS17 | `evaluation.probes[]` |
| intervening on the network (permuting/replacing latents) is model evidence, kept apart from causal evidence | subplan 5.5 | `model_intervention_evidence`, never in a causal dossier |
| the operational encoder receives only information available at `t`; the future target is allowed only in a declared `SYNTHETIC_OFFLINE` contract | subplan 6.2, FS03 | `conditioning_contract` |

### The decoder-free validation criterion (FS17)

Per feature/group, head, horizon, fold and seed, with the encoder frozen and a temporal probe of the same shape and
budget for every arm:

* `Delta_probe = Loss(Y, probe(Z_random, c)) - Loss(Y, probe(Z_trained, c))` — positive means training added usable
  information about the business target; `Z_random` is the same architecture with untrained weights;
* preservation: `Loss(Y, probe(X_raw, c))` against `Loss(Y, probe(Z, c))` — a degradation introduced by the
  extractor; the raw arm may have a different dimension, which is declared, not hidden;
* paired naive on identical rows per horizon (persistence for returns; base rate for `Y_b`), and differences kept at
  1e-5/1e-6 resolution with seed/week dispersion;
* latent diagnostics: effective dimension, sensitivity to `X`, stability across seeds (collapse detection, not a
  universal utility threshold).

A probe on the input's own future is `self-forecast` and is refused by FS02.

## 1. The shortlist

Pins (all `git ls-remote` on 2026-10-01; "main"/"master" means the branch head at that instant, no release tag exists):

| family | official reference | official code, pinned revision | licence at the pin |
|---|---|---|---|
| AE / denoising control | P. Vincent et al., "Stacked Denoising Autoencoders", JMLR 11, 2010 | this repository's `feature_extractor.encoders` plugins (`ann`, `cnn`, `lstm`, `transformer`, `vae`, `vae_small`, `rnn`) at `cf39e4b` | repository's own |
| contrastive, per-timestamp | Z. Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022, arXiv:2106.10466 | `https://github.com/zhihanyue/ts2vec` `main` @ `b0088e14a99706c05451316dc6db8d3da9351163` | MIT (read at the pin) |
| masked patch reconstruction | Y. Nie et al., "A Time Series is Worth 64 Words", ICLR 2023, arXiv:2211.14730 | `https://github.com/yuqinie98/PatchTST` `main` @ `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, folder `PatchTST_self_supervised` (`patchtst_pretrain.py --mask_ratio 0.4`, `patchtst_finetune.py`); a clone at this pin already exists under the operator's `sota_sources` | Apache-2.0 (read at the pin) |
| masked, alternative | SimMTM (NeurIPS 2023) | `https://github.com/thuml/SimMTM` `main` @ `169513bef74fb676e48d98a0e30f8823793f691c` | **LICENSE file absent at the pin (HTTP 404)** — licence UNVERIFIED |
| latent prediction (JEPA) | WDSLab, "CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization", preprint under review, arXiv:2606.07031 | `https://github.com/WDSLab/CF-JEPA` `main` @ `5d3d2fd1273c283fbfa03249c078619245e84033` | MIT (read at the pin) |
| contrastive, alternative | TS-TCC (IJCAI 2021) | `https://github.com/emadeldeen24/TS-TCC` `main` @ `15bdfd6eced6aeb6c638839828effe1b4b2ea391` | **LICENSE file absent at the pin (HTTP 404)** — licence UNVERIFIED |
| foundation, pretrained | M. Goswami et al., "MOMENT: A Family of Open Time-series Foundation Models", ICML 2024, arXiv:2402.03885 | `https://github.com/moment-timeseries-foundation-model/moment` `main` @ `38f7310ad594100747ca2a8357e9c7ca7d323e0e`; weights `AutonLab/MOMENT-1-{small,base,large}` on Hugging Face (revision NOT pinned here — no download was made) | MIT (read at the pin) |
| generative conditional (secondary) | K. Sohn et al., CVAE, NeurIPS 2015; A. Larsen et al., VAE-GAN, ICML 2016 | no official code exists for either; this repository's `vae` plugin is the local implementation; TimeGAN `https://github.com/jsyoon0823/TimeGAN` `master` @ `8f6181cb9b9d2fa0c930cd902411d9ac8a308e07` as the evaluation reference (TSTR/TRTR) | TimeGAN: not read |

## 2. Reproducible comparators (what number, which table, which dataset, how obtained)

| family | comparator | provenance | why it is or is not comparable to anything of ours |
|---|---|---|---|
| TS2Vec | **Table 7** (multivariate forecasting), Electricity, MSE/MAE: H=24 0.287/0.374; H=48 0.307/0.388; H=168 **0.332/0.407**; H=336 0.349/0.420; H=720 0.375/0.438. **Table 2/full univariate table**, Electricity H=168 0.427/0.394. Protocol (paper §4.2): encoder trained on the training split only, ridge regression on the per-timestamp representation, 60/20/20 chronological split, Electricity resampled hourly | **RE-DERIVED** from `arXiv:2106.10466` PDF text in this session | NOT_COMPARABLE with the ECL321/TSL protocol of `SOTA_REFERENCE_DOSSIER_2026_09_21` (different split and horizons); comparable only with a native-recipe reproduction of TS2Vec itself |
| PatchTST self-supervised | Table 4, ECL, L=512 pretrain+finetune: 96 0.126/0.221; 192 0.145/0.238; 336 0.164/0.256; 720 0.193/0.291 | **TRANSCRIBED** from `SOTA_REFERENCE_DOSSIER_2026_09_21.md` (RP91 read of arXiv:2211.14730 Table 4); not re-derived here — the self-supervised folder ships scripts, not result logs | comparable to the ECL dossier protocol (TSL `Dataset_Custom`); the predictor already has a Traffic/ECL replication path for the supervised recipe |
| CF-JEPA | **Table 5** (multivariate forecasting, 8 datasets, 3 seeds, 200 epochs): Avg MSE/MAE Ours 0.95/0.66, TS2Vec 0.94/0.67, CoST 0.75/0.58 (excluded from rank), T-Rep 0.90/0.65; Traffic: Ours 0.62/0.49, TS2Vec 0.78/0.57. **Table 6** (univariate): Avg Ours 0.36, TS2Vec 0.39 | **RE-DERIVED** from `arXiv:2606.07031` PDF text in this session | the paper's own Avg MSE does not favour CF-JEPA over TS2Vec (0.95 vs 0.94); its gains are on specific datasets (Traffic) and under its own unified protocol; preprint, under review |
| MOMENT | Table 6 (imputation, zero-shot vs linear probing), Electricity: MOMENT_0 0.250/0.371, MOMENT_LP 0.094/0.211, GPT4TS 0.072/0.183; Table 18 long-horizon forecasting exists per dataset (not extracted here) | **RE-DERIVED** (Table 6 row) from `arXiv:2402.03885` PDF text; Table 18 Electricity row NOT extracted | **the Time Series Pile contains Electricity (321 channels, hourly) and the ETT datasets (paper Tables 11 and 15)**: any benchmark number of MOMENT on those is in-corpus and the paper's "careful train-test splitting" is the only contamination control; nothing of ours is comparable until an audit of the Pile against EURUSD/FX is done |
| AE control | none published for our data | — | it is the control: raw, random-weights and trained arms are compared on our TRAIN folds only |
| CVAE / VAE-GAN | none published for our data; TimeGAN reports discriminative/predictive scores on its own datasets | — | secondary; evaluated by section 6.3 of the subplan (fidelity, TSTR/TRTR), never as the encoder's gate |

## 3. Admissibility per family for our data (hourly EURUSD features, 24-step branches, TRAIN-only)

| family | temporal contract | TRAIN-only | decoder-free criterion applies | verdict | reasons / blockers |
|---|---|---|---|---|---|
| AE / denoising control | **NOT_EVALUATED today, and undeclared**: the installed `cnn` encoder (read at `cf39e4b`, `configure_size`) emits a temporal output through two Conv1D layers with `strides=2`, i.e. a `(B, window_size/4, C)` grid (288 -> 72 by default) — not the 24-step common grid, no declared adapter, and no plugin declares a `latent_layout` (its docstring calls the output "a latent vector"); FS17 test `test_fs17_installed_cnn_encoder_declares_temporal_layout` is red for that reason | yes | reconstruction MEASURED, probes also required (FS06) | ADMISSIBLE as control **after** the temporal layout and grid are declared | the control must keep `(B,24,C)` or declare its adapter; the lane A engine (`da4ce7b4` branch spec) is the place, not a second encoder here |
| TS2Vec (contrastive) | per-timestamp output `(n, T, 320)`; a causal sliding mode exists (`encode(..., causal=True, sliding_length=1, sliding_padding=50)`), to be confirmed on integration | yes by construction (fit on our TRAIN) | yes: probes + latent diagnostics; reconstruction `NOT_APPLICABLE` | **ADMISSIBLE** | PyTorch dependency; dilated-CNN encoder; the hierarchical contrastive loss uses random cropping — crop windows must not cross the fold boundary |
| PatchTST self-supervised (masked) | patch grid (patch 16, stride 8 in the recipe) ≠ 24-step grid; needs a declared adapter (e.g. per-step projection of overlapping patches) validated by FS04, or native-recipe comparison | yes (pretrain on our TRAIN) | yes; the pretraining reconstruction head is diagnostic, not the gate | **ADMISSIBLE_WITH_ADAPTER** | adapter is its own experiment; RevIN/instance norm statistics must be fold-local |
| SimMTM (masked, alternative) | the CF-JEPA paper excludes it from per-timestep ranking because it does not produce per-timestep representations | yes | yes | **NOT_ADMISSIBLE until licence verified** | LICENSE 404 at the pin; pooled output |
| CF-JEPA (latent prediction) | per-timestamp (ranked with TS2Vec in its Table 5) | yes | yes; it has no decoder by design (`NOT_APPLICABLE`) | **ADMISSIBLE, EXPLORATORY** | preprint under review; EMA target encoder doubles encoder memory; its own table does not beat TS2Vec on average |
| TS-TCC (contrastive, alternative) | instance-level, classification-oriented | yes | yes | **NOT_ADMISSIBLE until licence verified**; low priority | LICENSE 404 at the pin; pooled by design |
| MOMENT (foundation) | patch-based with `embedding` mode (`reduction="none"` added per README) | **NO**: weights are `FOREIGN_PRETRAINED`; cannot be labelled TRAIN-only (FS17) | probes yes; reconstruction head exists | **NOT_ADMISSIBLE as TRAIN-only; ADMISSIBLE only as a labelled foreign-pretrained comparator after a corpus audit** | Pile includes FRED and finance-domain series: audit for EURUSD overlap required; large weights (small/base/large exist, sizes not verified here) |
| CVAE / VAE-GAN (generative) | this repository's `vae` plugin is conditional on `cvae_target_feature_names` — **an `OPERATIONAL` encoder may not be conditioned on future targets** (FS03); permitted only under `SYNTHETIC_OFFLINE` | yes | has decoder; generation evaluated separately (6.3) | **SECONDARY** | not a gate for selection; conditioning contract must be declared per run |

## 4. Order of piloting by expected utility to Y_s / Y_l / Y_b and cost — NOT a ranking of winners

No utility on our data has been measured for any family, so the order below ranks by (a) fit to the temporal
contract without an adapter, (b) TRAIN-only compliance, (c) reproducibility of the official reference and licence,
(d) expected cost per fit on the admitted GPUs. It says what to pilot first under the paired budget of subplan
10.2 (three routes: causal-first, representation-first, interleaved), not what will win.

| order | family | expected relevance to the heads (hypothesis, unmeasured) | cost class |
|---|---|---|---|
| 1 | AE/denoising control with temporal latent | baseline for every delta; required by FS05/FS06 | low |
| 2 | TS2Vec | per-timestamp, causal mode, cheap; Y_s (1–6 h) plausibly served by local context; Y_l unknown | low |
| 3 | PatchTST self-supervised + declared adapter | long-context masked pretraining plausibly serves Y_l (24–144 h); the adapter is the risk | medium |
| 4 | CF-JEPA | multi-horizon forward prediction is the closest objective to Y_s/Y_l by construction; exploratory | medium (EMA target) |
| 5 | MOMENT as foreign-pretrained comparator | only after the corpus audit; never TRAIN-only | high |
| 6 | CVAE/VAE-GAN | secondary; generation for PS7, not selection | medium |

Excluded for now with the reason recorded: SimMTM and TS-TCC (licence unverifiable at the pinned revision; pooled
outputs). They return to the list the day the licence is read.

## 5. What is NOT done

- No family was run, no weights downloaded, no dependency installed; TS2Vec/PatchTST/CF-JEPA/MOMENT are PyTorch
  while this repository is Keras/TensorFlow — the integration path (separate environment or port) is undecided.
- PatchTST's Table 4 numbers are TRANSCRIBED, not re-derived here.
- MOMENT's Table 18 Electricity forecasting row was not extracted; its corpus audit against FX data was not done.
- The licences of SimMTM and TS-TCC were not read (files absent at the pins).
- No installed encoder declares its latent layout or its grid; the `cnn` encoder emits a window_size/4 temporal grid without an adapter (FS17 red).
- The adapter from patch grid to the 24-step grid is specified as a requirement only.
