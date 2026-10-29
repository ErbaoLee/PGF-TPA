# PGF-TPA

Physics-Gated Fusion with Temporal-Pooled Aggregation for sky-image-assisted photovoltaic forecasting at **5, 15 and 30 minutes**. Each sample contains 15 consecutive one-minute RGB frames (64×64) and 15 historical power values. Predictions are in kW.

The release contains training code, real model checkpoints, per-sample predictions, validation predictions, processed power/time metadata, window indices and checksums. Original images are rebuilt from a pinned upstream dataset revision. Paper artwork, drafts, remote-access scripts and duplicate caches are excluded.

## Setup

Use Python 3.12. GPU training is recommended; saved-result recomputation only needs NumPy.

```bash
git lfs install
git clone https://github.com/ErbaoLee/PGF-TPA.git
cd PGF-TPA
git lfs pull
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a CUDA build, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/). Prefer Git cloning over a source ZIP, which can contain LFS pointers instead of weights.

## Recompute the released results

No raw images or PyTorch are needed for this path:

```bash
python -m pip install -r requirements-metrics.txt
python scripts/verify_assets.py
python scripts/reproduce.py
```

Outputs are written to `outputs/reproduction/`. The script verifies sample alignment and reference metrics and recomputes sunny/cloudy/overall and ramp-subset RMSE, MAE, R² and forecast skill.

| Model (15 min) | Overall RMSE (kW) | Ramp RMSE (kW) |
|---|---:|---:|
| TPA-E3 | 2.8240 | 6.1168 |
| TPA-RW-E3 | 2.8799 | 5.8343 |
| TPA-E18 | 2.8125 | 6.2004 |
| ConvLSTM-E3 | 2.9373 | 6.2745 |

E3 averages seeds 0/1/2. E18 combines three fixed-split members and 15 fold members. A ramp is `abs(P(t+15)-P(t)) > 3 kW`. Ramp weighting improves event error at the cost of overall error. Ordinary TPA-E3 achieves overall RMSE 2.2843 / 2.8240 / 3.1666 kW at 5/15/30 minutes. See [full tables](results/TABLE3.md).

## Rebuild the data

The source is [SolarBench SKIPPD](https://huggingface.co/datasets/solarbench/SKIPPD), pinned to revision `f51690de73d34cc409670ef6119fd69cc3aade8d`. Allow at least 12GB free disk space.

```bash
python scripts/download_data.py
python src/prepare_data.py
python src/prep_clearsky.py
python scripts/verify_assets.py --skip-assets --arrays --images
```

An existing Hugging Face `save_to_disk` Arrow directory can be used with `python src/prepare_data.py --source /path/to/My_SKIPPD_Data`. The pipeline preserves US/Pacific local wall-clock timestamps, original kW units and strict 60-second continuity. It generates indices for all three horizons. See [data details](data/README.md).

## Inference and training

```bash
python src/infer.py --family tpa-rw --horizon 15 --check-stored
python src/infer.py --family tpa --horizon 5 --check-stored
python src/infer.py --family tpa-rw --seeds 0 --limit 32 --check-stored
python scripts/train.py --family tpa-rw --horizons 15 --seeds 0 1 2
python scripts/train.py --family tpa --horizons 5 15 30 --seeds 0 1 2
```

Inference goes to `outputs/inference/`; new training runs go to `outputs/retrained/`. A limited smoke run is not a full test result. Retraining is not bitwise deterministic across hardware; the released checkpoints and predictions provide the direct replication path. Checkpoint inventory: [index](checkpoints/README.md). Exact protocols and implementation qualifications: [reproducibility](docs/REPRODUCIBILITY.md), [protocol](docs/PROTOCOL.md).

## SKIPP'D forecast-driven dispatch and frequency case

The new event case uses the released 15-minute predictions for 2017-09-20. It reproduces 28 numerical scenarios: three main cases, intervention ablations, forecast baselines, a diagnostic true-value input and the archived filter sensitivity. Only NumPy is required.

```bash
python scripts/run_skippd_case.py --check
python scripts/run_skippd_case.py --suite main --check
```

Outputs in `outputs/skippd_case/` include metrics, minute-resolution dispatch and issue/target-aligned alarm chains. See the [experiment guide](experiments/skippd_case/README.md) for parameters and exact scope. The archived event-timed preramp policy is not a fully online dispatch strategy; the archived smoothing coefficient is retained explicitly.

## Scope and license

This is a reconstructed-window benchmark (11,180 test windows at 15 minutes), not the untouched official HDF5 stack protocol. Cross-paper SOTA claims are not justified by mismatched protocols. Official test days comprise ten sunny and ten cloudy days. Rule-labeled rainy validation-day analyses are exploratory because validation was used for early stopping. Historical sample-level significance tests do not account for within-day dependence.

Project code and trained weights retain the repository's existing [GPL-3.0 license](LICENSE). Upstream data and attributed third-party material retain their own terms: [notices](THIRD_PARTY_NOTICES.md).
