# TSDC: TDCP-Supervised Doppler Correction

Reference implementation for

> H. Ham and J. So, *Training-Only TDCP Supervision for ADR-Free Smartphone Doppler Correction:
> Controlled Comparison with Adjacent-Epoch Averaging*, manuscript under review.

## Method

For every adjacent-epoch pair of the same satellite signal, a small network predicts a
correction μ to the Android pseudorange rate (PRR) and a Laplace scale b. Training uses
the TDCP-referenced target

    y = (ADR_k - ADR_{k-1}) / Δt - PRR_k

built from screened accumulated delta range (ADR). ADR is used only for this target:
inference reads neither ADR nor carrier phase. The corrected range change
(PRR_k + μ) Δt enters a horizontal-position-and-clock factor graph around the nominal
(WLS) trajectory with factor scale σ = max(1 m, 5 b Δt), solved by Huber IRLS.

## Scope

This repository covers the GSDC experiments of the paper:

- GSDC 2021 conversion to the common `device_gnss.csv` format
- adjacent-epoch pair construction, TDCP-referenced targets and the 68 input features
- the 68→192→192→96→2 network, Laplace NLL, training and checkpoint selection (seeds 42–46)
- inference with the output gates |μ| ≤ 3 m/s and b ≤ 1 m/s, run on data with ADR columns removed
- endpoint, averaged and TSDC temporal measurements; screened TDCP-FGO; the factor graph
- the score (P50 + P95)/2, aggregation by physical phone session and leakage group, bootstrap and Wilcoxon statistics

The code reproduces Tables 2–4 and the GSDC scores of Sections 5.1–5.4. It does not include:

- the μ − μ̄ control and the paired intervals of Section 5.4
- the device-level analyses of Section 5.3 and Online Resource 1, Section S6.4
- the source-validation searches (the selected settings are fixed in the code) and the construction of the split
- the external datasets (WHU, Hervanta, TU Wien), the post-hoc analyses of Section 6 and the other Online Resource experiments

| Paper | Code |
|---|---|
| training pairs, residual target y, residual gate | `tsdc/data.py` (`extract_training_pairs`, `build_pair_features`) |
| 68 features, source-training standardization | `tsdc/features.py` |
| network, log b clamp [−5, 2], Laplace NLL | `tsdc/model.py` |
| AdamW, batch 8192, ≤ 60 epochs, patience 10, source-validation MAE selection | `tsdc/train.py` |
| output gates, ADR-free inference | `tsdc/inference.py` |
| endpoint / average / TSDC measurements, σ rule, screened TDCP, shared ADR-valid intervals | `tsdc/factors.py` |
| factor coefficients, priors, clock constraints, Huber IRLS, block-tridiagonal solver | `tsdc/fgo.py` |
| score, sessions, leakage groups, bootstrap, Wilcoxon | `tsdc/metrics.py` |

## Environment

Python 3.12 with the versions in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The paper checkpoints were trained on one NVIDIA RTX 3090 with the CUDA 12.1 build of
PyTorch 2.5.1. With that build and GPU, `tsdc/train.py` reproduces all five checkpoints
bit for bit. The default PyPI wheel is the CUDA 12.4 build; the CUDA 12.1 build is
installed with

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

On other GPUs or library versions, expect small differences in the trained weights.
Run all commands from the repository root.

## Dataset preparation

Download the three GSDC training releases from Kaggle
([2021](https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge),
[2022](https://www.kaggle.com/competitions/smartphone-decimeter-2022),
[2023](https://www.kaggle.com/competitions/smartphone-decimeter-2023))
and extract each archive under `data/gsdc/<year>/`:

```
data/gsdc/
├── 2021/baseline_locations_train.csv
├── 2021/train/<collection>/<phone>/{<phone>_derived.csv, <phone>_GnssLog.txt, ground_truth.csv}
├── 2022/train/<drive>/<phone>/{device_gnss.csv, ground_truth.csv}
└── 2023/train/<drive>/<phone>/{device_gnss.csv, ground_truth.csv}
```

The 2021 release has no `device_gnss.csv`. Create it from the derived file, the raw
log and Google's baseline positions (the nominal trajectory for 2021):

```bash
python scripts/convert_gsdc2021.py --data-root data/gsdc
```

### Fixed split

The GSDC results in the paper use the archived split in `splits/`. The split is not regenerated here.

- `gsdc_5splits.json`: route assignment (source train 32, source validation 10, target adaptation 8, target evaluation 91 routes)
- `route_group_mapping.csv`: physical collection and leakage group for each of the 141 routes; used for session and group statistics
- `source_trace_manifest.csv`: the 77 training and 20 validation traces, one per physical phone session

A physical phone session is one phone model on one physical drive, with repeated GSDC releases of
the drive merged. A leakage group joins routes that overlap; groups are the units of the paired statistics.

## Training

```bash
python scripts/prepare_pairs.py --data-root data/gsdc   # work/pairs, about 3 min
python scripts/train_5seeds.py                          # work/checkpoints/tsdc_seed{42..46}.pth, about 4 min on one RTX 3090
```

`prepare_pairs.py` stores the 77 base columns of the original pipeline. Training uses
the 68 TSDC inputs, i.e. the base columns without the nine WLS-motion features.
One run of `python -m tsdc.train --seed 42` trains a single checkpoint.

## Evaluation

```bash
python scripts/evaluate_gsdc.py --data-root data/gsdc --workers 12
```

This evaluates the 216 target-evaluation traces for WLS, Endpoint and Averaged Doppler
FGO, screened TDCP-FGO and, for each checkpoint, TSDC-FGO and its controls. It writes
`work/eval/trace_detail.csv` and prints the tables; seed 42 is the representative run.
In the output, "TSDC mask" marks controls run on the same satellite–epoch intervals as
TSDC (Sections 5.2–5.4), and "common ADR-valid mask" marks the shared ADR-valid intervals
of Table 2. The run takes 1.5 to 2 hours with 12 workers on a 16-core CPU.
`--summary-only` reprints the tables from an existing `trace_detail.csv`, and
`--checkpoint-dir` and `--checkpoint-pattern` (default `tsdc_seed{seed}.pth`) select other checkpoints.

Inference runs on the CPU by default (`--cuda` for GPU). Multithreaded CPU and GPU
inference can change μ and b in the last float32 bit; in our checks this changed individual
trace scores by less than 1e-7 m. The 2021 files written by `convert_gsdc2021.py` differ
from those used for the paper by at most 4e-9 m in the nominal positions, which changes
2021 trace scores by less than 1e-6 m. The tables are unchanged at the reported precision.

## Expected main results

Target evaluation, 164 physical phone sessions (Table 3; ± is the standard deviation over five runs):

| Method | Mean P50 (m) | Mean P95 (m) | Score (m) |
|---|---|---|---|
| WLS | 2.5047 | 6.0464 | 4.2756 |
| Endpoint Doppler FGO | 2.3130 | 5.3029 | 3.8080 |
| Averaged Doppler FGO | 2.2777 | 5.2503 | 3.7640 |
| Screened ADR-derived TDCP-FGO | 2.3251 | 5.4288 | 3.8770 |
| TSDC-FGO | 2.2519 ± 0.0028 | 5.1452 ± 0.0072 | 3.6985 ± 0.0048 |

Paired TSDC-FGO differences over 58 leakage groups (Table 4; 20,000 bootstrap resamples):

| Comparator | Better sessions | Mean group Δ (m) | 95% CI (m) | Wilcoxon p |
|---|---|---|---|---|
| WLS | 154/164 | −0.6099 | [−0.7303, −0.4973] | 3.2e−10 |
| Averaged Doppler FGO | 96/164 | −0.0577 | [−0.0947, −0.0265] | 0.0020 |
| Screened ADR-derived TDCP-FGO | 125/164 | −0.2113 | [−0.3283, −0.1005] | 6.7e−6 |

The printed table also includes endpoint Doppler on the same intervals as TSDC (Section 5.3):
159/164 sessions, −0.8805 m, 95% CI [−1.0024, −0.7605].

Seed-42 controls: on the shared ADR-valid intervals (Table 2), endpoint / screened TDCP / TSDC
score 4.2696 / 3.8583 / 3.8674 m; on the intervals used by TSDC (Section 5.2), endpoint /
averaged / TSDC score 4.5171 / 3.7640 / 3.7028 m.

## Tests

Regression tests on synthetic inputs (requires `pytest`):

```bash
python -m pytest tests
```

## Data availability

No data are redistributed here. GSDC 2021–2023 are available from the Kaggle pages above.
The trained checkpoints used in the paper are available from the corresponding author on
reasonable request.

## Citation

Citation details will be added once the paper is published.

## License

MIT. See `LICENSE`.
