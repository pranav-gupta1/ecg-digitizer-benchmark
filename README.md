# ECG digitizer benchmark: Stenhede Open-ECG-Digitizer (V1) vs StenhedeV2

Runs both digitizers on the same images from ECG-Image-Database, saves every digitized
waveform, computes the npj-paper quality metrics (SNR, RMSE, MSE, Pearson, NaN fraction) per
lead, times every pipeline stage with device synchronization, and writes tables and figures.
Model code for both versions is vendored and pinned under `third_party/` (see the `PIN.txt`
files); nothing in it is modified.

## Setup (copy-paste)

```bash
git clone <this repo> ecg-digitizer-benchmark && cd ecg-digitizer-benchmark
conda env create -f environment.yml          # python 3.12 + torch (CUDA on Linux, MPS on Mac)
conda activate ecg-bench
bash scripts/download_weights.sh             # two checkpoints, 114 MB, shared by V1 and V2
```

Dataset: Kaggle `physionet/ecg-image-database` (36 GB, CC BY-ND 4.0). With a Kaggle token in
`~/.kaggle/access_token` (Kaggle > Settings > API > Create New Token):

```bash
bash scripts/download_dataset.sh ~/Documents/ecg-image-database
```

The script expects `<data_dir>/D0`, `D1`, `D2` (PTB-XL, Emory, Ahus) as in the Kaggle release.

## Run

```bash
# smoke test: 10 images, auto device (cuda > mps > cpu)
python bench/run_benchmark.py --data-dir ~/Documents/ecg-image-database --n-images 10 --output-dir outputs/smoke

# 300 images, stratified across the three sources
python bench/run_benchmark.py --data-dir ~/Documents/ecg-image-database --n-images 300 --output-dir outputs/n300

# full dataset on a GPU cluster
python bench/run_benchmark.py --data-dir /cluster/ecg-image-database --device cuda --output-dir outputs/full
```

Flags (all also settable in `configs/benchmark.yaml`): `--data-dir`, `--output-dir`,
`--n-images N` or `--fraction 0.1`, `--subset ahus|emory|ptbxl|all`, `--device auto|cpu|cuda|mps`,
`--pipelines v1,v2`, `--seed`, `--save-overlays N`, `--no-resume`.

Runs are resumable: re-running with the same `--output-dir` skips images already done, so an
interrupted cluster or Colab job continues where it stopped.

Pipeline names for `--pipelines` (comma-separated):

| name | what it is |
|---|---|
| `v1` | Open-ECG-Digitizer as published |
| `v2` | StenhedeV2 (the speed-optimised fork) |
| `v1@1`, `v2@1` | the same with torch seed 1, a seed control for the algorithm's own randomness |
| `v2+cache` | V2 releasing the accelerator cache after every stage; identical outputs, avoids swap on 16 GB Macs |
| `v2+orient` | experimental: V2 choosing the post-warp rotation by layout-matching cost (fixes portrait photos) |

`--image-list FILE` runs an explicit list of image ids (one per line) instead of a sample.

Outputs in `--output-dir`:

| file | content |
|---|---|
| `waveforms/<pipeline>/<image_id>.csv` | digitized 12-lead waveform, µV, 1000 Hz (NaN = unreadable) |
| `metrics_per_lead.csv` | SNR, Pearson, RMSE, MSE, shift, NaN fraction per image, pipeline and lead (`II_rhythm` = full strip) |
| `timing.csv` | wall time and per-stage seconds per image and pipeline, decode time, peak RSS |
| `table_*.csv` | quality by source / category / lead, speed, stage attribution, V1 vs V2 agreement |
| `summary.md`, `summary.json` | the report, including the comparison with the npj paper's numbers |
| `figures/` | SNR by category, V1 vs V2 scatter, speed by stage, shift histogram, Ahus SNR distributions, lead II percentile plots, overlays |
| `env_info.json` | hardware, OS, torch, device, pinned commits, FLOPs, memory, GPU utilization |

Other tools:

```bash
python bench/dataset_stats.py --data-dir ~/Documents/ecg-image-database   # dataset characteristics table
python -m pytest tests/equivalence -q                                      # V1 vs V2 module and end-to-end equivalence
```

## Layout

```
bench/            run_benchmark.py (entry point), pipelines.py (uniform V1/V2 wrapper, timers),
                  evaluate.py (paper metrics), dataset.py (index + labels), profiling.py,
                  report.py, overlay.py, dataset_stats.py
configs/          benchmark.yaml
third_party/      open_ecg_digitizer/ (V1, Ahus-AIM, pinned) and StenhendeV2/ (V2, pinned)
tests/equivalence module-level and end-to-end V1 vs V2 tests
docs/             audit.md (what changed between V1 and V2 and what it costs)
scripts/          download_weights.sh, download_dataset.sh
```

## Notes

- Both pipelines are built from V1's YAML configs, so every stage gets identical parameters;
  only class paths differ. The Ahus two-page Cabrera images use V1's `inference_wrapper_ahus_testset.yml`
  (reduced layout set), PTB-XL and Emory use the george-moody-2024 config, as in the paper.
- Timing calls `torch.cuda.synchronize()` / `torch.mps.synchronize()` around every stage, and the
  first images of each run are untimed warm-up passes.
- Memory: on a 16 GB Mac run one benchmark process at a time.
