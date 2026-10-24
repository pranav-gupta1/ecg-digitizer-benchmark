# Audit: Open-ECG-Digitizer (V1) vs StenhedeV2 (V2)

Status: code audit complete; numbers marked TODO are filled from benchmark runs.

## Provenance

- V1: https://github.com/Ahus-AIM/Open-ECG-Digitizer, pinned at `97a1508` (v1.9.3, 2026-06-04,
  CC BY-SA 4.0). The 18h32n mirror the mentor linked is this repository at commit `cca543c`
  (v1.9.2 + preprint note); the commits after it change only type hints (`dewarper.py`,
  `utils.py`), docs, CI and the LICENSE file. Model code is byte-identical between `cca543c`
  and `97a1508`.
- V2: https://github.com/justinadjasu/I-Love-Paper `src/StenhendeV2`, pinned at `fb99ccc`
  (2026-06-04). Its model files match V1 at `cca543c`, so V2 was forked from that state
  (equivalently from the mirror).
- Weights are identical (same SHA-256) and are downloaded from the Ahus-AIM LFS store.

## File-by-file diff (V1 `src/` vs V2 `digitizer/`)

| file | status |
|---|---|
| `model/unet.py`, `model/signal_extractor.py`, `model/lead_identifier.py`, `model/dewarper.py` | byte-identical (checked by SHA-256 in `tests/equivalence/test_modules.py`) |
| `model/inference_wrapper.py` | import path only (`from ..utils`) |
| `pipeline.py` (V2) vs `digitize.py` (V1) | identical function bodies; V2 drops the CLI `__main__` block (moved to `scripts/stenhede_digitize.py`) |
| `utils.py` | V2 removes training-only helpers (Ray `EarlyStopper`, data loaders, LR scheduler, plotting); `find_config_path` looks in the bundled `configs/` instead of `src/config/`; `import_class_from_path` unchanged |
| `config_loader.py` (V2) vs `config/default.py` | adds `${VAR}` expansion and sibling-YAML resolution; no config values change |
| `configs/*.yml` | identical values; only class-path prefixes and `${STENHEDE_WEIGHTS_PATH}` differ. V2 ships only the george-moody-2024 config and layouts; it does not ship `inference_wrapper_ahus_testset.yml` or `lead_layouts_reduced.yml` (the benchmark supplies V1's copies to both) |
| `model/cropper.py`, `model/pixel_size_finder.py`, `model/perspective_detector.py` | the speed changes, below |
| `overlay/` | new in V2 (no counterpart in V1) |

Conclusion on "skipped steps": V2 runs the same stages with the same parameters (resample 3000,
rotate on resample, dewarping disabled by config in both, same layout identifier settings).
`tests/equivalence/test_end_to_end.py::test_v2_runs_every_stage` and
`test_both_pipelines_get_identical_config` assert this.

## The three speed changes

### 1. `cropper.py`: Hough-bin accumulation vectorized

V1 loops in Python over the `granularity` (80) candidate lines per direction, doing
`buffer[y, x] += 1` per line, then builds a one-hot tensor of shape (H, W, 80) to sum
probabilities per bin. V2 computes all line pixels in one broadcasted expression,
accumulates with a single `scatter_add_`, and bins with `scatter_add_` instead of the one-hot
(which allocates H*W*80 ints, ~1.2 GB at 3000x2100).

Equivalence:
- Buffers are integer and identical by construction (each line adds exactly one count per row
  or column; within a line indices are unique, so V1's `+=` and V2's `scatter_add_` agree).
- Bin sums: a float reduction in a different order; differences at 1e-7.
- `dtype` int -> long: no effect.
- The new `clamp(max=granularity-1)`: after `cumsum`, the last column/row of each line reaches
  the value `granularity`. V1's `_fix_edge_effects` overwrites the last row and column, so this
  normally never reaches `one_hot`, which would raise. The only case where the clamp changes
  behaviour is a line that lands exactly on column W-2 (or row H-2) with every other line to
  its left: V1 raises `RuntimeError` (image fails), V2 folds the count into the last bin. The
  test asserts that no bin equals `granularity` on real images, so on those images the clamp
  is a no-op and V2 is strictly more robust.
- Destination points: created on the input device; same values.

### 2. `pixel_size_finder.py`: FFT autocorrelation and batched grid search

V1: `np.correlate(x, x, mode="full")` (O(n^2) on CPU, with a device->host copy), then a Python
loop over 1000 candidate spacings, each building a comb vector and taking a dot product.
V2: `irfft(|rfft(x)|^2)` on device, and one (samples x lags) matrix built with advanced
indexing, multiplied by the weighted autocorrelation in a single matmul, then `argmax`.

Equivalence:
- Autocorrelation: identical up to float32 roundoff (relative error < 1e-4 in the test).
- Comb construction: `arange(0, n, step)` vs `offsets * step` produce the same rounded indices;
  the major-line pattern `indices[::5]` is reproduced with `valid[:, ::5]`.
- Tie-breaking: V1 keeps the first maximum (`<` comparison); `argmax` also returns the first.
  Two candidates would have to tie to within float roundoff for the result to differ.
- Return type: float instead of 0-dim tensor; downstream only does arithmetic.

### 3. `perspective_detector.py`: tensors created on device

`randperm`, `meshgrid`, `arange` and the `inf` scalar are created on the input's device, which
removes host->device copies and host-side index generation.

Equivalence:
- On CPU: identical output for the same seed (`test_perspective_detector_identical_on_cpu_with_seed`).
- On CUDA/MPS: `torch.randperm(n, device=gpu)` draws from the device generator, so the random
  subsample of non-zero pixels (when more than `max_num_nonzero`) differs from V1's CPU draw.
  V1 itself does not seed this draw, so V1 is not deterministic across runs either; V2 moves the
  nondeterminism to a different generator. `test_accelerator_difference_is_only_randperm` forces
  V2's `randperm` onto the CPU generator and checks that outputs then agree to float precision.
  The size of the effect on real images is measured end-to-end (TODO: fill from the run).

## Overlay (new in V2)

`overlay/overlay.py` draws the extracted traces back on the original photo. The traces are
emitted in the rectified, rotated, y-cropped, x-trimmed coordinate space, so drawing them on
the input requires inverting: resample-to-3000 scaling, the perspective warp (via the cropper's
destination points and `cv2.getPerspectiveTransform`), the conditional 90-degree rotation, the
y-crop offset and the dropped empty columns. The last three are hidden state inside
`InferenceWrapper._crop_y`, `_rotate_on_resample` and `SignalExtractor.preprocess_lines`, which
the overlay captures by monkey-patching those methods for the duration of a call. Per-lead
colouring assumes the 3x4+1R layout (exactly 4 extracted lines, split by x-quartile); other
layouts are drawn in a single colour. Nothing comparable exists in V1 (its only "overlay" code is
training-time text-image augmentation in `src/transform/vision.py`, not included here).

## Measured results

TODO after the 300-image run and the mentor's full run:
- per-stage timing and speedup attribution (table_stage_attribution.csv)
- end-to-end waveform agreement on CPU (expected: identical) and on GPU (randperm effect)
- quality metrics V1 vs V2 by source and category
- further optimisation candidates from torch.profiler
