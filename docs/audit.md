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

### Equivalence tests (tests/equivalence, 13 passing)

- Configs: identical for V1 and V2 in both layout sets, apart from class paths.
- All stages run in both; same resample size (3000), same dewarping setting (off).
- Unchanged modules: byte-identical (unet, signal_extractor, lead_identifier); dewarper differs
  only in the two type-hint lines from upstream commit 69d68c1.
- Cropper: integer buffers identical, bin values within 1e-6, source points within 1e-4. No bin
  reached `granularity` on real images, so the new clamp is a no-op there.
- Pixel size finder: FFT autocorrelation within 1e-4 relative error; same pixel spacing.
- Perspective detector: identical on CPU with a fixed seed.
- End to end on CPU: waveforms identical (same NaN mask, max difference below 1e-3 uV).
- End to end on MPS: max 68 uV difference with each version's own RNG; 2.1 uV once V2's
  on-device `randperm` is forced onto the CPU generator. The remainder is float32 roundoff from
  the on-device FFT and `scatter_add`. Signal amplitudes are ~1000 uV.

### The speedup is device-specific (Apple M5, 16 GB, torch 2.14.1)

Single images, one process, after a warm-up pass (scratch probe, 2026-10-24):

| image | device | V1 wall | V2 wall | V1 cropping | V2 cropping |
|---|---|---|---|---|---|
| D1 color scan, 2130x1652 | MPS | 18.9 s | 3.0 s | 5.7 s | <0.1 s |
| D0 phone photo, 4032x3024 | MPS | 83.2 s | 68.8 s | 14.8 s | <0.1 s |
| D1 color scan | CPU | 47.9 s | 50.5 s | 0.5 s | <0.1 s |
| D0 phone photo | CPU | 250.4 s | 233.2 s | 0.8 s | <0.1 s |

On CPU, V1's cropping already takes under a second, and the two versions run at the same speed
within noise. On the GPU, V1's cropper builds its bin buffer on the CPU and moves an
(H, W, granularity) one-hot tensor (about 4.3 GB at 3000x2250) to the device; V2 keeps
everything on the device and never materialises the one-hot. So the published speedup is a
GPU-path effect, and the GPU is the setting that matters at scale. The mentor's CUDA run will
give the definitive numbers.

### Memory on a 16 GB Mac

At the 3000 px working resolution the pipeline needs more than 10.5 GB of accelerator memory
for a phone photo: capping MPS at 0.9x the recommended working set runs out of memory. Without
the cap it reaches 16 to 20 GB and the machine swaps, which inflates every stage, including
stages whose code is identical in V1 and V2 (segmentation 0.9 s on a scan vs 25 to 30 s on a
photo; layout identification 1 s to 60 s). V1's extra cropper memory pushes the later stages
further into swap (layout identification averaged 10 to 13 s after V1's cropper vs 0.8 to 3 s
after V2's on the same scans). Timings on photos from this machine therefore measure memory
pressure as much as compute; scans fit in memory and give clean timings.

### Failure mode found with the overlay: portrait monitor photos

Monitor photos (variant 07) are portrait images (3000x4000) of a landscape ECG.
`InferenceWrapper._rotate_on_resample` rotates the warped tensor by 90 degrees whenever its
canvas is taller than wide. The perspective warp keeps the input canvas, so a portrait photo is
always rotated and the landscape ECG ends up sideways. The overlay shows the result: on
D0_000324_07 only the last ~1 cm of each row was extracted (the drawn stubs land exactly on the
printed traces, so the overlay geometry itself is right), and on D0_007904_07 nothing usable was
extracted (SNR about 0 dB). The logic is identical in V1 and V2, so it is a limitation of the
published algorithm and plausibly explains the very low SNR the paper reports for photos.
Candidate fix, to be evaluated separately from the speed claims: decide rotation from the
shape of the detected paper quadrilateral (source points) instead of the canvas.

### Still to fill

- 200/300-image run: per-stage attribution, V1 vs V2 waveform agreement, quality by category
- torch.profiler pass on V2 for further optimisation candidates
- CUDA numbers from the mentor's cluster run
