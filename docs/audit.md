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

### 200-image run (Apple M5 MPS, stratified sample, seed 0; outputs/mac_mps_n200)

127 PTB-XL, 52 Emory, 21 Ahus images; V1 and V2 interleaved per image, 0 failures.

Speed (mean seconds per image; photos are inflated by swapping on this 16 GB machine):

| | V1 | V2 | V1/V2 |
|---|---|---|---|
| mean wall time | 49.1 s | 36.3 s | 1.35x |
| median wall time | 27.3 s | 14.3 s | 1.91x |
| scans (clean, fits in memory) | ~21 s | ~5 to 9 s | 4.3 to 4.7x |
| phone/monitor photos | ~80 to 92 s | ~69 to 83 s | 1.1 to 1.2x |

Where the 12.7 s per image saved comes from: cropping 76% (9.83 s to 0.10 s), feature-map
resampling 12% (2.20 s to 0.63 s), perspective detection 7%, layout identification 6%, pixel
size search 1%. Segmentation and signal extraction are unchanged within noise. Achieved
throughput of the networks is the same in both (about 52 GFLOP/s on this machine); the gain is
entirely in the non-network stages, as expected from the code diff.

Quality (V2 minus V1, per image, mean SNR over the 12 leads):

- mean +0.013 dB, 95% CI [-0.015, +0.041] dB: no detectable difference
- median absolute difference 0.00008 dB; 90th percentile 0.10 dB
- 1 image worse by more than 0.5 dB in V2, 2 better
- per source and category, means agree within ~0.1 dB except the small monitor-photo groups
- waveforms: 148 of 200 images agree within 10 uV everywhere; the rest differ through the GPU
  random-subsample difference (see the seed control below)

Reproduction of the paper's numbers (V1, small n for Ahus):

| | ours | paper |
|---|---|---|
| Ahus scanner SNR | 20.4 dB (n=6) | 19.65 dB |
| Ahus OnePlus SNR | 12.5 dB (n=10) | 12.19 dB |
| Ahus iPhone SNR | 9.3 dB (n=5) | 10.47 dB |
| Emory color scan SNR | 7.6 dB (n=18) | 7.34 dB |
| Emory grayscale scan SNR | 8.7 dB (n=17) | 7.24 dB |

### Photo orientation failures (V1 and V2 alike)

Every portrait-orientation monitor or phone photo in the sample failed (12/12, mean SNR below
1 dB), while landscape photos of the same categories averaged 6 to 13.5 dB. Two portrait
degraded-paper photos, where the paper itself was photographed sideways, succeeded. Overlays
show two mechanisms: (1) portrait photos of a landscape ECG are rotated by 90 degrees because
the warped canvas is taller than wide, and (2) sideways paper photographed the other way round
ends up upside down, because `_rotate_on_resample` always turns clockwise and the shipped config
disables the flip check (`possibly_flipped: false`). Photos without these failures average 8.1 dB.

### Seed control: V2 differs from V1 no more than V1 differs from itself

V1 was run again on the same 200 images with torch seed 1 (`--pipelines v1@1`, same device).
The perspective detector's random pixel subsample is the only stochastic step.

| comparison | mean SNR diff (95% CI) | 90th pct abs SNR diff | images off by > 0.5 dB | 90th pct max waveform diff | images within 10 uV |
|---|---|---|---|---|---|
| V1 vs V1 (seed 1) | -0.0005 dB [-0.016, 0.015] | 0.074 dB | 3 | 346 uV | 152 |
| V1 vs V2 | +0.013 dB [-0.015, 0.041] | 0.098 dB | 3 | 450 uV | 148 |
| V1 (seed 1) vs V2 | +0.013 dB [-0.015, 0.042] | 0.113 dB | 2 | 375 uV | 154 |

The images that differ between V1 and V2 are largely the same images that differ between two V1
seeds (rank correlation 0.71 on the first 50). V2's differences from V1 are therefore within the
algorithm's own run-to-run randomness. The one systematic difference is float32 roundoff of about
2 uV (median max difference) from the on-device FFT and scatter_add, which is why only 54 images
match V1 within 1 uV against 151 for the V1 reseed. V1's timing also reproduced across the two
runs (49.1 s vs 49.3 s mean per image), so the speed measurements are stable.

Conclusion for the paper: on 200 validated images, V2 gives the same digitization quality as V1
(indistinguishable from re-running V1 with another seed) while removing the cropper bottleneck.

### Orientation selection (experimental `v2+orient`, outputs/mac_mps_orient)

All 16 portrait images from the 200-image sample plus 3 landscape photo controls. When the
upstream rule rotates the warped tensor, `v2+orient` also runs counter-clockwise and no
rotation, and keeps the candidate with the lowest layout-matching cost. No ground truth is used.

| | V2 | V2 + orientation selection |
|---|---|---|
| portrait images failing (SNR < 1 dB or no usable leads) | 14 / 16 | 4 / 16 |
| sideways phone photos (4) | -2.4 to -6.8 dB (read upside down) | +4.3 to +8.9 dB, all chose counter-clockwise |
| portrait monitor photos (10) | all failed | 6 rescued (8 to 13 dB), all chose no rotation |
| portrait degraded-paper photos (2) | 6.7, 8.3 dB | unchanged (default kept) |
| landscape controls (3) | | bit-identical (rule does not fire) |
| time per portrait image (Mac MPS) | 80 s | 338 s (4.2x; landscape images unaffected) |

The 3 images that scored lower by more than 0.5 dB were failures under both (about 0 dB to
-1.5 dB). The 4 remaining failures are PTB-XL monitor photos with other problems (for example a
6x4 layout misidentification). This is a quality change, not a speed change, so it is reported
separately from the V1 vs V2 equivalence claim. The extra cost could be cut by selecting the
rotation from a cheaper signal than a full re-run (for example the lead-name U-Net alone).

### Ahus reproduction with more images (outputs/mac_mps_ahus60 pooled with the 21 from the 200-image run)

78 unique Ahus images (32 scanner, 23 OnePlus, 23 iPhone), 120/120 runs successful in the new set.

| device | V1 mean SNR | V2 mean SNR | paper (Stenhede et al. 2026) |
|---|---|---|---|
| scanner | 19.62 dB | 19.67 dB | 19.65 dB |
| OnePlus | 11.82 dB | 11.84 dB | 12.19 dB |
| iPhone | 9.55 dB | 9.57 dB | 10.47 dB |

The scanner result reproduces the published value almost exactly; OnePlus and iPhone are within
0.4 and 0.9 dB with the same device ordering. V2 minus V1 per image: +0.03 dB, 95% CI
[-0.01, +0.08]; median absolute difference 0.0002 dB. Speed on Ahus on this machine is only
1.14x (67.3 s vs 59.2 s per image): the 4624 to 7016 px images are resampled to 3000 px and
swap on 16 GB, which dominates both versions.

### Is the photo gap a time-scale error? Mostly not (outputs/experiments/time_scale_fit_ahus.csv)

The worst Ahus image in the new set (D2_000062_l02, OnePlus, 2.8 dB) has an overlay that sits on
the printed traces almost everywhere, amplitudes within 7% of the reference, and a constant
~33 ms best shift on every lead, which suggested a small horizontal (pixel-spacing) error. On
that image a 1% stretch of the time axis lifts its SNR from 1.8 to 5.9 dB. Across all 78 Ahus
images, though, the best stretch is exactly 1.00 at the median on every device; only 3 of 23
OnePlus images (13%) need 1% or more, and none of the iPhone or scanner images. A perfect time
scale would add about 1.0 dB (OnePlus), 0.5 dB (iPhone) and 0.1 dB (scanner), against an 8 to
10 dB gap between scans and photos. The photo gap is therefore mostly trace-extraction quality,
not pixel-spacing estimation. (Diagnostic only: the stretch is chosen using the ground truth.)

### Why layout identification is slower inside V1 (outputs/experiments/layout_id_attribution.csv)

The layout identifier is identical code in V1 and V2, yet it took ~5 s per scan inside V1 and
~0.7 s inside V2. Two hypotheses, tested on two scans (3 timed runs each, after a warm-up):

| variant | layout identification (median) |
|---|---|
| V1 as published | ~5 s |
| V1, pixel size cast to Python float (type hypothesis) | ~4.6 s (within noise) |
| V1, MPS cache emptied right after cropping (memory hypothesis) | 0.75 s |
| V2 | 0.68 s |

The memory hypothesis explains it: the cached blocks from V1's one-hot cropper buffer slow the
later stages on unified memory. V2's cropper rewrite therefore saves time twice, in cropping
itself and in removing that downstream memory penalty, so the "layout identification" share of
the speedup in the stage attribution is caused by the cropper change too.

### Further optimisation candidates (torch.profiler on V2, one scan, outputs/experiments)

V2 takes 2.66 s on a 2130x1652 scan; over half of that is overhead from many tiny operations:
`aten::copy_` (9,087 calls, 0.77 s), `.item()` host syncs (912 calls, 0.71 s) and
`aten::nonzero` (9,453 calls, 0.68 s). Stage times: segmentation 0.72 s, signal extraction
0.75 s, layout identification 0.71 s, everything else under 0.3 s. Candidates, none applied yet
(each would need to pass the equivalence tests):

1. Signal extractor (identical in V1 and V2, runs on CPU): per-region Python loops and a greedy
   column-by-column path trace (`_trace_horizontal_path`) with an `.item()` per column. A
   vectorised or compiled (e.g. numba) version is the largest non-network target. Note it also
   adds `torch.randn(...) * 1e-6` tie-break noise, a second (negligible) source of randomness.
2. Host syncs and small copies in the lead identifier and signal extractor (about 900 `.item()`
   calls per image).
3. On memory-limited hardware, releasing the MPS cache between stages changes no arithmetic and
   may avoid swap; tested in outputs/experiments/empty_cache_between_stages.csv.
4. The lead-name U-Net runs at full working resolution and dominates layout identification on
   photos; a lower resolution would be faster but changes outputs, so it needs its own validation.

### Still to fill

- extension from 200 to 300 images (resumable; stopped on request)
- torch.profiler pass on V2 for further optimisation candidates
- CUDA numbers from the mentor's cluster run
