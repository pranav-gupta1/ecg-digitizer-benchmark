# CLAUDE.md: Stenhede vs StenhedeV2 ECG Digitizer Benchmark

Handoff from a claude.ai planning chat (Oct 2026). Read fully before starting work,
including Appendix A (the user's original notes from the mentor meeting, verbatim).
If anything below conflicts with Appendix A, Appendix A wins; ask the user.

## Goal and framing

Research project with a mentor. Outputs: poster, conference publication, and a clean
public-quality project repo.

- **The contribution:** a faster version of the Stenhede digitizer (StenhedeV2, by Rayan)
  that gives up essentially nothing in digitization quality, plus a new feature: an
  overlay that draws the digitized traces back onto the original image, which helps
  interpretability and makes output quality easy to see.
- **The research part:** benchmarking the Stenhede model on an external, validated dataset
  (ECG-Image-Database).
- **The key claim to prove:** at scale, find the fastest method that trades off almost
  nothing, document exactly which change gives the speedup, and prove it holds on a
  validated dataset.
- **Why speed matters:** digitizing slowly blocks any large-scale downstream task. A fast,
  validated digitizer opens up many downstream studies (see Extension studies).
- **Context:** the mentor's group is also externally validating an external-source model,
  which will take a while. The lab is getting IRB approval for a research study on
  digitization.

## Sources

- Paper: Stenhede, Bjørnstad, Ranjbar. "Digitizing paper ECGs at scale: an open-source
  algorithm for clinical research." npj Digital Medicine 9, 145 (2026).
  https://www.nature.com/articles/s41746-025-02327-1 (validation info is in its
  Results > Validation data / Validation metrics / Ahus and Emory sections)
- Official baseline code: https://github.com/Ahus-AIM/Open-ECG-Digitizer
- Third-party copy the mentor linked: https://github.com/18h32n/Open-ECG-Digitizer
  (last commit cca543c, Oct 2025). Open question: which one V2 was forked from.
- V2: https://github.com/justinadjasu/I-Love-Paper/tree/main/src/StenhendeV2
  (folder spelled "Stenhende"; the author's name is Stenhede)
- Dataset: https://www.kaggle.com/datasets/physionet/ecg-image-database/
- Dataset paper: Reyna et al., Physiol. Meas. 47(7) 075015 (2026);
  arXiv 2409.16612; PMC13402943
- Rayan (StenhedeV2 author) can provide extra figures if needed.

## The two models

- **V1:** Stenhede Open-ECG-Digitizer (original, as published).
- **V2:** StenhedeV2 in the mentor's/Rayan's I-Love-Paper repo.

### I-Love-Paper repo context

Stanford BMDS 212 project: ECG photo acquisition, digitization, and evaluation. Aims:
(1) acquisition-quality thresholds and smartphone capture guidance, (2) whether digitized
waveforms preserve fiducials, intervals and morphology, (3) downstream utility with
signal-based tools. Python 3.12, managed with uv, PEP 723 inline-dependency scripts.

Most of that repo is not needed here. Relevant pieces:
- src/StenhendeV2/digitizer/ (pipeline), src/StenhendeV2/overlay/ (overlay)
- src/StenhendeV2/scripts/stenhede_digitize.py (folder of images to CSV + 4-panel PNG)
  and stenhede_overlay.py (single image to colored overlay PNG)
- Weights path comes from STENHEDE_WEIGHTS_PATH in data.env
- src/ReconPW/ (lead reconstruction with confidence intervals, includes a
  StenhendeV2-to-ReconPW bridge) is relevant to the extension study later
- src/QTNet, src/QualitySignals, src/SyntheticImagePipeline may be useful later

### V1 vs V2 differences (verified by reading code)

IMPORTANT: this diff was taken against the 18h32n copy (commit cca543c), not the official
Ahus-AIM repo. Re-run the diff against the official repo before relying on it.

Identical files: unet.py, signal_extractor.py, lead_identifier.py, dewarper.py.
Identical weights (same LFS SHA-256):
- unet_weights_07072025.pt (sha256 17fe7071..., 90 MB)
- lead_name_unet_weights_07072025.pt (sha256 840bd6bf..., 23 MB)

Weights download works from:
https://media.githubusercontent.com/media/Ahus-AIM/Open-ECG-Digitizer/main/weights/<file>.pt
(or git lfs pull in a clone).

Changed files (all speed changes):
- cropper.py: Python loops over (theta, rho) replaced with vectorized scatter_add_;
  the one_hot buffer replaced with scatter_add_; tensors created on the input's device.
  The dtype changes from int to long, and there is a new clamp(max=granularity-1).
  Check that the clamp never alters results.
- pixel_size_finder.py: np.correlate autocorrelation replaced with a torch FFT
  autocorrelation; the per-candidate Python loop in the grid search replaced with one
  batched matmul + argmax. Returns floats. Check equivalence: FFT float error, and
  tie-breaking (the old loop kept the first max via strict <; argmax also returns the first).
- perspective_detector.py: randperm/meshgrid/arange created on device (avoids
  CPU-GPU copies). randperm subsampling is random, so seed it for comparisons.
- inference_wrapper.py: import path change only.
- Config: only class paths and ${STENHEDE_WEIGHTS_PATH} expansion differ. V2 ships
  only the george-moody-2024 config. V1 also has inference_wrapper_ahus_testset.yml
  (uses lead_layouts_reduced.yml), which is needed for Ahus images. Both hardcode
  device 'cuda:0'; this must be parameterized (cpu / cuda / mps).

The mentor warned that some past version might have skipped steps and hurt output
quality, which is not acceptable now. The audit must confirm that V2 runs every step V1
runs (same stages, same config values, same resample size, nothing silently disabled).

### Overlay (new in V2)

src/StenhendeV2/overlay/ inverts the geometric chain (resample to 3000 px, perspective
warp, optional 90 deg rotation, y-crop, empty-column drop) by monkey-patching _crop_y and
_rotate_on_resample to capture hidden state. Per-lead coloring only works for the
3x4+1R layout (exactly 4 extracted lines). Nothing equivalent exists in the original
repo (only training-time augmentation "overlays" in src/transform/vision.py).

Tasks:
1. Verify the overlay works: traces land on the printed traces across scans and photos,
   including rotated and perspective-distorted images.
2. Confirm it is not in the original codebase and is not a trivial addition (document
   what it takes: inverting the geometric chain, capturing hidden state).
3. If unique, generate representative overlay plots (good, median, and failure cases).

## Experiments

### 1. Dataset characteristics

ECG-Image-Database: 37,191 images from 2,243 records across three sources
(D0 PTB-XL, D1 Emory, D3 Ahus). Emory and PTB-XL images are 25 mm/s; Ahus is 50 mm/s.
It contains the paper's Ahus set (1,596 images, 266 ECGs, 266 patients) and Emory set
(35,595 images), so the benchmark doubles as the paper reproduction. The full download
may be very large; check file sizes before downloading.

Produce a table with:
- unique patients, unique records/waveforms, total images, per source
- breakdown of image types: scans (color, black-and-white), mobile photos, by device
  (scanner, iPhone, OnePlus for Ahus), and degradation conditions (clean, stained,
  soaked, mold, wrinkles, etc. for Emory)
- paper speed and gain (mm/s, mm/mV), layouts, resolution

### 2. Run both digitizers and compute quality metrics

Run V1 and V2 on the same images. Save the digitized waveform from each. Compute per lead:
SNR (dB), MSE, RMSE, Pearson, and NaN fraction. Lead II at the bottom is the longest
strip, so report it separately as "rhythm".

Use the paper's method (implemented in V1 src/evaluate.py; V2 dropped it, so use V1's
evaluator for both):
- SNR = 10 log10(sum y^2 / sum (y - yhat)^2)
- Ground truth at 1000 Hz; resample the digitized signal to match
- Align: zero-center both; horizontal shift up to +/-100 ms (evaluate.yml shift_steps 200)
- The algorithm sets unreadable samples to NaN; report the NaN fraction
- Output units: microvolts
- Analyze separately by capture device and condition

### 3. Speed and performance metrics

- Wall time per image and per pipeline stage, throughput, min/mean/max
- FLOPs: the networks are identical in V1 and V2, so model FLOPs are identical. Still
  report FLOPs per image for both U-Nets (for the paper), and achieved FLOP/s.
  The speedup shows up in the non-network stages, so wall time is the main comparison.
- GPU usage if possible: utilization, peak memory. CPU peak RSS on CPU runs.
- Record hardware, OS, torch version, and device for every run.

### 4. Audit V1 vs V2

- Know exactly which changes give the speedup (per-stage attribution).
- Check whether any change cost output quality (module-level equivalence tests plus
  end-to-end waveform comparison).
- Identify anything broken, and anything that can be sped up further (profile with
  torch.profiler). Any new optimization must pass the equivalence tests.
- Document the final "fastest method with near-zero tradeoff" and prove it on the dataset.

### 5. Reproduce the npj paper's validation

Reproduce the metrics and figures from the paper's validation section on the same data.

- Ahus: mean SNR scanner 19.65 dB, OnePlus 12.19, iPhone 10.47; RMSE and correlation
  (Table 1); NaN fraction about 0.21%; time-shift histogram (Fig 2); per-device SNR
  scatter and distributions (Fig 3); Lead II at the 5th/25th/50th/75th/95th SNR
  percentiles per device (Figs 4 to 6).
- Emory: SNR by capture condition (Table 2): 7.34 dB color scans, 7.24 b/w scans,
  1.05 to 2.03 dB photos and degraded paper.
- Speed: 5.88 s/image on GPU (min 5.43, max 7.45); 16.78 s/image on CPU (15.53 to
  23.39). Hardware is in the paper's Table 3.

## Deliverable: this repo

A clean repo with both pipelines in tidy folders (original and V2), leaving out the
unneeded I-Love-Paper code. It must be shareable with someone on another machine with a
compute cluster.

```
ecg-digitizer-benchmark/
  environment.yml            # conda env, python 3.12, torch (cpu/cuda/mps)
  README.md
  CLAUDE.md
  configs/benchmark.yaml     # paths, n_images, device, seed, layouts
  third_party/               # pinned copies of V1 and V2
  bench/
    run_benchmark.py         # main entry point
    pipelines.py             # uniform wrapper around V1 and V2
    evaluate.py              # ported paper evaluator (+ MSE, rhythm strip)
    profiling.py             # per-stage timing, peak memory, GPU util, FLOPs
    overlay.py               # V2 overlay figures
    report.py                # summary tables + figures
    dataset_stats.py         # dataset characteristics table
  notebooks/colab_run.ipynb  # later, optional (see Colab option)
  scripts/download_weights.sh
  tests/equivalence/         # V1 vs V2 module-level equivalence tests
```

run_benchmark.py flags (all overridable in YAML):
- --data-dir PATH (path to the ECG database, customizable)
- --n-images N or --fraction 0.1 (deterministic sample with --seed)
- --subset ahus|emory|ptbxl|all
- --output-dir PATH (where waveforms, metrics, and figures are saved)
- --device auto|cpu|cuda|mps (auto checks for a GPU: cuda, then mps, then cpu)
- --pipelines v1,v2
- --save-overlays N

Outputs per run: digitized waveforms (CSV per image per pipeline), per-lead metrics CSV,
timing CSV (per stage), summary log (summary.md + summary.json), figures (SNR by
device/condition, V1 vs V2 scatter, speed bars, percentile Lead II plots like paper
Figs 4 to 6, overlays), and env info (hardware, versions, git commits).

### Sharing workflow

1. The user sets everything up and tests it on 10, then 100 to 300 images on a Mac
   (mps or cpu). The mentor said even 100 working images would be great.
2. The project folder goes on GitHub, and/or Google Drive.
3. The mentor sets up the conda environment, corrects the paths, changes the flags
   (--data-dir, --n-images, --device cuda), and runs the full dataset on a cluster.
   The script then generates everything for the full set.

Keep setup for the mentor to a few copy-paste commands in the README.

### Colab option (backup, probably not used)

The mentor described an alternative: keep a local project folder synced to Google Drive,
with the two GitHub repos cloned into it. Add a single notebook at the top level that
opens in Google Colab and has about three steps:
1. Inline unzip of the dataset into the Colab runtime (!unzip ...).
2. A code cell that runs the full benchmark script above.
3. Save outputs to a specified location (e.g. the mounted Drive).
The idea: generate the codebase and notebook on the laptop, then open the synced notebook
in Colab and run it on Colab's GPU. Note: the Colab runtime can disconnect, so the script
should resume from where it stopped.

## Plan

1. Scaffold repo, environment, weight download. Pin V1 and V2 commits.
2. Pipeline wrappers with device parameterization and per-stage timers (same code path
   for both so timing is fair; warm up the GPU before timing; synchronize before reading
   timers). Make runs resumable (skip images already done).
3. Port the evaluator; match ground truth to images using the dataset's naming convention.
4. Dataset characteristics table.
5. Smoke test on 10 images, then 100 to 300.
6. Equivalence tests: feed identical tensors to V1 and V2 versions of cropper,
   pixel_size_finder, perspective_detector; assert outputs match within tolerance.
   Then end-to-end: V1 vs V2 waveform difference per image. Confirm no skipped steps.
7. Speedup attribution per stage; FLOPs and GPU usage; list further optimization
   candidates.
8. Overlay verification and figures.
9. Report: dataset table, metrics tables, figures, comparison against paper numbers.
10. Hand off to the mentor.

## Extension studies (later, not now)

Only after the main project works and the speedup is rigorously confirmed. Save the
digitized waveforms from the main run so these can reuse them, assuming they are not
inferior to the baseline.

Idea A (may be complicated): use a strong ECG foundation model (e.g. ECGFounder) on the
digitized waveforms. The image dataset's disease labels may not be known. ECGFounder may
have been trained on PTB-XL, so there could be data leakage. That could be acceptable if
the model is only used to pseudo-label the image dataset. The user is not sure about the
details.

Idea B (preferred): assume digitization is near-perfect and use an open-source deep
learning ECG model that has not seen any of this data. Test whether feeding digitized ECGs
into it retains the performance it gets on the original signals. This mirrors the user's
earlier Ribeiro-model project:
- Leg 1: digitized ECG into an off-the-shelf model. Does it do well?
- Leg 2: reconstruct the missing portions of the digitized ECG (e.g. with ReconPW) and
  feed that in. Does performance improve?
That earlier end-to-end pipeline was never published. The goal now is to do it properly
with a well-defined model, benchmark the open-source digitization approach, show the
improvement, and publish quickly.

## Open questions (ask the user)

- V2 fork source: official Ahus-AIM or 18h32n? Default: official, pinned to one commit.
- Has anyone measured V2's speedup? On what hardware?
- "A past version skipped steps and hurt output quality": specific commit or general caution?
- Is it OK to copy StenhendeV2 code into this repo? Neither upstream repo has a LICENSE file.
- Repo public or private? Default: private.
- Any IRB or data-use limits on sharing outputs?

## Conventions

- Keep prose concise and direct. No em dashes.
- Give one recommended path, not menus of options.
- Give copy-paste ready terminal commands.
- Never commit the dataset, weights, or outputs (add them to .gitignore).

---

## Appendix A: the user's original notes (verbatim)

From a meeting with the research mentor. Typos kept as written. Some numbering came from
formatting and may not be meaningful.

```
[this is from a research project mentor about some optimized Stenhende benchmark. i am putting everything related to the project, not sure what is directly for it]: (ignore numbers that are sorting the information, that got put in becuase of formatting, i don't know how releveant it is)


1. Database: ECG-image-database
   * List the characteristics of the dataset, i.e., the number of unique patients, waveforms, etc.
   * The breakdown of image types, photography types, scans, etc.

Models:
   * Stenhende Open-ECG-Digtizer
   * Stenhende V2 Model on my Github

Experiments:
   * Run both digitizers on the open ECG dataset. Collect the digitized outputs (the waveform) for each model. Compute metrics for SNR (dB), MSE, RMSE, Pearson, etc. per-lead. (Lead II strip at the bottom is the longest, so might have to designate that separately as "rhythm").
   * Verify with LLM of choice / coding agent of choice that the overlay functionality of Stenhende V2 works and is not encompassed in the original Stenhende codebase. IF unique, generate some representative plots of the overlays.

Ask Rayan for some other figures if need be.
2. Audit the difference between the original stenhende and v2, so you know what changes contributed to the speed up and did it cost something in terms of output quality / digitization quality or not.
3. - audit the optimizations as well. identify things that are messed up or can be sped up more. cool step is the step where an overlay of the reconcstructuve (like the gitized image) back onto the orignal image. it's good for interpretability and visualizating output quality. feature in the v2 version where one can overlay what was digitized back onto the image image. have to make sure that's not a trivial addition. some past version might've skipped steps and messed up output quality whcih we don't want now. the main thing is at scale is to find the fastest method that tradeoff almost nothign and document that change. prove that is true on a vlaidated dataset. bc we also doing external validation of an external soruce model whic will take a while. the point is that the we will contribute it via speed of like a faster version. and the faster version has this like new feature. if results are good of this proejct that's ideal (research part is the becnharking of the stenhende model on external dataset). 
4. okay so also what i can do is repoduce the validation study in the original npj digital medicine paper. so looking at info on the paper, i can repdoduce like the metrics and figures of the datbase they did their stuff on in npj paper so ahus paper digitzization ecg databse or whatevre databse they are talking about. bc proabbly it's the same. this rleevant info is under the vlidation section of the nature diigtal meidicne paper. and then what i need to idenity is metrics and like the flops of perofrmance metrics if possible. 
5. i should create a nice github. should have a nice folder with like the two versions/pipliens of the model lie the orignal and stenhende v2 portion. there is a lot of code in the stendende v2 github that is already there that is not necssary but like what i could do is genreate a nice codebase tha tworks and have a conda environement etc. such that i can run this pipeline....Create a codebase and script and yaml file etc. such that you could share with someone on another environment who has access to a compute cluster the project your work. The script should have a flag such that you can specify what % or how many numbers of the ECG-image database you want to run, and then it generates summary statistics, saves those waveforms, etc. all the above. Specify the path to the ECG database and such as well so that it can be customizable. That way you can set everything up and test it on like 300 images, but then pass the project folder to someone else to run the whole thing. the idea is that sure let's say the compute for this stuff will take forever but i have a mac or something can kind of survice. what i can do is create a codebase with a script, yaml, etc. with my ai agent of choice. host on drive or gogle drive or somthign liek that. i can have it work on like 300 images or something. it prodcues all the exact outputs and metrics and figures and saves the waveforms to like a specificied path. then i can send this to my mentor he can run it with a full dataset. like he just changes the flaog of the script. like he sets up the conda environemtn and like correct the paths and then the mentor runs on like a cluster and then so if i don't have compute access then the metnor can jut do it on all of them and it'll just generate for the mentor easily/reasonably. 

(probably won't use this option but just putting it here since it's part of the the meeting i had with my mentor: this is a differentaive alternateive. bascialy i create a folder on my local computer and sync it will google drive. so like on my computer and it's on my directory on drive. and then i get a coding agent and i tell it, "here i git cloned these two things which is implying the two github repos. here is the epxiernment. i need to haev this work such that at the top level of the directory you create a single ipynb file for like notebook which will open up in google colab which will have like three steps where do like an inline upzip of the dataset. runtime woudl go out. and then i can like run a code cell that runs the whole script that I'm describing above ["Create a codebase and script and yaml file etc. such that you could share with someone on another environment who has access to a compute cluster the project your work. The script should have a flag such that you can specify what % or how many numbers of the ECG-image database you want to run, and then it generates summary statistics, saves those waveforms, etc. all the above. Specify the path to the ECG database and such as well so that it can be customizable. That way you can set everything up and test it on like 300 images, but then pass the project folder to someone else to run the whole thing. !unzip of the dataset to the runtime on Colab"] and saves it to somewhere specified on Colab. idea is that can generate the whole codebase on my laptop and ipynb file. and then i should be abke to go to colab there it should be synced and i run the nobtebook there as a googel colab notebook.

1. extension study for later, not now based on this project: have to get the main proecjt working first and stringently make sure that the way faster works bc then the way faster version opens up so many more downstream studies. can't realtisitacly wait a long time to digitize an ecg/ecgs for any large-cale task. so the next study that is possible probably has somethign to do with: so i'll save the digitize waveforms let's asusme they're not inferior to the baseline to the abseline model. then what i can do i pull out a foundation model of chocie such as ecgfounder (ecg founder is okay bc there will be data leakeg since ecgfounder is trained on ptb right which is fine, wait actually this one part might not be true. what trying to say is that don't konw what the disease labels are in lie the image dataset. let's say use overpowered foundation model for ecgs. and even if there's data leakege that fine becasuse then can us that to psuedo-label the image datset. but this might be complciated...so will explain a differnt extensino study)
2. okay so let's assume i digitize the ecg data perfectly and let's pretend there is a foundation model that hasn't seen any of the data in there and it predicts a bunch of differnt things. this is similar to waht i did with a ribiero model proejct many moths ago where in that project it was like checking if i put in the digitzed ecg into an out of the box model, does it do well; and then in the second leg of the study, if i take the digitized ecg and reconstruct the missing portions and then put it into the model like an out of the box model, does it imporove; and now that's like a full end-to-end pipeline that we didn't published but it's whatever. so like bsacially tha tproejct and this project is becnhamrk digitization approach that's open source, would like to improve upon it and then i add another thing and i show that it imporves open-source usage. so kind of thinking about where i'm like if i digitiza a model and then feed the digitzation into an open-source deep learning model, do i retain the performance? so bsaically this but properly so like for a much more defined model and then report that and publish it immediately.

1. !unzip of the dataset to the runtime on Colab
2. Run a code cell that runs the whole script that I'm describing above and saves it to somewhere specified on Colab

1. FLOP/s 
Other performance metrics 
GPU usage if possible.

overall goal is poster, conference publication, project, etc.

https://www.kaggle.com/datasets/physionet/ecg-image-database/
https://www.nature.com/articles/s41746-025-02327-1
https://github.com/justinadjasu/I-Love-Paper
https://github.com/justinadjasu/I-Love-Paper/tree/main/src/StenhendeV2
https://github.com/18h32n/Open-ECG-Digitizer


It's okay if you can make a GitHub, or at least some kind of project folder or directory, that has:

- An environment file that I can use, or some kind of Python virtual environment, so I can load all the dependencies.

- A script that has parameters for the directory that holds the images and how many images you'll run the benchmarking test with.

It should just run the original Stenhende and the optimized one on those same images, produce a summary log, and save the outputs as well. The summary log should be of the performance, like signal-to-noise ratio, RMSE, etc.

That way, I can take that codebase and point it to a larger folder with more images in it. You would probably be doing this on a laptop or something. So just ask Codex or something to have the option to set the device pytorch uses as well. In your case, Torch will use the CPU if you don't have a gaming computer. 

So just have it be parameterized such that I can put a flag in the script for GPU and have it check for GPU, etc. 
If you could get it to work on even 100 images that would be great. 

My lab is working on getting IRB approval for a research study on digitization
```
