# Dynamic Model Switching for Real-Time UAV Inspection

Reference implementation for my master's thesis on **dynamic model
switching (DMS)** for object detection on UAV inspection footage.

The core idea is simple: instead of committing to a single detector, run
a **fast** model (`n`) by default and switch to a **slower, more
accurate** model (`s`) only on the frames where the extra cost is
likely to pay off. A lightweight *policy* makes that per-frame decision
from cheap scene statistics or the detector's own confidence, so the
system keeps the fast model's latency on easy frames while recovering
the accurate model's detections on hard ones.

The accompanying ISCSET 2026 paper is included under
[`paper/`](paper/ISCSET2026_DMS_UAV_Inspection.pdf).

## What the thesis shows

1. **Dynamic switching improves detection quality** over a static
   lightweight model, and the gain is largest on the more challenging
   inspection domains.
2. **The best switching policy is domain-dependent, but stable across
   model architectures.** On glass-insulator footage `combined_hyst` is
   the strongest non-trivial policy on all four model-pair
   configurations; on porcelain it is `entropy_only`, again on all four.
   The same split holds across YOLOv8 and YOLO26 and across `n↔s` and
   `n↔l` size pairs.
3. **Confidence-based switching needs careful, per-domain calibration.**
   Its confidence-drop trigger is F1-optimal at different thresholds on
   glass (`c_high ≈ 0.02`) and porcelain (`c_high ≈ 0.12`), so a single
   global threshold does not transfer.
4. **Scene-feature policies outperform confidence-only switching** in
   these inspection tasks.
5. **The payoff depends on the hardware latency gap** between the fast
   and accurate models. On a saturated platform (RTX 5090 with small
   YOLOs, sub-millisecond gap) switching buys quality at no latency
   cost; the latency trade-off itself only appears on edge hardware
   (CPU, Jetson).

## Repository layout

```
dms_thesis/
  paths.py             path helpers and external data roots
  configs/
    base.yaml          default parameters; per-platform configs override
  dms/                 the library — pure decision + measurement logic
    controllers.py       RollingPercentile, Hysteresis, SwitchStabiliser
    policies.py          the seven switching policies
    proxies.py           scene-feature computation
    matching.py          Hungarian / IoU detection matching
    metrics.py           iou_match, count_agree, latency
    inference.py         fast / accurate detector wrapper
  experiments/         command-line runners (orchestration only)
    run_pipeline.sh      runs steps 1-5 below on one video in one command
    run_reference.py     n_only + s_only on one video (no switching)
    run_features.py      per-frame scene proxies (no inference)
    run_sweep.py         all policies on one video
    validate_sweep.py    sanity gates that must pass before scaling up
    build_master_table.py
  analysis/            result analysis: trigger validity, Pareto, figures
  tests/               unit tests for the library
  results/             experiment outputs (not tracked)
```

The `dms/` package holds all the logic; the runners under
`experiments/` only stream video, call into `dms/`, and write results.
Figures and CSV/JSON outputs are regenerated from the analysis scripts
and are intentionally not committed.

## Quick start

### 1. Install

```bash
conda create -n dms python=3.10 -y
conda activate dms
pip install -e ".[infer,analysis,dev]"
```

`infer` pulls the detector backend (`ultralytics`, which installs
`torch`); `analysis` adds `pandas` + `matplotlib` for the figures;
`dev` adds the test runner. The `dms/` library on its own only needs
`numpy`, `opencv-python`, and `pyyaml`.

### 2. Provide weights and a video

Trained detector weights are not bundled. Put your own `.pt` / `.onnx`
/ `.engine` files and an input video anywhere — the convention below
uses a `data/` folder:

```
data/
  models/
    glass_y8n_fast.pt        # fast (n) detector
    glass_y8s_accurate.pt    # accurate (s) detector
  videos/
    glass.mp4
```

### 3. Run the whole pipeline (one command)

```bash
# 50-frame CPU smoke test — finishes in a couple of minutes:
bash experiments/run_pipeline.sh \
    data/videos/glass.mp4 \
    data/models/glass_y8n_fast.pt \
    data/models/glass_y8s_accurate.pt \
    cpu 50

# Full run on GPU (0 = all frames):
bash experiments/run_pipeline.sh \
    data/videos/glass.mp4 \
    data/models/glass_y8n_fast.pt \
    data/models/glass_y8s_accurate.pt \
    cuda 0
```

This runs steps 1–5 below in order and writes everything under
`results/`, ending with `results/master_table.csv`.

## The pipeline, step by step

The one-command script above is just these five runners in sequence.
Run them individually when you want to inspect a single stage — they
are ordered so that a failure at step *k* points straight at the layer
that broke, without debugging the whole composition at once.

```bash
# 1. reference — inference + timing + metrics only (no switching)
python -m experiments.run_reference \
    --video data/videos/glass.mp4 \
    --fast-weights data/models/glass_y8n_fast.pt \
    --accurate-weights data/models/glass_y8s_accurate.pt \
    --device cpu --max-frames 50 --run-id demo

# 2. features — per-frame scene proxies, no inference
python -m experiments.run_features \
    --video data/videos/glass.mp4 --max-frames 50 --run-id demo

# 3. sweep — every policy on the video (writes per-frame CSV + summary)
python -m experiments.run_sweep \
    --video data/videos/glass.mp4 \
    --fast-weights data/models/glass_y8n_fast.pt \
    --accurate-weights data/models/glass_y8s_accurate.pt \
    --device cpu --max-frames 50 --run-id demo

# 4. validate — G1-G5 sanity gates; exits non-zero on any failure
python -m experiments.validate_sweep --run-dir results/stage_sweep/demo

# 5. aggregate — collect every run into one master_table.csv
python -m experiments.build_master_table --results-root results
```

Each runner takes `--help` for the full set of options (device, image
size, frame range, policy subset, conf-EMA calibration, and so on).

## Scaling up

For a full multi-pair sweep on a remote GPU, drive the batch script and
detach it so a dropped SSH session does not kill the run (finished model
pairs are skipped when you re-launch):

```bash
nohup bash experiments/run_stage3_sweep.sh > stage3.log 2>&1 &
tail -f stage3.log
```

## Figures

The scripts under `analysis/` turn `results/master_table.csv` and the
per-frame CSVs into the thesis figures and tables (Pareto frontiers,
trigger-validity curves, ablations). They need the `analysis` extra.
For example:

```bash
python -m analysis.stage3_pareto --master-table results/master_table.csv
```

## Tests

```bash
python -m pytest tests/ -v
```

## The seven policies

| name           | family     | reads                                |
| -------------- | ---------- | ------------------------------------ |
| n_only         | trivial    | nothing (fast baseline)              |
| s_only         | trivial    | nothing (always-accurate reference)  |
| entropy_only   | scene      | H                                    |
| combined       | scene      | L, H                                 |
| combined_hyst  | scene      | L, H, with hysteresis                |
| conf_ema       | confidence | fast-model mean confidence (EMA)     |
| multi_proxy    | composite  | L, H, colour entropy, Tenengrad, …   |

Optional comparison policies (`lr_policy`, `rf_policy`) and an
exploratory single-feature candidate (`local_contrast_hyst`) exist for
completeness but are not part of the locked seven-policy baseline.

## Terminology

The codebase and thesis text use one fixed vocabulary:

| term                       | meaning                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| fast model (n)             | YOLOv8n / YOLO26n                                                                         |
| accurate model (s)         | YOLOv8s / YOLOv8l / YOLO26s / YOLO26l                                                     |
| model pair                 | a (fast, accurate) tuple, e.g. `yolov8_n_s_glass`                                         |
| benefit-positive frame     | accurate detects ≥1 valid object that no fast detection matches at IoU ≥ 0.5             |
| trigger                    | the observable signal a policy reads (a proxy or confidence)                             |
| trigger validity           | how well a trigger predicts the benefit-positive label (AUC, MI)                        |
| policy                     | the decision function mapping a trigger to `{n, s}`                                       |
| always-accurate reference  | the per-frame `s_only` output, used as operational ground truth on unlabelled video      |
| iou_match                  | frames where policy detections match the reference at IoU ≥ 0.5 (primary correctness)    |
| count_agree                | frames where the policy's detection count equals the reference count (secondary metric)  |
| Pareto frontier            | the non-dominated set of (latency, iou_match) across policies                            |

## Datasets

| label                 | what it is                                     |
| --------------------- | ---------------------------------------------- |
| INS-S                 | independent labelled APOLI subset (validation) |
| INS-M / glass         | glass-insulator UAV inspection set             |
| INS-M / porcelain     | porcelain-insulator UAV inspection set         |
| UAV-video / glass     | glass-insulator UAV video benchmark            |
| UAV-video / porcelain | porcelain-insulator UAV video benchmark        |
