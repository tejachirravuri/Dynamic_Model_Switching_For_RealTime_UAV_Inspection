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

## Setup

```
conda activate uav-thesis
pip install -e .[dev]
```

Trained detector weights are not bundled — point the runners at your own
`.pt` / `.onnx` / `.engine` files with `--fast-weights` and
`--accurate-weights`.

## Running an experiment

The runners are ordered so that a failure at step *k* tells you exactly
which layer is broken, without debugging the whole pipeline at once:

```
1. python -m experiments.run_reference  --video <v> --fast-weights <f> --accurate-weights <a>
     inference + timing + metrics + I/O only.

2. python -m experiments.run_features   --video <v> --proxy-size 160
     the scene-proxy layer in isolation (no inference).

3. python -m experiments.run_sweep      --video <v> --fast-weights <f> --accurate-weights <a>
     composes the two: runs every policy, writes per-frame CSV + summary JSON.

4. python -m experiments.validate_sweep --run-dir results/stage_sweep/<run_id>
     sanity gates (G1–G5); exits non-zero on failure. Run before any large sweep.

5. python -m experiments.build_master_table --results-root results
     aggregates every summary into one master_table.csv for analysis.
```

For long sweeps over an unreliable connection, detach the process so a
dropped SSH session does not kill it (finished model pairs are skipped
on re-run):

```bash
nohup bash experiments/run_stage3_sweep.sh > stage3.log 2>&1 &
tail -f stage3.log
```

## Tests

```
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
