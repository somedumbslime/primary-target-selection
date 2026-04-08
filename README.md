# Primary Target Selection (`pts`)

`pts` is a lightweight **post-tracker selection layer**.

It does one job:
- take tracked objects from any tracker,
- select and hold one primary target,
- return explainable state, reasons, and events.

No detector/tracker runtime is bundled in core package.

## Problem statement

MOT alone answers: "what objects are tracked now?"

`pts` answers a different system question:
- "which single target should stay primary right now?"
- "when should target switch happen?"
- "why was this decision made?"

This is why `pts` is positioned as a **selection layer over tracking**, not as detector/tracker runtime.

## What `pts` does NOT do

- does not run detection
- does not run tracking backend internally
- does not require `ultralytics`/`torch` in core package
- does not include UI/replay stack

## Core contract

Input per frame:
- `frame_size` `(width, height)`
- list of tracks (`track_id`, `bbox_xyxy`, `confidence`, optional class info)

Output per frame:
- `primary_track_id`
- `selection_state` (`no_target`, `locked`, `switch_pending`, `lost`)
- `selection_reason`
- `primary_score`
- `switch_candidate_id`
- `events` (`target_acquired`, `target_lost`, `target_switched`, optional debug events)

## Install (core only)

```bash
pip install -e .
```

Core package dependencies are minimal (`pyyaml` only).

## Quick start (any tracker)

```python
from pts import PrimaryTargetSelection, SelectionTrack

selector = PrimaryTargetSelection()

tracks = [
    SelectionTrack(track_id=7, bbox_xyxy=(100, 120, 180, 260), confidence=0.81, class_id=0, class_name="soldier"),
    SelectionTrack(track_id=12, bbox_xyxy=(400, 200, 500, 360), confidence=0.74, class_id=0, class_name="soldier"),
]

out = selector.update(
    tracks=tracks,
    frame_size=(1920, 1080),
    frame_idx=0,
    timestamp_s=0.0,
)

print(out.selection_state, out.selection_reason)
print(out.primary_track_id)
```

Integration template:
- `selection_layer_embed_example.py`

## Policy plugins and external signals

`pts` supports pluggable scoring policies via config (`scoring.policy_name`):
- `single_best` (default)
- `center_biased`
- `stable_target`
- `largest_target`
- `class_priority`
- `auto` (stateful router: `search -> hold -> center_assist`)

You can also inject external hints per frame:

```python
out = selector.update(
    tracks=tracks,
    frame_size=(1920, 1080),
    frame_idx=i,
    timestamp_s=t,
    policy_name="center_biased",  # optional override
    external_signals={
        "preferred_track_id": 12,
        "track_score_bias": {12: 0.15},
        "class_score_bias": {"soldier": 0.10},
        "external_hint_score": {7: 0.08},
    },
)
```

Guardrails are enabled in default config (`pts/resources/target_selection.yaml`) to keep policy/external bonuses bounded:
- `max_policy_bonus_abs`
- `max_external_bonus_abs`
- `max_total_bonus_abs`

Policy intent (recommended usage):
- `single_best`: baseline/default, most predictable behavior
- `stable_target`: conservative lock behavior on noisy scenes
- `center_biased`: optional FPV-oriented bias, monitor switch rate carefully
- `largest_target` / `class_priority`: advanced/experimental
- `auto`: adaptive mode that keeps `stable_target` as default hold policy and applies `center_biased` only as assist mode

## Ultralytics example pipeline (outside core layer)

This repo also includes scripts for experiments with YOLO `model.track()`:

- `external_signal_sim.py` (video processing example)
- `benchmark_suite.py` (multi-video benchmark)

These scripts require heavy runtime dependencies (`ultralytics`, `opencv-python`).

Install them only when needed:

```bash
pip install -e .[examples]
```

Run example:

```bash
python external_signal_sim.py \
  --input-dir data/input \
  --output-dir data/output \
  --models-dir models \
  --device cuda \
  --tracker bytetrack
```

## Benchmark workflow (3-5 videos)

```bash
python benchmark_suite.py \
  --input-dir data/input \
  --output-dir data/output/benchmark \
  --models-dir models \
  --device cuda \
  --trackers botsort bytetrack \
  --policies single_best center_biased stable_target \
  --profiles balanced fast recall \
  --video-limit 5 \
  --selection-config pts/resources/target_selection.yaml \
  --baseline botsort:balanced:single_best \
  --demo-profile balanced \
  --demo-videos 2
```

Outputs:
- `reports/benchmark_per_video.csv`
- `reports/benchmark_aggregate.csv`
- `reports/benchmark_summary.json`
- `reports/benchmark_report.md`
- `demos/*.mp4`

Suggested grid for practical validation:
- trackers: `botsort`, `bytetrack`
- policies: `single_best`, `stable_target` (core), `center_biased` (validation only)
- profiles: `fast`, `balanced`
- videos: at least `3-5` different scenes

## Metrics

Selection quality:
- `lost_per_1k_frames` (lower is better)
- `switched_per_1k_frames` (lower is better)
- `switch_rate_per_minute` (lower is better)
- `target_acquisition_delay_frames` (lower is better)
- `primary_target_presence_ratio` (higher is better)
- `primary_target_stability_ratio` (higher is better)
- `mean_lock_duration_frames` / `max_lock_duration_frames` (higher is better)
- `target_run_avg_frames` (higher is better, legacy compatibility metric)
- `flap_ratio` (lower is better)

Runtime:
- `avg_processing_ms` (lower is better)
- `avg_processing_fps` (higher is better)

## Selection diagnostics (event logs)

```bash
python selection_diagnostics.py \
  --events data/output/benchmark/<run_id>/events \
  --glob "*.jsonl" \
  --output-json reports/selection_diagnostics.json \
  --output-csv reports/selection_diagnostics.csv
```

`selection_diagnostics.py` also supports `external_signal_sim` logs (`*_signals.jsonl`) via automatic nested-event parsing.

This utility aggregates:
- `target_acquired / target_lost / target_switched / candidate_rejected`
- dominant `selection_reason` and `reject_reason`
- explainability contribution stats (`avg_policy_contrib`, `avg_external_contrib`)
- lightweight failure flags (high switch/flap, low presence)

Built-in interpretation thresholds (configurable in CLI):
- `bonus_dominated_ratio <= 0.40`
- `policy_dominated_ratio <= 0.40`
- `external_dominated_ratio <= 0.50`
- `primary_target_presence_ratio >= 0.30`
- `primary_target_stability_ratio >= 0.90`
- `policy_clip_ratio <= 0.50`
- `external_clip_ratio <= 0.50`

These thresholds turn raw metrics into actionable diagnostics (`PASS` / `WARN` with reasons).

## Recommended configs

| Mode | Tracker/Profile/Policy | Why |
|---|---|---|
| Edge default | `bytetrack:fast:stable_target` | Best throughput with stable lock in typical runs |
| Minimal baseline | `bytetrack:fast:single_best` | Simplest fast setup for A/B checks |
| Conservative mode | `botsort:balanced:single_best` | Lower throughput, useful when ID continuity is prioritized |

Treat `center_biased` as scenario-dependent: validate with benchmark before using as default.

## Default recommendation

For edge setups, `bytetrack` is usually the default practical choice:
- much higher throughput,
- comparable or better lock-quality metrics in typical tests.

`botsort` remains useful when stronger ID persistence under occlusions is more important than FPS.

## Public API

```python
from pts import (
    PrimaryTargetSelection,
    SelectionTrack,
    SelectionOutput,
    SelectionEvent,
    SelectionScoreBreakdown,
)
```

## Release readiness

Pre-release checklist: `RELEASE_CHECKLIST.md`.
