# Primary Target Selection (`pts`)

`pts` is a lightweight **post-tracker selection layer**.

It does one job:
- take tracked objects from any tracker,
- select and hold one primary target,
- return explainable state, reasons, and events.

No detector/tracker runtime is bundled in core package.

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
  --profiles balanced fast recall \
  --video-limit 5 \
  --demo-profile balanced \
  --demo-videos 2
```

Outputs:
- `reports/benchmark_per_video.csv`
- `reports/benchmark_aggregate.csv`
- `reports/benchmark_summary.json`
- `reports/benchmark_report.md`
- `demos/*.mp4`

## Metrics

Selection quality:
- `lost_per_1k_frames` (lower is better)
- `switched_per_1k_frames` (lower is better)
- `target_run_avg_frames` (higher is better)
- `flap_ratio` (lower is better)

Runtime:
- `avg_processing_fps` (higher is better)

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
