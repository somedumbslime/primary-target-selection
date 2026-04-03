from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from pts import PrimaryTargetSelection, SelectionTrack
from pts.target_selection.reporting import load_event_rows, summarize_event_rows


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MODEL_EXTENSIONS = (".onnx", ".pt", ".engine")
TRACKER_TO_ULTRA = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


@dataclass
class Profile:
    name: str
    conf: float
    iou: float


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _pick_model(models_dir: Path, explicit_path: str = "") -> Path:
    if explicit_path:
        path = _resolve(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")
    candidates = [p for p in sorted(models_dir.iterdir()) if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(f"No model files found in {models_dir} ({MODEL_EXTENSIONS})")
    return candidates[0]


def _collect_videos(input_dir: Path, video_limit: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_dir}")
    if video_limit > 0:
        videos = videos[:video_limit]
    return videos


def _load_profiles(profiles_path: Path, profile_names: list[str]) -> list[Profile]:
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
    with profiles_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError(f"Invalid profiles YAML: {profiles_path}")

    selected_names = profile_names if profile_names else list(profiles_raw.keys())
    profiles: list[Profile] = []
    for name in selected_names:
        cfg = profiles_raw.get(name)
        if not isinstance(cfg, dict):
            raise KeyError(f"Profile not found: {name}")
        profiles.append(Profile(name=name, conf=float(cfg.get("conf", 0.22)), iou=float(cfg.get("iou", 0.50))))
    return profiles


def _class_name(prediction: Any, cls_id: int) -> str:
    names = getattr(prediction, "names", {})
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _prediction_to_tracks(prediction: Any) -> list[SelectionTrack]:
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []
    ids = getattr(boxes, "id", None)
    if ids is None:
        return []

    tracks: list[SelectionTrack] = []
    for i, track_id_val in enumerate(ids.tolist()):
        bbox = tuple(float(v) for v in boxes.xyxy[i].tolist())
        conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        tracks.append(
            SelectionTrack(
                track_id=int(track_id_val),
                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=_class_name(prediction, cls_id),
                visible=True,
            )
        )
    return tracks


def _reset_ultralytics_trackers(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    if hasattr(predictor, "trackers"):
        try:
            delattr(predictor, "trackers")
        except Exception:
            predictor.trackers = []


def _draw_selection_overlay(frame: Any, output: Any) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]
    primary_id = output.primary_track_id

    for track in output.candidates:
        if not track.visible:
            continue
        x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        is_primary = int(track.track_id) == int(primary_id) if primary_id is not None else False
        color = (0, 80, 255) if is_primary else (60, 200, 50)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_primary else 2, lineType=cv2.LINE_AA)

        score_txt = "n/a" if track.score is None else f"{track.score:.2f}"
        label = f"{track.class_name} id={track.track_id} score={score_txt}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = max(0, min(x1, w - tw - 8))
        by1 = max(0, y1 - th - baseline - 6)
        bx2 = bx1 + tw + 8
        by2 = by1 + th + baseline + 6
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thickness=-1)
        cv2.putText(
            out,
            label,
            (bx1 + 4, by2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def _process_video(
    model: Any,
    selector: PrimaryTargetSelection,
    tracker_name: str,
    tracker_ref: str,
    profile: Profile,
    video_path: Path,
    events_path: Path,
    demo_output_path: Path | None,
    max_frames: int,
    progress_every: int,
    device: str,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    h, w = first_frame.shape[:2]
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps = source_fps if source_fps > 1e-6 else 25.0

    writer = None
    if demo_output_path is not None:
        demo_output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(demo_output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))

    selector.reset()
    _reset_ultralytics_trackers(model)
    if events_path.exists():
        events_path.unlink()
    selector.set_event_output(str(events_path))

    frame = first_frame
    frame_idx = 0
    started = time.perf_counter()
    try:
        while True:
            ts = frame_idx / max(source_fps, 1e-6)
            pred = model.track(
                source=frame,
                conf=float(profile.conf),
                iou=float(profile.iou),
                tracker=tracker_ref,
                persist=True,
                device=str(device),
                verbose=False,
            )[0]
            tracks = _prediction_to_tracks(pred)
            out = selector.update(
                tracks=tracks,
                frame_size=(w, h),
                frame_idx=frame_idx,
                timestamp_s=ts,
            )

            if writer is not None:
                writer.write(_draw_selection_overlay(frame, out))

            frame_idx += 1
            if progress_every > 0 and frame_idx % progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"[BENCH] {video_path.name}: {frame_idx} frames | proc_fps={frame_idx / elapsed:.2f}", flush=True)

            if max_frames > 0 and frame_idx >= max_frames:
                break

            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                break
            frame = next_frame
    finally:
        cap.release()
        selector.set_event_output(None)
        if writer is not None:
            writer.release()

    elapsed = max(time.perf_counter() - started, 1e-6)
    events = summarize_event_rows(load_event_rows(events_path))
    switched = _safe_int(events.get("switched"), 0)
    lost = _safe_int(events.get("lost"), 0)
    flap_checks = _safe_int(events.get("flap_checks_total"), 0)
    flap_le_5 = _safe_int(events.get("flap_le_5_frames"), 0)

    return {
        "frames": frame_idx,
        "elapsed_sec": elapsed,
        "avg_processing_fps": (frame_idx / elapsed) if frame_idx > 0 else 0.0,
        "target_switched": switched,
        "target_lost": lost,
        "switched_per_1k_frames": (1000.0 * switched / frame_idx) if frame_idx > 0 else 0.0,
        "lost_per_1k_frames": (1000.0 * lost / frame_idx) if frame_idx > 0 else 0.0,
        "target_runs_count": _safe_int(events.get("target_runs_count"), 0),
        "target_run_avg_frames": _safe_float(events.get("target_run_avg_frames"), 0.0),
        "flap_ratio": (flap_le_5 / flap_checks) if flap_checks > 0 else 0.0,
        "events": events,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["tracker"]), str(row["profile"]))].append(row)

    agg_rows: list[dict[str, Any]] = []
    for (tracker, profile), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        frames_total = sum(_safe_int(i["frames"]) for i in items)
        elapsed_total = sum(_safe_float(i["elapsed_sec"]) for i in items)
        switched_total = sum(_safe_int(i["target_switched"]) for i in items)
        lost_total = sum(_safe_int(i["target_lost"]) for i in items)
        flap_num = sum(_safe_float(i["flap_ratio"]) * max(_safe_int(i["events"]["flap_checks_total"]), 1) for i in items)
        flap_den = sum(max(_safe_int(i["events"]["flap_checks_total"]), 1) for i in items)
        target_runs_total = sum(_safe_int(i["target_runs_count"]) for i in items)
        run_avg_weighted_num = sum(_safe_float(i["target_run_avg_frames"]) * _safe_int(i["target_runs_count"]) for i in items)

        agg_rows.append(
            {
                "tracker": tracker,
                "profile": profile,
                "videos_count": len(items),
                "frames_total": frames_total,
                "elapsed_sec_total": elapsed_total,
                "avg_processing_fps": (frames_total / elapsed_total) if elapsed_total > 1e-9 else 0.0,
                "target_switched_total": switched_total,
                "target_lost_total": lost_total,
                "switched_per_1k_frames": (1000.0 * switched_total / frames_total) if frames_total > 0 else 0.0,
                "lost_per_1k_frames": (1000.0 * lost_total / frames_total) if frames_total > 0 else 0.0,
                "flap_ratio": (flap_num / flap_den) if flap_den > 0 else 0.0,
                "target_run_avg_frames": (run_avg_weighted_num / target_runs_total) if target_runs_total > 0 else 0.0,
            }
        )
    return agg_rows


def _pick_baseline(agg_rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    if not agg_rows:
        raise ValueError("No aggregate rows to compare")
    if baseline:
        try:
            b_tracker, b_profile = baseline.split(":", 1)
        except ValueError as exc:
            raise ValueError("--baseline must be tracker:profile, e.g. botsort:balanced") from exc
        for row in agg_rows:
            if row["tracker"] == b_tracker and row["profile"] == b_profile:
                return row
        raise ValueError(f"Baseline not found in results: {baseline}")

    for row in agg_rows:
        if row["tracker"] == "botsort" and row["profile"] == "balanced":
            return row
    return agg_rows[0]


def _enrich_with_deltas(agg_rows: list[dict[str, Any]], baseline_row: dict[str, Any]) -> list[dict[str, Any]]:
    b_fps = _safe_float(baseline_row["avg_processing_fps"], 0.0)
    b_lost = _safe_float(baseline_row["lost_per_1k_frames"], 0.0)
    b_switch = _safe_float(baseline_row["switched_per_1k_frames"], 0.0)
    b_run = _safe_float(baseline_row["target_run_avg_frames"], 0.0)

    out: list[dict[str, Any]] = []
    for row in agg_rows:
        fps = _safe_float(row["avg_processing_fps"], 0.0)
        lost = _safe_float(row["lost_per_1k_frames"], 0.0)
        switch = _safe_float(row["switched_per_1k_frames"], 0.0)
        run_avg = _safe_float(row["target_run_avg_frames"], 0.0)
        row2 = dict(row)
        row2["delta_fps_pct_vs_baseline"] = 0.0 if abs(b_fps) < 1e-12 else (100.0 * (fps - b_fps) / b_fps)
        row2["delta_lost_pct_vs_baseline"] = 0.0 if abs(b_lost) < 1e-12 else (100.0 * (lost - b_lost) / b_lost)
        row2["delta_switch_pct_vs_baseline"] = 0.0 if abs(b_switch) < 1e-12 else (100.0 * (switch - b_switch) / b_switch)
        row2["delta_target_run_avg_pct_vs_baseline"] = 0.0 if abs(b_run) < 1e-12 else (100.0 * (run_avg - b_run) / b_run)
        out.append(row2)
    return out


def _choose_recommended(agg_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    b_lost = _safe_float(baseline["lost_per_1k_frames"], 0.0)
    b_switch = _safe_float(baseline["switched_per_1k_frames"], 0.0)
    b_run = _safe_float(baseline["target_run_avg_frames"], 0.0)

    candidates: list[dict[str, Any]] = []
    for row in agg_rows:
        lost = _safe_float(row["lost_per_1k_frames"], 0.0)
        switch = _safe_float(row["switched_per_1k_frames"], 0.0)
        run_avg = _safe_float(row["target_run_avg_frames"], 0.0)
        pass_lost = True if b_lost <= 1e-12 else (lost <= b_lost * 1.20)
        pass_switch = True if b_switch <= 1e-12 else (switch <= b_switch * 1.20)
        pass_run = True if b_run <= 1e-12 else (run_avg >= b_run * 0.80)
        row2 = dict(row)
        row2["_quality_gate_passed"] = bool(pass_lost and pass_switch and pass_run)
        candidates.append(row2)

    passed = [r for r in candidates if r["_quality_gate_passed"]] or candidates
    passed.sort(key=lambda r: _safe_float(r["avg_processing_fps"], 0.0), reverse=True)
    return passed[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_markdown(path: Path, agg_rows: list[dict[str, Any]], baseline: dict[str, Any], recommended: dict[str, Any], video_count: int) -> None:
    def _fmt(v: Any, d: int = 3) -> str:
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Videos evaluated: {video_count}")
    lines.append("- Metrics focus: single-target lock quality + runtime throughput")
    lines.append(f"- Baseline: `{baseline['tracker']}:{baseline['profile']}`")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| tracker | profile | fps | lost/1k | switched/1k | target_run_avg | flap_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in agg_rows:
        lines.append(
            f"| `{row['tracker']}` | `{row['profile']}` | {_fmt(row['avg_processing_fps'])} | "
            f"{_fmt(row['lost_per_1k_frames'])} | {_fmt(row['switched_per_1k_frames'])} | "
            f"{_fmt(row['target_run_avg_frames'])} | {_fmt(row['flap_ratio'])} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append(
        f"- Recommended config: `{recommended['tracker']}:{recommended['profile']}` "
        f"(fps={_fmt(recommended['avg_processing_fps'])}, lost/1k={_fmt(recommended['lost_per_1k_frames'])})."
    )
    lines.append("- Trade-off policy: prefer highest FPS among configs that do not heavily degrade lock quality.")
    lines.append("- Practical conclusion: ByteTrack is usually the best edge default when quality metrics stay close.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-video benchmark suite for pts selection layer.")
    parser.add_argument("--input-dir", type=str, default="data/input")
    parser.add_argument("--output-dir", type=str, default="data/output/benchmark")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--profiles-file", type=str, default="benchmark_profiles.yaml")
    parser.add_argument("--profiles", nargs="*", default=[], help="Profile names from YAML. Empty = all.")
    parser.add_argument("--trackers", nargs="*", default=["botsort", "bytetrack"])
    parser.add_argument("--video-limit", type=int, default=5, help="Use first N videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap per video.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--demo-profile", type=str, default="balanced")
    parser.add_argument("--demo-videos", type=int, default=1, help="How many first videos to render for demo.")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--baseline", type=str, default="", help="tracker:profile baseline (e.g. botsort:balanced).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise ImportError("Install example dependencies first: pip install -e .[examples]") from exc

    input_dir = _resolve(args.input_dir)
    output_root = _resolve(args.output_dir)
    models_dir = _resolve(args.models_dir)
    profiles_file = _resolve(args.profiles_file)
    trackers = [t.strip().lower() for t in args.trackers if t.strip()]
    for t in trackers:
        if t not in {"botsort", "bytetrack"}:
            raise ValueError(f"Unsupported tracker: {t}")

    profiles = _load_profiles(profiles_path=profiles_file, profile_names=list(args.profiles))
    videos = _collect_videos(input_dir=input_dir, video_limit=int(args.video_limit))
    model_path = _pick_model(models_dir=models_dir, explicit_path=args.model)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    reports_dir = run_dir / "reports"
    events_dir = run_dir / "events"
    demos_dir = run_dir / "demos"
    reports_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    demos_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SUITE] model={model_path}")
    print(f"[SUITE] videos={len(videos)} trackers={trackers} profiles={[p.name for p in profiles]}")
    print(f"[SUITE] output={run_dir}")

    per_video_rows: list[dict[str, Any]] = []
    started_suite = time.perf_counter()

    for tracker in trackers:
        tracker_ref = TRACKER_TO_ULTRA[tracker]
        for profile in profiles:
            print(f"[SUITE] tracker={tracker} profile={profile.name} conf={profile.conf} iou={profile.iou}", flush=True)
            model = YOLO(str(model_path))
            selector = PrimaryTargetSelection()

            for idx, video in enumerate(videos):
                demo_out = None
                if profile.name == args.demo_profile and idx < max(0, int(args.demo_videos)):
                    demo_out = demos_dir / f"{video.stem}__{tracker}__{profile.name}.mp4"
                event_path = events_dir / f"{video.stem}__{tracker}__{profile.name}.jsonl"
                metrics = _process_video(
                    model=model,
                    selector=selector,
                    tracker_name=tracker,
                    tracker_ref=tracker_ref,
                    profile=profile,
                    video_path=video,
                    events_path=event_path,
                    demo_output_path=demo_out,
                    max_frames=int(args.max_frames),
                    progress_every=max(0, int(args.progress_every)),
                    device=str(args.device),
                )
                row = {
                    "video": video.name,
                    "tracker": tracker,
                    "profile": profile.name,
                    "conf": profile.conf,
                    "iou": profile.iou,
                }
                row.update(metrics)
                per_video_rows.append(row)
                print(
                    f"[DONE] {video.name} | {tracker}:{profile.name} "
                    f"fps={row['avg_processing_fps']:.2f} lost/1k={row['lost_per_1k_frames']:.2f}",
                    flush=True,
                )

    agg_rows = _aggregate(per_video_rows)
    baseline = _pick_baseline(agg_rows, args.baseline)
    agg_rows_delta = _enrich_with_deltas(agg_rows, baseline_row=baseline)
    recommended = _choose_recommended(agg_rows_delta, baseline=baseline)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "input_dir": str(input_dir),
        "model_path": str(model_path),
        "device": str(args.device),
        "videos_count": len(videos),
        "trackers": trackers,
        "profiles": [p.__dict__ for p in profiles],
        "baseline": baseline,
        "recommended": recommended,
        "per_video": per_video_rows,
        "aggregate": agg_rows_delta,
        "elapsed_sec_total": max(time.perf_counter() - started_suite, 1e-6),
    }

    summary_json = reports_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_csv(
        reports_dir / "benchmark_per_video.csv",
        rows=per_video_rows,
        fieldnames=[
            "video",
            "tracker",
            "profile",
            "conf",
            "iou",
            "frames",
            "elapsed_sec",
            "avg_processing_fps",
            "target_switched",
            "target_lost",
            "switched_per_1k_frames",
            "lost_per_1k_frames",
            "target_runs_count",
            "target_run_avg_frames",
            "flap_ratio",
        ],
    )
    _write_csv(
        reports_dir / "benchmark_aggregate.csv",
        rows=agg_rows_delta,
        fieldnames=[
            "tracker",
            "profile",
            "videos_count",
            "frames_total",
            "elapsed_sec_total",
            "avg_processing_fps",
            "target_switched_total",
            "target_lost_total",
            "switched_per_1k_frames",
            "lost_per_1k_frames",
            "flap_ratio",
            "target_run_avg_frames",
            "delta_fps_pct_vs_baseline",
            "delta_lost_pct_vs_baseline",
            "delta_switch_pct_vs_baseline",
            "delta_target_run_avg_pct_vs_baseline",
        ],
    )
    _write_markdown(
        path=reports_dir / "benchmark_report.md",
        agg_rows=agg_rows_delta,
        baseline=baseline,
        recommended=recommended,
        video_count=len(videos),
    )

    print(f"[SUITE] JSON: {summary_json}")
    print(f"[SUITE] CSV per-video: {reports_dir / 'benchmark_per_video.csv'}")
    print(f"[SUITE] CSV aggregate: {reports_dir / 'benchmark_aggregate.csv'}")
    print(f"[SUITE] Markdown: {reports_dir / 'benchmark_report.md'}")
    print(f"[SUITE] Recommended: {recommended['tracker']}:{recommended['profile']}")


if __name__ == "__main__":
    main()
