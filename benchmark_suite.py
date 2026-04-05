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
        raw = Path(explicit_path).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((models_dir / raw).resolve())
            candidates.append(_resolve(explicit_path))

        seen: set[Path] = set()
        unique_candidates: list[Path] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        for path in unique_candidates:
            if path.exists() and path.is_file():
                return path

        raise FileNotFoundError(
            "Model file not found. Tried: "
            + ", ".join(str(path) for path in unique_candidates)
        )
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
    policy_name: str,
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
                policy_name=policy_name,
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
    events = summarize_event_rows(load_event_rows(events_path), frames_total=frame_idx)
    switched = _safe_int(events.get("switched"), 0)
    lost = _safe_int(events.get("lost"), 0)
    flap_checks = _safe_int(events.get("flap_checks_total"), 0)
    flap_le_5 = _safe_int(events.get("flap_le_5_frames"), 0)
    video_minutes = (frame_idx / max(source_fps, 1e-6)) / 60.0

    return {
        "frames": frame_idx,
        "elapsed_sec": elapsed,
        "avg_processing_fps": (frame_idx / elapsed) if frame_idx > 0 else 0.0,
        "avg_processing_ms": (1000.0 * elapsed / frame_idx) if frame_idx > 0 else 0.0,
        "target_switched": switched,
        "target_lost": lost,
        "lost_target_count": lost,
        "switched_per_1k_frames": (1000.0 * switched / frame_idx) if frame_idx > 0 else 0.0,
        "lost_per_1k_frames": (1000.0 * lost / frame_idx) if frame_idx > 0 else 0.0,
        "switch_rate_per_minute": (switched / video_minutes) if video_minutes > 1e-9 else 0.0,
        "target_acquisition_delay_frames": events.get("target_acquisition_delay_frames"),
        "primary_target_presence_ratio": _safe_float(events.get("primary_target_presence_ratio"), 0.0),
        "primary_target_stability_ratio": _safe_float(events.get("primary_target_stability_ratio"), 0.0),
        "mean_lock_duration_frames": _safe_float(events.get("mean_lock_duration_frames"), 0.0),
        "max_lock_duration_frames": _safe_int(events.get("max_lock_duration_frames"), 0),
        "bonus_dominated_ratio": (
            _safe_int(events.get("bonus_dominated_count"), 0) / max(_safe_int(events.get("score_rows_count"), 0), 1)
        ),
        "policy_dominated_ratio": (
            _safe_int(events.get("policy_dominated_count"), 0) / max(_safe_int(events.get("score_rows_count"), 0), 1)
        ),
        "external_dominated_ratio": (
            _safe_int(events.get("external_dominated_count"), 0) / max(_safe_int(events.get("score_rows_count"), 0), 1)
        ),
        "policy_clip_ratio": _safe_float(events.get("policy_clip_ratio"), 0.0),
        "external_clip_ratio": _safe_float(events.get("external_clip_ratio"), 0.0),
        "total_bonus_clip_ratio": _safe_float(events.get("total_bonus_clip_ratio"), 0.0),
        "avg_policy_clip_abs": _safe_float(events.get("avg_policy_clip_abs"), 0.0),
        "avg_external_clip_abs": _safe_float(events.get("avg_external_clip_abs"), 0.0),
        "avg_total_bonus_clip_abs": _safe_float(events.get("avg_total_bonus_clip_abs"), 0.0),
        "target_runs_count": _safe_int(events.get("target_runs_count"), 0),
        "target_run_avg_frames": _safe_float(events.get("target_run_avg_frames"), 0.0),
        "flap_ratio": (flap_le_5 / flap_checks) if flap_checks > 0 else 0.0,
        "events": events,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["tracker"]), str(row["profile"]), str(row.get("policy", "single_best")))].append(row)

    agg_rows: list[dict[str, Any]] = []
    for (tracker, profile, policy), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        frames_total = sum(_safe_int(i["frames"]) for i in items)
        elapsed_total = sum(_safe_float(i["elapsed_sec"]) for i in items)
        switched_total = sum(_safe_int(i["target_switched"]) for i in items)
        lost_total = sum(_safe_int(i["target_lost"]) for i in items)
        flap_num = sum(_safe_float(i["flap_ratio"]) * max(_safe_int(i["events"]["flap_checks_total"]), 1) for i in items)
        flap_den = sum(max(_safe_int(i["events"]["flap_checks_total"]), 1) for i in items)
        target_runs_total = sum(_safe_int(i["target_runs_count"]) for i in items)
        run_avg_weighted_num = sum(_safe_float(i["target_run_avg_frames"]) * _safe_int(i["target_runs_count"]) for i in items)
        presence_ratio_avg = sum(_safe_float(i.get("primary_target_presence_ratio"), 0.0) for i in items) / max(len(items), 1)
        stability_ratio_avg = sum(_safe_float(i.get("primary_target_stability_ratio"), 0.0) for i in items) / max(len(items), 1)
        acq_delay_values = [
            _safe_int(i.get("target_acquisition_delay_frames"), -1)
            for i in items
            if _safe_int(i.get("target_acquisition_delay_frames"), -1) >= 0
        ]
        mean_lock_weighted_num = sum(_safe_float(i.get("mean_lock_duration_frames"), 0.0) * _safe_int(i["target_runs_count"]) for i in items)
        max_lock = max((_safe_int(i.get("max_lock_duration_frames"), 0) for i in items), default=0)
        bonus_dominated_total = sum(_safe_int(i["events"].get("bonus_dominated_count"), 0) for i in items)
        policy_dominated_total = sum(_safe_int(i["events"].get("policy_dominated_count"), 0) for i in items)
        external_dominated_total = sum(_safe_int(i["events"].get("external_dominated_count"), 0) for i in items)
        policy_clip_total = sum(_safe_int(i["events"].get("policy_clip_count"), 0) for i in items)
        external_clip_total = sum(_safe_int(i["events"].get("external_clip_count"), 0) for i in items)
        total_bonus_clip_total = sum(_safe_int(i["events"].get("total_bonus_clip_count"), 0) for i in items)
        policy_clip_abs_sum = sum(_safe_float(i["events"].get("avg_policy_clip_abs"), 0.0) * max(_safe_int(i["events"].get("score_rows_count"), 0), 1) for i in items)
        external_clip_abs_sum = sum(_safe_float(i["events"].get("avg_external_clip_abs"), 0.0) * max(_safe_int(i["events"].get("score_rows_count"), 0), 1) for i in items)
        total_bonus_clip_abs_sum = sum(_safe_float(i["events"].get("avg_total_bonus_clip_abs"), 0.0) * max(_safe_int(i["events"].get("score_rows_count"), 0), 1) for i in items)
        score_rows_total = sum(max(0, _safe_int(i["events"].get("score_rows_count"), 0)) for i in items)

        agg_rows.append(
            {
                "tracker": tracker,
                "profile": profile,
                "policy": policy,
                "videos_count": len(items),
                "frames_total": frames_total,
                "elapsed_sec_total": elapsed_total,
                "avg_processing_fps": (frames_total / elapsed_total) if elapsed_total > 1e-9 else 0.0,
                "avg_processing_ms": (1000.0 * elapsed_total / frames_total) if frames_total > 0 else 0.0,
                "target_switched_total": switched_total,
                "target_lost_total": lost_total,
                "lost_target_count_total": lost_total,
                "switched_per_1k_frames": (1000.0 * switched_total / frames_total) if frames_total > 0 else 0.0,
                "lost_per_1k_frames": (1000.0 * lost_total / frames_total) if frames_total > 0 else 0.0,
                "switch_rate_per_minute": (
                    sum(_safe_float(i.get("switch_rate_per_minute"), 0.0) for i in items) / max(len(items), 1)
                ),
                "target_acquisition_delay_frames_avg": (
                    sum(acq_delay_values) / len(acq_delay_values) if acq_delay_values else 0.0
                ),
                "primary_target_presence_ratio": presence_ratio_avg,
                "primary_target_stability_ratio": stability_ratio_avg,
                "mean_lock_duration_frames": (
                    mean_lock_weighted_num / target_runs_total if target_runs_total > 0 else 0.0
                ),
                "max_lock_duration_frames": max_lock,
                "bonus_dominated_ratio": (bonus_dominated_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "policy_dominated_ratio": (policy_dominated_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "external_dominated_ratio": (external_dominated_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "policy_clip_ratio": (policy_clip_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "external_clip_ratio": (external_clip_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "total_bonus_clip_ratio": (total_bonus_clip_total / score_rows_total) if score_rows_total > 0 else 0.0,
                "avg_policy_clip_abs": (policy_clip_abs_sum / score_rows_total) if score_rows_total > 0 else 0.0,
                "avg_external_clip_abs": (external_clip_abs_sum / score_rows_total) if score_rows_total > 0 else 0.0,
                "avg_total_bonus_clip_abs": (total_bonus_clip_abs_sum / score_rows_total) if score_rows_total > 0 else 0.0,
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
            b_tracker, b_profile, b_policy = baseline.split(":", 2)
        except ValueError as exc:
            raise ValueError("--baseline must be tracker:profile:policy, e.g. botsort:balanced:single_best") from exc
        for row in agg_rows:
            if row["tracker"] == b_tracker and row["profile"] == b_profile and row["policy"] == b_policy:
                return row
        raise ValueError(f"Baseline not found in results: {baseline}")

    for row in agg_rows:
        if row["tracker"] == "botsort" and row["profile"] == "balanced" and row["policy"] == "single_best":
            return row
    return agg_rows[0]


def _enrich_with_deltas(agg_rows: list[dict[str, Any]], baseline_row: dict[str, Any]) -> list[dict[str, Any]]:
    b_fps = _safe_float(baseline_row["avg_processing_fps"], 0.0)
    b_lost = _safe_float(baseline_row["lost_per_1k_frames"], 0.0)
    b_switch = _safe_float(baseline_row["switched_per_1k_frames"], 0.0)
    b_run = _safe_float(baseline_row["target_run_avg_frames"], 0.0)
    b_presence = _safe_float(baseline_row.get("primary_target_presence_ratio"), 0.0)
    b_stability = _safe_float(baseline_row.get("primary_target_stability_ratio"), 0.0)

    out: list[dict[str, Any]] = []
    for row in agg_rows:
        fps = _safe_float(row["avg_processing_fps"], 0.0)
        lost = _safe_float(row["lost_per_1k_frames"], 0.0)
        switch = _safe_float(row["switched_per_1k_frames"], 0.0)
        run_avg = _safe_float(row["target_run_avg_frames"], 0.0)
        presence = _safe_float(row.get("primary_target_presence_ratio"), 0.0)
        stability = _safe_float(row.get("primary_target_stability_ratio"), 0.0)
        row2 = dict(row)
        row2["delta_fps_pct_vs_baseline"] = 0.0 if abs(b_fps) < 1e-12 else (100.0 * (fps - b_fps) / b_fps)
        row2["delta_lost_pct_vs_baseline"] = 0.0 if abs(b_lost) < 1e-12 else (100.0 * (lost - b_lost) / b_lost)
        row2["delta_switch_pct_vs_baseline"] = 0.0 if abs(b_switch) < 1e-12 else (100.0 * (switch - b_switch) / b_switch)
        row2["delta_target_run_avg_pct_vs_baseline"] = 0.0 if abs(b_run) < 1e-12 else (100.0 * (run_avg - b_run) / b_run)
        row2["delta_presence_pct_vs_baseline"] = 0.0 if abs(b_presence) < 1e-12 else (100.0 * (presence - b_presence) / b_presence)
        row2["delta_stability_pct_vs_baseline"] = 0.0 if abs(b_stability) < 1e-12 else (100.0 * (stability - b_stability) / b_stability)
        out.append(row2)
    return out


def _choose_recommended(agg_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    b_lost = _safe_float(baseline["lost_per_1k_frames"], 0.0)
    b_switch = _safe_float(baseline["switched_per_1k_frames"], 0.0)
    b_run = _safe_float(baseline["target_run_avg_frames"], 0.0)
    b_presence = _safe_float(baseline.get("primary_target_presence_ratio"), 0.0)
    b_stability = _safe_float(baseline.get("primary_target_stability_ratio"), 0.0)

    candidates: list[dict[str, Any]] = []
    for row in agg_rows:
        lost = _safe_float(row["lost_per_1k_frames"], 0.0)
        switch = _safe_float(row["switched_per_1k_frames"], 0.0)
        run_avg = _safe_float(row["target_run_avg_frames"], 0.0)
        presence = _safe_float(row.get("primary_target_presence_ratio"), 0.0)
        stability = _safe_float(row.get("primary_target_stability_ratio"), 0.0)
        bonus_dom = _safe_float(row.get("bonus_dominated_ratio"), 0.0)
        external_dom = _safe_float(row.get("external_dominated_ratio"), 0.0)
        policy_clip_ratio = _safe_float(row.get("policy_clip_ratio"), 0.0)
        external_clip_ratio = _safe_float(row.get("external_clip_ratio"), 0.0)
        pass_lost = True if b_lost <= 1e-12 else (lost <= b_lost * 1.20)
        pass_switch = True if b_switch <= 1e-12 else (switch <= b_switch * 1.20)
        pass_run = True if b_run <= 1e-12 else (run_avg >= b_run * 0.80)
        pass_presence = True if b_presence <= 1e-12 else (presence >= b_presence * 0.80)
        pass_stability = True if b_stability <= 1e-12 else (stability >= b_stability * 0.90)
        pass_bonus = bonus_dom <= 0.40
        pass_external = external_dom <= 0.50
        pass_clip = (policy_clip_ratio <= 0.50) and (external_clip_ratio <= 0.50)
        row2 = dict(row)
        row2["_quality_gate_passed"] = bool(
            pass_lost
            and pass_switch
            and pass_run
            and pass_presence
            and pass_stability
            and pass_bonus
            and pass_external
            and pass_clip
        )
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
    lines.append(f"- Baseline: `{baseline['tracker']}:{baseline['profile']}:{baseline.get('policy', 'single_best')}`")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| tracker | profile | policy | fps | lost/1k | switched/1k | presence | stability | lock_avg | bonus_dom | ext_dom | pol_clip | ext_clip | flap_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in agg_rows:
        lines.append(
            f"| `{row['tracker']}` | `{row['profile']}` | `{row.get('policy', 'single_best')}` | {_fmt(row['avg_processing_fps'])} | "
            f"{_fmt(row['lost_per_1k_frames'])} | {_fmt(row['switched_per_1k_frames'])} | "
            f"{_fmt(row.get('primary_target_presence_ratio', 0.0))} | {_fmt(row.get('primary_target_stability_ratio', 0.0))} | "
            f"{_fmt(row['target_run_avg_frames'])} | {_fmt(row.get('bonus_dominated_ratio', 0.0))} | "
            f"{_fmt(row.get('external_dominated_ratio', 0.0))} | {_fmt(row.get('policy_clip_ratio', 0.0))} | "
            f"{_fmt(row.get('external_clip_ratio', 0.0))} | {_fmt(row['flap_ratio'])} |"
        )
    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append("- `primary_target_presence_ratio >= 0.30`")
    lines.append("- `primary_target_stability_ratio >= 0.90`")
    lines.append("- `bonus_dominated_ratio <= 0.40`")
    lines.append("- `external_dominated_ratio <= 0.50`")
    lines.append("")
    lines.append("Values outside these ranges indicate overly aggressive policy/external influence or unstable lock behavior.")
    lines.append("")
    lines.append("## Policy Conclusions")
    lines.append("")
    policy_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agg_rows:
        policy_groups[str(row.get("policy", "single_best"))].append(row)

    policy_summary: dict[str, dict[str, float]] = {}
    for policy, items in policy_groups.items():
        policy_summary[policy] = {
            "fps": sum(_safe_float(i.get("avg_processing_fps"), 0.0) for i in items) / max(len(items), 1),
            "switch": sum(_safe_float(i.get("switched_per_1k_frames"), 0.0) for i in items) / max(len(items), 1),
            "lost": sum(_safe_float(i.get("lost_per_1k_frames"), 0.0) for i in items) / max(len(items), 1),
            "presence": sum(_safe_float(i.get("primary_target_presence_ratio"), 0.0) for i in items) / max(len(items), 1),
            "stability": sum(_safe_float(i.get("primary_target_stability_ratio"), 0.0) for i in items) / max(len(items), 1),
            "lock_avg": sum(_safe_float(i.get("target_run_avg_frames"), 0.0) for i in items) / max(len(items), 1),
        }

    if "single_best" in policy_summary and "center_biased" in policy_summary:
        sb = policy_summary["single_best"]
        cb = policy_summary["center_biased"]
        if cb["switch"] > sb["switch"] + 0.05:
            lines.append(
                f"- `center_biased` increases switch rate vs `single_best` "
                f"({cb['switch']:.3f} vs {sb['switch']:.3f} per 1k frames)."
            )
        else:
            lines.append("- `center_biased` does not materially increase switch rate vs `single_best` on this run.")

    if "single_best" in policy_summary and "stable_target" in policy_summary:
        sb = policy_summary["single_best"]
        st = policy_summary["stable_target"]
        if st["lock_avg"] >= sb["lock_avg"] * 1.02 and st["stability"] >= sb["stability"] - 1e-6:
            lines.append("- `stable_target` improves lock duration without stability degradation.")
        elif st["stability"] < sb["stability"] - 0.01:
            lines.append("- `stable_target` reduces stability on this run; keep it as optional policy.")
        else:
            lines.append("- `stable_target` is close to `single_best`; choose based on scene behavior.")

    tracker_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agg_rows:
        tracker_groups[str(row.get("tracker", "unknown"))].append(row)
    tracker_fps = {
        tracker: sum(_safe_float(i.get("avg_processing_fps"), 0.0) for i in items) / max(len(items), 1)
        for tracker, items in tracker_groups.items()
    }
    if tracker_fps:
        fastest_tracker = max(tracker_fps.items(), key=lambda kv: kv[1])[0]
        lines.append(f"- Fastest tracker family on this run: `{fastest_tracker}`.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(
        f"- Recommended config: `{recommended['tracker']}:{recommended['profile']}:{recommended.get('policy', 'single_best')}` "
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
    parser.add_argument("--selection-config", type=str, default="", help="Optional pts selection config YAML path.")
    parser.add_argument("--profiles-file", type=str, default="benchmark_profiles.yaml")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=["fast", "balanced"],
        help="Profile names from YAML. Default is reduced grid: fast balanced.",
    )
    parser.add_argument("--trackers", nargs="*", default=["botsort", "bytetrack"])
    parser.add_argument(
        "--policies",
        nargs="*",
        default=["single_best"],
        help="Selection policies to benchmark (e.g. single_best center_biased stable_target).",
    )
    parser.add_argument("--video-limit", type=int, default=5, help="Use first N videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap per video.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--demo-profile", type=str, default="balanced")
    parser.add_argument("--demo-videos", type=int, default=1, help="How many first videos to render for demo.")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="tracker:profile:policy baseline (e.g. botsort:balanced:single_best).",
    )
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
    selection_config = str(_resolve(args.selection_config)) if str(args.selection_config).strip() else None
    policies = [p.strip() for p in args.policies if str(p).strip()]
    if not policies:
        policies = ["single_best"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    reports_dir = run_dir / "reports"
    events_dir = run_dir / "events"
    demos_dir = run_dir / "demos"
    reports_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    demos_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SUITE] model={model_path}")
    print(
        f"[SUITE] videos={len(videos)} trackers={trackers} profiles={[p.name for p in profiles]} policies={policies}"
    )
    print(f"[SUITE] output={run_dir}")

    per_video_rows: list[dict[str, Any]] = []
    started_suite = time.perf_counter()

    for tracker in trackers:
        tracker_ref = TRACKER_TO_ULTRA[tracker]
        for profile in profiles:
            for policy_name in policies:
                print(
                    f"[SUITE] tracker={tracker} profile={profile.name} policy={policy_name} "
                    f"conf={profile.conf} iou={profile.iou}",
                    flush=True,
                )
                model = YOLO(str(model_path))
                selector = PrimaryTargetSelection(config_path=selection_config)

                for idx, video in enumerate(videos):
                    demo_out = None
                    if (
                        policy_name == "single_best"
                        and profile.name == args.demo_profile
                        and idx < max(0, int(args.demo_videos))
                    ):
                        demo_out = demos_dir / f"{video.stem}__{tracker}__{profile.name}__{policy_name}.mp4"
                    event_path = events_dir / f"{video.stem}__{tracker}__{profile.name}__{policy_name}.jsonl"
                    metrics = _process_video(
                        model=model,
                        selector=selector,
                        tracker_name=tracker,
                        tracker_ref=tracker_ref,
                        profile=profile,
                        policy_name=policy_name,
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
                        "policy": policy_name,
                        "conf": profile.conf,
                        "iou": profile.iou,
                    }
                    row.update(metrics)
                    per_video_rows.append(row)
                    print(
                        f"[DONE] {video.name} | {tracker}:{profile.name}:{policy_name} "
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
        "policies": policies,
        "selection_config": selection_config,
        "profiles": [p.__dict__ for p in profiles],
        "baseline": baseline,
        "recommended": recommended,
        "per_video": per_video_rows,
        "aggregate": agg_rows_delta,
        "elapsed_sec_total": max(time.perf_counter() - started_suite, 1e-6),
        "interpretation_thresholds": {
            "min_presence_ratio": 0.30,
            "min_stability_ratio": 0.90,
            "max_bonus_dominated_ratio": 0.40,
            "max_external_dominated_ratio": 0.50,
            "max_policy_clip_ratio": 0.50,
            "max_external_clip_ratio": 0.50,
        },
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
            "policy",
            "conf",
            "iou",
            "frames",
            "elapsed_sec",
            "avg_processing_fps",
            "avg_processing_ms",
            "target_switched",
            "target_lost",
            "lost_target_count",
            "switched_per_1k_frames",
            "lost_per_1k_frames",
            "switch_rate_per_minute",
            "target_acquisition_delay_frames",
            "primary_target_presence_ratio",
            "primary_target_stability_ratio",
            "mean_lock_duration_frames",
            "max_lock_duration_frames",
            "bonus_dominated_ratio",
            "policy_dominated_ratio",
            "external_dominated_ratio",
            "policy_clip_ratio",
            "external_clip_ratio",
            "total_bonus_clip_ratio",
            "avg_policy_clip_abs",
            "avg_external_clip_abs",
            "avg_total_bonus_clip_abs",
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
            "policy",
            "videos_count",
            "frames_total",
            "elapsed_sec_total",
            "avg_processing_fps",
            "avg_processing_ms",
            "target_switched_total",
            "target_lost_total",
            "lost_target_count_total",
            "switched_per_1k_frames",
            "lost_per_1k_frames",
            "switch_rate_per_minute",
            "target_acquisition_delay_frames_avg",
            "primary_target_presence_ratio",
            "primary_target_stability_ratio",
            "mean_lock_duration_frames",
            "max_lock_duration_frames",
            "bonus_dominated_ratio",
            "policy_dominated_ratio",
            "external_dominated_ratio",
            "policy_clip_ratio",
            "external_clip_ratio",
            "total_bonus_clip_ratio",
            "avg_policy_clip_abs",
            "avg_external_clip_abs",
            "avg_total_bonus_clip_abs",
            "flap_ratio",
            "target_run_avg_frames",
            "delta_fps_pct_vs_baseline",
            "delta_lost_pct_vs_baseline",
            "delta_switch_pct_vs_baseline",
            "delta_target_run_avg_pct_vs_baseline",
            "delta_presence_pct_vs_baseline",
            "delta_stability_pct_vs_baseline",
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
    print(f"[SUITE] Recommended: {recommended['tracker']}:{recommended['profile']}:{recommended.get('policy', 'single_best')}")


if __name__ == "__main__":
    main()
