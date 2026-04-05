from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from pts import PrimaryTargetSelection, SelectionTrack


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MODEL_EXTENSIONS = (".onnx", ".pt", ".engine")
TRACKER_TO_ULTRA = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _pick_model(models_dir: Path, explicit_path: str = "") -> Path:
    if explicit_path:
        raw = Path(explicit_path).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            # Priority 1: treat --model as file name/path inside --models-dir.
            candidates.append((models_dir / raw).resolve())
            # Priority 2: treat --model as path relative to project root.
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


def _collect_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_dir}")
    return videos


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
    ids_list = ids.tolist()
    for i, track_id_val in enumerate(ids_list):
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

        score = "n/a" if track.score is None else f"{track.score:.2f}"
        label = f"{track.class_name} id={track.track_id} score={score}"
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

    status = f"state={output.selection_state} primary={primary_id} reason={output.selection_reason}"
    cv2.putText(out, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return out


def _signal_from_output(out: Any) -> dict[str, Any]:
    primary_track_id = out.primary_track_id
    primary_candidate = next(
        (candidate for candidate in out.candidates if candidate.track_id == primary_track_id),
        None,
    )
    payload: dict[str, Any] = {
        "frame_index": out.frame_index,
        "timestamp_s": out.timestamp_s,
        "selection_state": out.selection_state,
        "selection_reason": out.selection_reason,
        "tracks_count": len(out.candidates),
        "events": [asdict(evt) for evt in out.events],
        "primary": None,
    }
    if primary_candidate is not None:
        payload["primary"] = {
            "track_id": primary_candidate.track_id,
            "class_name": primary_candidate.class_name,
            "confidence": primary_candidate.confidence,
            "score": out.primary_score,
            "bbox_xyxy": primary_candidate.bbox_xyxy,
        }
    return payload


def _process_video(
    model: Any,
    selector: PrimaryTargetSelection,
    tracker_ref: str,
    video_path: Path,
    output_video: Path,
    output_signals: Path | None,
    conf: float,
    iou: float,
    device: str,
    render: bool,
    max_frames: int | None,
    policy_name: str | None,
    external_signals: dict[str, Any] | None,
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

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))

    selector.reset()
    _reset_ultralytics_trackers(model)

    if output_signals is not None:
        output_signals.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_signals.open("w", encoding="utf-8") if output_signals is not None else None

    frame = first_frame
    frame_idx = 0
    started = time.perf_counter()
    last_primary_id: int | None = None

    try:
        while True:
            ts = frame_idx / max(source_fps, 1e-6)
            pred = model.track(
                source=frame,
                conf=float(conf),
                iou=float(iou),
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
                external_signals=external_signals,
            )

            current_primary = out.primary_track_id
            if current_primary != last_primary_id:
                print(
                    f"[SIGNAL] {video_path.name}: frame={frame_idx} primary={current_primary} reason={out.selection_reason}",
                    flush=True,
                )
                last_primary_id = current_primary

            if jsonl_file is not None:
                jsonl_file.write(json.dumps(_signal_from_output(out), ensure_ascii=False) + "\n")

            writer.write(_draw_selection_overlay(frame, out) if render else frame)
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                break
            frame = next_frame
    finally:
        cap.release()
        writer.release()
        if jsonl_file is not None:
            jsonl_file.close()

    elapsed = max(time.perf_counter() - started, 1e-6)
    return {
        "video": str(video_path),
        "output_video": str(output_video),
        "output_signals": str(output_signals) if output_signals is not None else None,
        "frames": frame_idx,
        "elapsed_sec": elapsed,
        "processing_fps": frame_idx / elapsed if frame_idx > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External tracker -> pts selection integration example")
    parser.add_argument("--input-dir", type=str, default="data/input")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument("--tracker-config", type=str, default="", help="Optional tracker YAML path override")
    parser.add_argument("--selection-config", type=str, default="", help="Optional pts selection config YAML path")
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Optional selection policy override (single_best, center_biased, stable_target, largest_target, class_priority).",
    )
    parser.add_argument(
        "--external-signals-json",
        type=str,
        default="",
        help="Optional path to JSON with external signals (preferred_track_id, track_score_bias, class_score_bias, external_hint_score).",
    )
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-signals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise ImportError("Install example dependencies first: pip install -e .[examples]") from exc

    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    models_dir = _resolve(args.models_dir)
    model_path = _pick_model(models_dir=models_dir, explicit_path=args.model)
    videos = _collect_videos(input_dir)

    tracker_ref = str(_resolve(args.tracker_config)) if args.tracker_config else TRACKER_TO_ULTRA[str(args.tracker)]
    selection_cfg = str(_resolve(args.selection_config)) if args.selection_config else None
    policy_name = str(args.policy).strip() or None
    external_signals: dict[str, Any] | None = None
    if str(args.external_signals_json).strip():
        raw_external = _resolve(str(args.external_signals_json))
        with raw_external.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("--external-signals-json must point to JSON object")
        external_signals = payload

    model = YOLO(str(model_path))
    selector = PrimaryTargetSelection(config_path=selection_cfg)
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None
    summary: list[dict[str, Any]] = []

    print(f"[INIT] model={model_path}")
    print(
        f"[INIT] tracker={args.tracker} tracker_ref={tracker_ref} policy={policy_name or 'config_default'} "
        f"device={args.device} videos={len(videos)}"
    )

    for video in videos:
        output_video = output_dir / f"{video.stem}_pred.mp4"
        output_signals = None if args.no_signals else output_dir / f"{video.stem}_signals.jsonl"
        stats = _process_video(
            model=model,
            selector=selector,
            tracker_ref=tracker_ref,
            video_path=video,
            output_video=output_video,
            output_signals=output_signals,
            conf=float(args.conf),
            iou=float(args.iou),
            device=str(args.device),
            render=not bool(args.no_render),
            max_frames=max_frames,
            policy_name=policy_name,
            external_signals=external_signals,
        )
        summary.append(stats)
        print(
            f"[DONE] {video.name}: frames={stats['frames']} fps={stats['processing_fps']:.2f} -> {output_video.name}",
            flush=True,
        )

    summary_path = output_dir / "run_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": summary}, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
