from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import SelectionOutput


def draw_selection_overlay(
    frame: Any,
    output: SelectionOutput,
    *,
    show_rejected: bool = True,
    show_status: bool = True,
) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for draw_selection_overlay. Install package extras: pip install 'primary-target-selection[adapters]'"
        ) from exc

    out = frame.copy()
    h, w = out.shape[:2]
    primary_id = output.primary_track_id

    for track in output.candidates:
        if not track.visible:
            continue
        if not show_rejected and not track.accepted:
            continue
        x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        is_primary = int(track.track_id) == int(primary_id) if primary_id is not None else False
        if is_primary:
            color = (0, 80, 255)
        elif track.accepted:
            color = (60, 200, 50)
        else:
            color = (120, 120, 120)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_primary else 2, lineType=cv2.LINE_AA)

        score = "n/a" if track.score is None else f"{track.score:.2f}"
        if track.accepted:
            label = f"{track.class_name} id={track.track_id} score={score}"
        else:
            reason = track.reject_reason or "rejected"
            label = f"{track.class_name} id={track.track_id} {reason}"

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

    if show_status:
        status = (
            f"state={output.selection_state} primary={primary_id} "
            f"reason={output.selection_reason}"
        )
        cv2.putText(
            out,
            status,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def selection_output_to_signal(output: SelectionOutput) -> dict[str, Any]:
    primary_track_id = output.primary_track_id
    primary_candidate = next(
        (candidate for candidate in output.candidates if candidate.track_id == primary_track_id),
        None,
    )
    payload: dict[str, Any] = {
        "frame_index": output.frame_index,
        "timestamp_s": output.timestamp_s,
        "selection_state": output.selection_state,
        "selection_reason": output.selection_reason,
        "tracks_count": len(output.candidates),
        "events": [asdict(evt) for evt in output.events],
        "primary": None,
        "effective_policy_name": output.effective_policy_name,
        "auto_mode": output.auto_mode,
        "auto_mode_reason": output.auto_mode_reason,
    }
    if primary_candidate is not None:
        payload["primary"] = {
            "track_id": primary_candidate.track_id,
            "class_name": primary_candidate.class_name,
            "confidence": primary_candidate.confidence,
            "score": output.primary_score,
            "bbox_xyxy": primary_candidate.bbox_xyxy,
        }
    return payload
