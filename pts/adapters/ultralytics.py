from __future__ import annotations

from typing import Any

from ..types import SelectionTrack


def resolve_class_name(
    prediction: Any,
    cls_id: int,
    class_names: dict[int, str] | None = None,
) -> str:
    if isinstance(class_names, dict) and cls_id in class_names:
        return str(class_names[cls_id])

    names = getattr(prediction, "names", {})
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def prediction_to_tracks(
    prediction: Any,
    class_names: dict[int, str] | None = None,
    visible: bool = True,
) -> list[SelectionTrack]:
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    ids = getattr(boxes, "id", None)
    if ids is None:
        return []

    tracks: list[SelectionTrack] = []
    for i, track_id_val in enumerate(ids.tolist()):
        if track_id_val is None:
            continue

        bbox = tuple(float(v) for v in boxes.xyxy[i].tolist())
        conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1

        tracks.append(
            SelectionTrack(
                track_id=int(track_id_val),
                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=resolve_class_name(prediction, cls_id, class_names=class_names),
                visible=bool(visible),
            )
        )
    return tracks


def frame_size_from_prediction(
    prediction: Any,
    fallback_frame: Any | None = None,
) -> tuple[int, int]:
    orig_shape = getattr(prediction, "orig_shape", None)
    if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
        h, w = int(orig_shape[0]), int(orig_shape[1])
        return (w, h)

    if fallback_frame is not None and hasattr(fallback_frame, "shape") and len(fallback_frame.shape) >= 2:
        h, w = int(fallback_frame.shape[0]), int(fallback_frame.shape[1])
        return (w, h)

    raise ValueError("Unable to resolve frame_size from prediction. Pass fallback_frame or explicit frame size.")


def prediction_to_selection_input(
    prediction: Any,
    class_names: dict[int, str] | None = None,
    visible: bool = True,
    fallback_frame: Any | None = None,
) -> tuple[list[SelectionTrack], tuple[int, int]]:
    tracks = prediction_to_tracks(prediction, class_names=class_names, visible=visible)
    frame_size = frame_size_from_prediction(prediction, fallback_frame=fallback_frame)
    return tracks, frame_size


def reset_ultralytics_trackers(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    if hasattr(predictor, "trackers"):
        try:
            delattr(predictor, "trackers")
        except Exception:
            predictor.trackers = []
