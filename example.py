import cv2
import numpy as np
from pts import PrimaryTargetSelection, SelectionTrack

# import onnxruntime as ort
from ultralytics import YOLO
import os
from typing import Any


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

        is_primary = (
            int(track.track_id) == int(primary_id) if primary_id is not None else False
        )
        color = (0, 80, 255) if is_primary else (60, 200, 50)
        cv2.rectangle(
            out, (x1, y1), (x2, y2), color, 3 if is_primary else 2, lineType=cv2.LINE_AA
        )

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


vid = cv2.VideoCapture("data/input/1.mp4")
w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = YOLO(os.path.join("models/model26n_2_7394f.onnx"))
selector = PrimaryTargetSelection()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    pred = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)[0]

    tracks = _prediction_to_tracks(pred)
    out = selector.update(tracks, (w, h))

    res_frame = _draw_selection_overlay(frame, out)

    cv2.imshow("Video", res_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
