from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from pts import PrimaryTargetSelection
from pts.adapters.ultralytics import prediction_to_tracks, reset_ultralytics_trackers
from pts.visualization import draw_selection_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Ultralytics -> PTS pipeline.")
    parser.add_argument("--source", type=str, default="data/input/1.mp4")
    parser.add_argument("--model", type=Path, default=Path("models/model26n_2_7394f.onnx"))
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--selector-config", type=str, default=None)
    parser.add_argument("--show-rejected", action="store_true")
    return parser


def parse_source(raw: str) -> int | str:
    return int(raw) if raw.isdigit() else raw


def main() -> None:
    args = build_parser().parse_args()

    model = YOLO(str(args.model), task="detect")
    reset_ultralytics_trackers(model)

    selector = PrimaryTargetSelection(config_path=args.selector_config)

    vid = cv2.VideoCapture(parse_source(args.source))
    if not vid.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            pred = model.track(
                frame,
                tracker=args.tracker,
                persist=True,
                conf=float(args.conf),
                iou=float(args.iou),
                verbose=False,
            )[0]

            tracks = prediction_to_tracks(pred)
            out = selector.update_with_frame(tracks, frame)
            rendered = draw_selection_overlay(frame, out, show_rejected=bool(args.show_rejected))

            cv2.imshow("PTS", rendered)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
