from __future__ import annotations

"""
Template: integrate PrimaryTargetSelection with any external tracker.

Replace `get_external_tracks(...)` with your tracker output adapter.
"""

from typing import Any

import cv2

from pts import PrimaryTargetSelection, SelectionTrack


def get_external_tracks(frame: Any) -> list[SelectionTrack]:
    """
    Adapt your tracker output here.
    Example shape required by pts:
      - track_id
      - bbox_xyxy
      - confidence
      - class_id / class_name (optional)
    """
    # TODO: replace with actual tracker output parsing.
    return []


def main() -> None:
    selector = PrimaryTargetSelection()
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    fps = 30.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        tracks = get_external_tracks(frame)
        out = selector.update(
            tracks=tracks,
            frame_size=(w, h),
            frame_idx=frame_idx,
            timestamp_s=frame_idx / max(fps, 1e-6),
        )

        # Your downstream logic can consume:
        # out.primary_track_id, out.events, out.selection_state
        if out.primary_track_id is not None:
            print(
                f"primary={out.primary_track_id} "
                f"score={(out.primary_score or 0.0):.3f} "
                f"reason={out.selection_reason}"
            )

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    main()
