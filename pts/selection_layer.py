from __future__ import annotations

from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any, Sequence
import tempfile

from .target_selection.domain.models import TrackObservation
from .target_selection.logging import JsonlEventLogger
from .target_selection.pipeline import (
    TargetSelectionFrameProcessor,
    TargetSelectionPipelineConfig,
    load_target_selection_config,
)
from .types import (
    SelectionCandidate,
    SelectionEvent,
    SelectionOutput,
    SelectionScoreBreakdown,
    SelectionTrack,
)


RESOURCE_PACKAGE = "pts.resources"
RESOURCE_CACHE_DIR = Path(tempfile.gettempdir()) / "pts_resources"
TARGET_SELECTION_RESOURCE = "target_selection.yaml"


def _to_breakdown(raw: Any) -> SelectionScoreBreakdown:
    return SelectionScoreBreakdown(
        conf_contrib=float(raw.conf_contrib),
        lifetime_contrib=float(raw.lifetime_contrib),
        area_contrib=float(raw.area_contrib),
        growth_contrib=float(raw.growth_contrib),
        center_contrib=float(raw.center_contrib),
        stability_contrib=float(raw.stability_contrib),
        final_score=float(raw.final_score),
    )


def _to_event(raw_event: Any) -> SelectionEvent:
    breakdown = None
    if isinstance(raw_event.score_breakdown, dict):
        breakdown = SelectionScoreBreakdown(
            conf_contrib=float(raw_event.score_breakdown.get("conf_contrib", 0.0)),
            lifetime_contrib=float(raw_event.score_breakdown.get("lifetime_contrib", 0.0)),
            area_contrib=float(raw_event.score_breakdown.get("area_contrib", 0.0)),
            growth_contrib=float(raw_event.score_breakdown.get("growth_contrib", 0.0)),
            center_contrib=float(raw_event.score_breakdown.get("center_contrib", 0.0)),
            stability_contrib=float(raw_event.score_breakdown.get("stability_contrib", 0.0)),
            final_score=float(raw_event.score_breakdown.get("final_score", 0.0)),
        )

    return SelectionEvent(
        event_type=str(raw_event.event_type.value),
        track_id=raw_event.track_id,
        previous_track_id=raw_event.previous_track_id,
        final_score=raw_event.final_score,
        selection_reason=str(raw_event.selection_reason),
        score_breakdown=breakdown,
    )


def _normalize_selection_state(raw_state: str, events: list[SelectionEvent]) -> str:
    if raw_state == "target_locked":
        return "locked"
    if raw_state == "switch_pending":
        return "switch_pending"
    if raw_state == "no_target":
        if any(evt.event_type == "target_lost" for evt in events):
            return "lost"
        return "no_target"
    return "no_target"


class PrimaryTargetSelection:
    """
    Generic selection layer that runs after any external tracker.

    Input:
      - tracks: id, bbox_xyxy, confidence, class_id/class_name, visible
      - frame_size: (width, height)
      - frame_idx, timestamp_s

    Output:
      - primary_track_id
      - selection_state
      - selection_reason
      - primary_score
      - switch_candidate_id
      - events
      - optional debug: candidates, scores
    """

    def __init__(
        self,
        config: TargetSelectionPipelineConfig | dict[str, Any] | None = None,
        config_path: str | None = None,
        save_events_jsonl: bool = False,
        events_output_path: str | None = None,
    ) -> None:
        if isinstance(config, TargetSelectionPipelineConfig):
            pipeline_cfg = config
        elif isinstance(config, dict):
            pipeline_cfg = TargetSelectionPipelineConfig.from_dict(config)
        elif config_path:
            pipeline_cfg = load_target_selection_config(Path(config_path))
        else:
            default_path = self._materialize_default_config()
            if default_path is not None:
                pipeline_cfg = load_target_selection_config(default_path)
            else:
                pipeline_cfg = TargetSelectionPipelineConfig.from_dict({})

        self.logger = JsonlEventLogger(enabled=bool(save_events_jsonl))
        if events_output_path:
            self.logger.set_output_path(Path(events_output_path))
        self.processor = TargetSelectionFrameProcessor(config=pipeline_cfg, logger=self.logger)
        self.frame_idx = 0

    @staticmethod
    def _materialize_default_config() -> Path | None:
        try:
            data = resources.files(RESOURCE_PACKAGE).joinpath(TARGET_SELECTION_RESOURCE).read_bytes()
        except Exception:
            return None
        RESOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target_path = RESOURCE_CACHE_DIR / TARGET_SELECTION_RESOURCE
        target_path.write_bytes(data)
        return target_path

    def describe(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "config": asdict(self.processor.config),
            "events_logging_enabled": bool(self.logger.enabled),
            "events_output_path": None if self.logger.output_path is None else str(self.logger.output_path),
        }

    def reset(self) -> None:
        self.frame_idx = 0
        self.processor.reset()

    def set_event_output(self, output_path: str | None) -> None:
        if not output_path:
            self.logger.enabled = False
            self.logger.output_path = None
            return
        self.logger.enabled = True
        self.logger.set_output_path(Path(output_path))

    def update(
        self,
        tracks: Sequence[SelectionTrack | dict[str, Any]],
        frame_size: tuple[int, int],
        frame_idx: int | None = None,
        timestamp_s: float | None = None,
    ) -> SelectionOutput:
        idx = self.frame_idx if frame_idx is None else int(frame_idx)
        ts = float(idx) if timestamp_s is None else float(timestamp_s)
        observations = self._build_observations(
            tracks=tracks,
            frame_idx=idx,
            timestamp_s=ts,
            frame_size=frame_size,
        )

        processed = self.processor.process_observations(
            observations=observations,
            frame_idx=idx,
            timestamp_s=ts,
        )
        self.frame_idx = idx + 1
        return self._build_output(processed)

    def update_from_prediction(
        self,
        prediction: Any,
        frame_shape: tuple[int, int],
        frame_idx: int | None = None,
        timestamp_s: float | None = None,
        class_names: dict[int, str] | None = None,
    ) -> SelectionOutput:
        idx = self.frame_idx if frame_idx is None else int(frame_idx)
        ts = float(idx) if timestamp_s is None else float(timestamp_s)
        processed = self.processor.process_prediction(
            prediction=prediction,
            frame_idx=idx,
            timestamp_s=ts,
            frame_shape=frame_shape,
            class_names=class_names or {},
        )
        self.frame_idx = idx + 1
        return self._build_output(processed)

    @staticmethod
    def _to_track(track: SelectionTrack | dict[str, Any]) -> SelectionTrack:
        if isinstance(track, SelectionTrack):
            return track
        if not isinstance(track, dict):
            raise TypeError(f"Unsupported track type: {type(track)!r}")
        bbox = track.get("bbox_xyxy", track.get("bbox"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("track['bbox_xyxy'] (or 'bbox') must contain 4 values")
        return SelectionTrack(
            track_id=int(track.get("track_id")),
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            confidence=float(track.get("confidence", track.get("conf", 0.0))),
            class_id=int(track.get("class_id", track.get("cls_id", -1))),
            class_name=str(track.get("class_name", track.get("label", "object"))),
            visible=bool(track.get("visible", True)),
        )

    def _build_observations(
        self,
        tracks: Sequence[SelectionTrack | dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        frame_size: tuple[int, int],
    ) -> list[TrackObservation]:
        frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
        observations: list[TrackObservation] = []
        for item in tracks:
            track = self._to_track(item)
            observations.append(
                TrackObservation(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    track_id=int(track.track_id),
                    bbox=(
                        float(track.bbox_xyxy[0]),
                        float(track.bbox_xyxy[1]),
                        float(track.bbox_xyxy[2]),
                        float(track.bbox_xyxy[3]),
                    ),
                    confidence=float(track.confidence),
                    class_id=int(track.class_id),
                    class_name=str(track.class_name),
                    frame_width=frame_w,
                    frame_height=frame_h,
                    visible=bool(track.visible),
                )
            )
        return observations

    @staticmethod
    def _build_output(processed: Any) -> SelectionOutput:
        events = [_to_event(evt) for evt in processed.events]
        score_map = {int(track_id): _to_breakdown(score) for track_id, score in processed.scores.items()}

        candidates: list[SelectionCandidate] = []
        for candidate in processed.candidates:
            candidate_score = processed.scores.get(candidate.track_id)
            candidates.append(
                SelectionCandidate(
                    track_id=int(candidate.track_id),
                    bbox_xyxy=tuple(float(v) for v in candidate.bbox),
                    confidence=float(candidate.avg_conf),
                    class_id=int(candidate.class_id),
                    class_name=str(candidate.class_name),
                    visible=bool(candidate.current_visible),
                    accepted=bool(candidate.accepted),
                    reject_reason=candidate.reject_reason,
                    score=(None if candidate_score is None else float(candidate_score.final_score)),
                    score_breakdown=score_map.get(int(candidate.track_id)),
                    lifetime_frames=int(candidate.lifetime_frames),
                    center_norm=(
                        float(candidate.smoothed_center[0]),
                        float(candidate.smoothed_center[1]),
                    ),
                    area_ratio=float(candidate.smoothed_area_ratio),
                )
            )

        primary_track_id = (
            None if processed.selection.primary_target_id is None else int(processed.selection.primary_target_id)
        )
        primary_score = processed.selection.primary_score
        if primary_score is None and primary_track_id is not None:
            breakdown = score_map.get(primary_track_id)
            primary_score = None if breakdown is None else breakdown.final_score

        return SelectionOutput(
            frame_index=int(processed.frame_idx),
            timestamp_s=float(processed.timestamp_s),
            primary_track_id=primary_track_id,
            selection_state=_normalize_selection_state(str(processed.selection.selection_state.value), events),
            selection_reason=str(processed.selection.selection_reason),
            primary_score=None if primary_score is None else float(primary_score),
            switch_candidate_id=processed.selection.switch_candidate_id,
            events=events,
            candidates=candidates,
            scores=score_map,
        )

