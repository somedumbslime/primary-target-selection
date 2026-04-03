from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..domain.enums import EventType
from ..domain.models import (
    FeatureVector,
    FrameProcessingResult,
    ScoreBreakdown,
    SelectionResult,
    TrackCandidate,
    TrackObservation,
)
from ..features import FeatureExtractor, FeatureExtractorConfig
from ..logging import JsonlEventLogger
from ..logging.schemas import make_event_record
from ..scoring import ScoringConfig, TargetScorer
from ..selection import PrimarySelectorConfig, PrimaryTargetSelector
from ..tracking import TrackFilter, TrackFilterConfig, TrackStateStore, TrackStateStoreConfig


@dataclass
class TargetSelectionPipelineConfig:
    @dataclass
    class LoggingConfig:
        log_candidate_rejected: bool = False
        log_score_updated: bool = False
        score_update_top_k: int = 3

    store: TrackStateStoreConfig
    filtering: TrackFilterConfig
    features: FeatureExtractorConfig
    scoring: ScoringConfig
    selection: PrimarySelectorConfig
    logging: LoggingConfig

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "TargetSelectionPipelineConfig":
        logging_cfg = TargetSelectionPipelineConfig.LoggingConfig(**(raw.get("logging") or {}))
        return TargetSelectionPipelineConfig(
            store=TrackStateStoreConfig(**(raw.get("store") or {})),
            filtering=TrackFilterConfig(**(raw.get("filtering") or {})),
            features=FeatureExtractorConfig(**(raw.get("features") or {})),
            scoring=ScoringConfig(**(raw.get("scoring") or {})),
            selection=PrimarySelectorConfig(**(raw.get("selection") or {})),
            logging=logging_cfg,
        )


def load_target_selection_config(config_path: str | Path) -> TargetSelectionPipelineConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TargetSelectionPipelineConfig.from_dict(raw)


class TargetSelectionFrameProcessor:
    def __init__(self, config: TargetSelectionPipelineConfig, logger: JsonlEventLogger | None = None):
        self.config = config
        self.store = TrackStateStore(config.store)
        self.track_filter = TrackFilter(config.filtering)
        self.feature_extractor = FeatureExtractor(config.features)
        self.scorer = TargetScorer(config.scoring)
        self.selector = PrimaryTargetSelector(config.selection)
        self.logger = logger or JsonlEventLogger(enabled=False)

    def set_event_log_path(self, output_path: Path) -> None:
        self.logger.set_output_path(output_path)

    def reset(self) -> None:
        self.store.reset()
        self.selector.reset()

    def process_prediction(
        self,
        prediction,
        frame_idx: int,
        timestamp_s: float,
        frame_shape: tuple[int, int],
        class_names: dict[int, str] | None = None,
    ) -> FrameProcessingResult:
        observations = self._observations_from_prediction(
            prediction=prediction,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            frame_size=(frame_shape[1], frame_shape[0]),
            class_names=class_names or {},
        )
        return self.process_observations(
            observations=observations,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
        )

    def process_observations(
        self,
        observations: list[TrackObservation],
        frame_idx: int,
        timestamp_s: float,
    ) -> FrameProcessingResult:
        states = self.store.update(observations=observations, frame_idx=frame_idx)

        candidates = self.track_filter.build_candidates(states)
        accepted_candidates = [c for c in candidates if c.accepted]

        features: dict[int, FeatureVector] = {}
        for cand in accepted_candidates:
            features[cand.track_id] = self.feature_extractor.extract(cand)

        scores: dict[int, ScoreBreakdown] = self.scorer.score_many(features)
        selection: SelectionResult = self.selector.select(scores)

        events = self._build_events(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            candidates=candidates,
            scores=scores,
            selection=selection,
        )
        for event in events:
            self.logger.log(event)

        return FrameProcessingResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            candidates=candidates,
            features=features,
            scores=scores,
            selection=selection,
            events=events,
            active_tracks=dict(states),
        )

    def _class_name(self, prediction, cls_id: int, class_names: dict[int, str]) -> str:
        if cls_id in class_names:
            return class_names[cls_id]
        names = getattr(prediction, "names", {})
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def _observations_from_prediction(
        self,
        prediction,
        frame_idx: int,
        timestamp_s: float,
        frame_size: tuple[int, int],
        class_names: dict[int, str],
    ) -> list[TrackObservation]:
        w, h = frame_size
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = getattr(boxes, "id", None)
        if ids is None:
            return []

        observations: list[TrackObservation] = []
        for i, track_id_val in enumerate(ids.tolist()):
            track_id = int(track_id_val)
            bbox = tuple(float(v) for v in boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            observations.append(
                TrackObservation(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    track_id=track_id,
                    bbox=bbox,  # type: ignore[arg-type]
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self._class_name(prediction, cls_id, class_names),
                    frame_width=w,
                    frame_height=h,
                    visible=True,
                )
            )
        return observations

    def _build_events(
        self,
        frame_idx: int,
        timestamp_s: float,
        candidates: list[TrackCandidate],
        scores: dict[int, ScoreBreakdown],
        selection: SelectionResult,
    ):
        events = []
        candidate_map = {c.track_id: c for c in candidates}

        if selection.event_type is not None:
            selected_candidate = (
                candidate_map.get(selection.primary_target_id)
                if selection.primary_target_id is not None
                else None
            )
            breakdown = scores.get(selection.primary_target_id) if selection.primary_target_id is not None else None
            events.append(
                make_event_record(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    event_type=selection.event_type,
                    candidate=selected_candidate,
                    previous_track_id=selection.previous_target_id,
                    breakdown=breakdown,
                    reason=selection.selection_reason,
                )
            )

        if self.config.logging.log_candidate_rejected:
            for candidate in candidates:
                if candidate.accepted:
                    continue
                events.append(
                    make_event_record(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        event_type=EventType.CANDIDATE_REJECTED,
                        candidate=candidate,
                        previous_track_id=None,
                        breakdown=None,
                        reason=candidate.reject_reason or "rejected",
                    )
                )

        if self.config.logging.log_score_updated and scores:
            ranked = sorted(scores.values(), key=lambda s: s.final_score, reverse=True)
            top_k = max(1, int(self.config.logging.score_update_top_k))
            for breakdown in ranked[:top_k]:
                candidate = candidate_map.get(breakdown.track_id)
                events.append(
                    make_event_record(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        event_type=EventType.SCORE_UPDATED,
                        candidate=candidate,
                        previous_track_id=None,
                        breakdown=breakdown,
                        reason="score_updated_top_k",
                    )
                )

        return events
