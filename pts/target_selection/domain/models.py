from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .enums import EventType, SelectionState


BBox = tuple[float, float, float, float]


@dataclass
class TrackObservation:
    frame_idx: int
    timestamp_s: float
    track_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    frame_width: int
    frame_height: int
    visible: bool = True

    @property
    def center_norm(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        cx = ((x1 + x2) * 0.5) / max(self.frame_width, 1)
        cy = ((y1 + y2) * 0.5) / max(self.frame_height, 1)
        return (cx, cy)

    @property
    def area_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        frame_area = max(float(self.frame_width * self.frame_height), 1.0)
        return area / frame_area


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    history_size: int
    observations: deque[TrackObservation] = field(default_factory=deque)
    center_history: deque[tuple[float, float]] = field(default_factory=deque)
    area_history: deque[float] = field(default_factory=deque)
    confidence_history: deque[float] = field(default_factory=deque)
    frame_indices: deque[int] = field(default_factory=deque)
    visible_history: deque[bool] = field(default_factory=deque)
    smoothed_center: tuple[float, float] = (0.0, 0.0)
    smoothed_area_ratio: float = 0.0
    lifetime_frames: int = 0
    miss_count: int = 0
    avg_conf: float = 0.0
    last_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    last_frame_idx: int = -1
    current_visible: bool = False

    def __post_init__(self) -> None:
        self.observations = deque(self.observations, maxlen=self.history_size)
        self.center_history = deque(self.center_history, maxlen=self.history_size)
        self.area_history = deque(self.area_history, maxlen=self.history_size)
        self.confidence_history = deque(self.confidence_history, maxlen=self.history_size)
        self.frame_indices = deque(self.frame_indices, maxlen=self.history_size)
        self.visible_history = deque(self.visible_history, maxlen=self.history_size)


@dataclass
class TrackCandidate:
    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    current_visible: bool
    smoothed_center: tuple[float, float]
    smoothed_area_ratio: float
    lifetime_frames: int
    avg_conf: float
    miss_count: int
    center_jitter: float
    area_jitter: float
    area_history: list[float]
    accepted: bool
    reject_reason: str | None = None


@dataclass
class FeatureVector:
    track_id: int
    conf_score: float
    lifetime_score: float
    area_score: float
    growth_score: float
    center_score: float
    stability_score: float


@dataclass
class ScoreBreakdown:
    track_id: int
    conf_contrib: float
    lifetime_contrib: float
    area_contrib: float
    growth_contrib: float
    center_contrib: float
    stability_contrib: float
    final_score: float
    policy_contrib: float = 0.0
    external_contrib: float = 0.0
    policy_raw_contrib: float = 0.0
    external_raw_contrib: float = 0.0
    policy_clip_abs: float = 0.0
    external_clip_abs: float = 0.0
    total_bonus_clip_abs: float = 0.0
    policy_clip_applied: float = 0.0
    external_clip_applied: float = 0.0
    total_bonus_clip_applied: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "conf_contrib": self.conf_contrib,
            "lifetime_contrib": self.lifetime_contrib,
            "area_contrib": self.area_contrib,
            "growth_contrib": self.growth_contrib,
            "center_contrib": self.center_contrib,
            "stability_contrib": self.stability_contrib,
            "policy_contrib": self.policy_contrib,
            "external_contrib": self.external_contrib,
            "policy_raw_contrib": self.policy_raw_contrib,
            "external_raw_contrib": self.external_raw_contrib,
            "policy_clip_abs": self.policy_clip_abs,
            "external_clip_abs": self.external_clip_abs,
            "total_bonus_clip_abs": self.total_bonus_clip_abs,
            "policy_clip_applied": self.policy_clip_applied,
            "external_clip_applied": self.external_clip_applied,
            "total_bonus_clip_applied": self.total_bonus_clip_applied,
            "final_score": self.final_score,
        }


@dataclass
class SelectionResult:
    primary_target_id: int | None
    primary_score: float | None
    selection_state: SelectionState
    switch_candidate_id: int | None
    selection_reason: str
    event_type: EventType | None = None
    previous_target_id: int | None = None


@dataclass
class EventRecord:
    frame_idx: int
    video_time_sec: float
    event_type: EventType
    track_id: int | None
    previous_track_id: int | None
    final_score: float | None
    score_breakdown: dict[str, float] | None
    bbox: list[float] | None
    avg_conf: float | None
    lifetime: int | None
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "video_time_sec": round(self.video_time_sec, 4),
            "event_type": self.event_type.value,
            "track_id": self.track_id,
            "previous_track_id": self.previous_track_id,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown,
            "bbox": self.bbox,
            "avg_conf": self.avg_conf,
            "lifetime": self.lifetime,
            "selection_reason": self.selection_reason,
        }


@dataclass
class FrameProcessingResult:
    frame_idx: int
    timestamp_s: float
    candidates: list[TrackCandidate]
    features: dict[int, FeatureVector]
    scores: dict[int, ScoreBreakdown]
    selection: SelectionResult
    events: list[EventRecord]
    active_tracks: dict[int, TrackState]
