from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SelectionStateName = Literal["no_target", "locked", "switch_pending", "lost"]
SelectionEventType = Literal[
    "target_acquired",
    "target_lost",
    "target_switched",
    "candidate_rejected",
    "score_updated",
]

SelectionReasonName = Literal[
    "no_valid_candidates",
    "acquire_score_below_threshold",
    "acquire_persistence_not_reached",
    "initial_acquire",
    "hold_current_target",
    "hold_lost_grace",
    "target_lost",
    "switch_margin_not_reached",
    "switch_persistence_not_reached",
    "switched_to_better_candidate",
]

RejectReasonName = Literal[
    "candidate_rejected_low_conf",
    "candidate_rejected_low_lifetime",
    "candidate_rejected_small_area",
    "candidate_rejected_unstable",
]

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectionTrack:
    """Input track schema for PrimaryTargetSelection."""

    track_id: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    class_id: int = -1
    class_name: str = "object"
    visible: bool = True


@dataclass(frozen=True)
class SelectionScoreBreakdown:
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


@dataclass(frozen=True)
class SelectionEvent:
    event_type: SelectionEventType
    track_id: int | None
    previous_track_id: int | None
    final_score: float | None
    selection_reason: SelectionReasonName | RejectReasonName | str
    score_breakdown: SelectionScoreBreakdown | None = None


@dataclass(frozen=True)
class SelectionCandidate:
    track_id: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    class_id: int
    class_name: str
    visible: bool
    accepted: bool
    reject_reason: RejectReasonName | str | None
    score: float | None
    score_breakdown: SelectionScoreBreakdown | None
    lifetime_frames: int
    center_norm: tuple[float, float]
    area_ratio: float


@dataclass
class SelectionOutput:
    frame_index: int
    timestamp_s: float
    primary_track_id: int | None
    selection_state: SelectionStateName
    selection_reason: SelectionReasonName | str
    primary_score: float | None
    switch_candidate_id: int | None
    events: list[SelectionEvent] = field(default_factory=list)
    candidates: list[SelectionCandidate] = field(default_factory=list)
    scores: dict[int, SelectionScoreBreakdown] = field(default_factory=dict)
