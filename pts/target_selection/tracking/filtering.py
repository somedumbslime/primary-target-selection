from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import TrackCandidate, TrackState
from ..domain.reasons import RejectReason
from .smoothing import area_jitter, center_jitter


@dataclass
class TrackFilterConfig:
    min_track_lifetime: int = 3
    min_avg_conf: float = 0.35
    min_area_ratio: float = 0.0005
    max_miss_count: int = 3


class TrackFilter:
    def __init__(self, config: TrackFilterConfig):
        self.config = config

    def build_candidates(self, states: dict[int, TrackState]) -> list[TrackCandidate]:
        candidates: list[TrackCandidate] = []

        for state in states.values():
            c_jitter = center_jitter(list(state.center_history))
            a_jitter = area_jitter(list(state.area_history))
            accepted = True
            reason: str | None = None

            if state.lifetime_frames < self.config.min_track_lifetime:
                accepted = False
                reason = RejectReason.LOW_LIFETIME.value
            elif state.avg_conf < self.config.min_avg_conf:
                accepted = False
                reason = RejectReason.LOW_CONF.value
            elif state.smoothed_area_ratio < self.config.min_area_ratio:
                accepted = False
                reason = RejectReason.SMALL_AREA.value
            elif state.miss_count > self.config.max_miss_count:
                accepted = False
                reason = RejectReason.UNSTABLE.value

            candidates.append(
                TrackCandidate(
                    track_id=state.track_id,
                    class_id=state.class_id,
                    class_name=state.class_name,
                    bbox=state.last_bbox,
                    current_visible=state.current_visible,
                    smoothed_center=state.smoothed_center,
                    smoothed_area_ratio=state.smoothed_area_ratio,
                    lifetime_frames=state.lifetime_frames,
                    avg_conf=state.avg_conf,
                    miss_count=state.miss_count,
                    center_jitter=c_jitter,
                    area_jitter=a_jitter,
                    area_history=list(state.area_history),
                    accepted=accepted,
                    reject_reason=reason,
                )
            )

        return candidates
