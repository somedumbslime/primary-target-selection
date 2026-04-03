from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import TrackObservation, TrackState
from .smoothing import ema, ema_point


@dataclass
class TrackStateStoreConfig:
    history_size: int = 20
    alpha_center: float = 0.4
    alpha_area: float = 0.3
    max_missed_frames: int = 15


class TrackStateStore:
    def __init__(self, config: TrackStateStoreConfig):
        self.config = config
        self._states: dict[int, TrackState] = {}

    @property
    def states(self) -> dict[int, TrackState]:
        return self._states

    def reset(self) -> None:
        self._states.clear()

    def update(self, observations: list[TrackObservation], frame_idx: int) -> dict[int, TrackState]:
        seen: set[int] = set()

        for obs in observations:
            seen.add(obs.track_id)
            state = self._states.get(obs.track_id)
            if state is None:
                state = TrackState(
                    track_id=obs.track_id,
                    class_id=obs.class_id,
                    class_name=obs.class_name,
                    history_size=self.config.history_size,
                )
                self._states[obs.track_id] = state

            state.class_id = obs.class_id
            state.class_name = obs.class_name
            state.last_bbox = obs.bbox
            state.last_frame_idx = frame_idx
            state.current_visible = obs.visible
            state.miss_count = 0

            state.observations.append(obs)
            state.center_history.append(obs.center_norm)
            state.area_history.append(obs.area_ratio)
            state.confidence_history.append(obs.confidence)
            state.frame_indices.append(obs.frame_idx)
            state.visible_history.append(obs.visible)

            prev_center = None if state.lifetime_frames == 0 else state.smoothed_center
            prev_area = None if state.lifetime_frames == 0 else state.smoothed_area_ratio

            state.smoothed_center = ema_point(obs.center_norm, prev_center, self.config.alpha_center)
            state.smoothed_area_ratio = ema(obs.area_ratio, prev_area, self.config.alpha_area)
            state.lifetime_frames += 1
            state.avg_conf = sum(state.confidence_history) / max(len(state.confidence_history), 1)

        for track_id, state in list(self._states.items()):
            if track_id in seen:
                continue
            state.current_visible = False
            state.miss_count += 1
            state.visible_history.append(False)
            if state.miss_count > self.config.max_missed_frames:
                self._states.pop(track_id, None)

        return self._states
