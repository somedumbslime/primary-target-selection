from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Mapping

from ..domain.enums import EventType
from ..domain.models import FeatureVector, SelectionResult, TrackCandidate


@dataclass
class AutoPolicyConfig:
    center_assist_enter_dist: float = 0.30
    center_assist_exit_dist: float = 0.22
    min_primary_stability: float = 0.45
    min_primary_conf: float = 0.35
    min_primary_lifetime: int = 6
    min_margin_for_center_assist: float = 0.03
    min_mode_dwell_frames: int = 8
    transition_persistence_frames: int = 2
    scene_window_frames: int = 60
    scene_lost_threshold: int = 2
    scene_switch_threshold: int = 2

    @staticmethod
    def from_dict(raw: Mapping[str, object] | None) -> "AutoPolicyConfig":
        if not isinstance(raw, Mapping):
            return AutoPolicyConfig()

        def _as_float(name: str, default: float) -> float:
            try:
                return float(raw.get(name, default))
            except (TypeError, ValueError):
                return default

        def _as_int(name: str, default: int) -> int:
            try:
                return int(raw.get(name, default))
            except (TypeError, ValueError):
                return default

        return AutoPolicyConfig(
            center_assist_enter_dist=_as_float("center_assist_enter_dist", 0.30),
            center_assist_exit_dist=_as_float("center_assist_exit_dist", 0.22),
            min_primary_stability=_as_float("min_primary_stability", 0.45),
            min_primary_conf=_as_float("min_primary_conf", 0.35),
            min_primary_lifetime=_as_int("min_primary_lifetime", 6),
            min_margin_for_center_assist=_as_float("min_margin_for_center_assist", 0.03),
            min_mode_dwell_frames=_as_int("min_mode_dwell_frames", 8),
            transition_persistence_frames=_as_int("transition_persistence_frames", 2),
            scene_window_frames=_as_int("scene_window_frames", 60),
            scene_lost_threshold=_as_int("scene_lost_threshold", 2),
            scene_switch_threshold=_as_int("scene_switch_threshold", 2),
        )


@dataclass(frozen=True)
class AutoPolicyDecision:
    effective_policy_name: str
    mode: str
    reason: str
    scene_is_unstable: bool
    primary_is_acceptable: bool
    primary_center_dist: float | None
    primary_margin_to_runner_up: float | None


class AutoPolicyRouter:
    SEARCH_MODE = "search_mode"
    HOLD_MODE = "hold_mode"
    CENTER_ASSIST_MODE = "center_assist_mode"

    _MODE_TO_POLICY = {
        SEARCH_MODE: "single_best",
        HOLD_MODE: "stable_target",
        CENTER_ASSIST_MODE: "center_biased",
    }

    _IMMEDIATE_SEARCH_REASONS = {
        "no_candidates_search",
        "no_primary_target",
        "primary_unacceptable",
    }

    def __init__(
        self,
        config: AutoPolicyConfig,
        w_conf: float,
        w_lifetime: float,
        w_area: float,
        w_growth: float,
        w_center: float,
        w_stability: float,
    ) -> None:
        self.config = config
        self._weights = (
            float(w_conf),
            float(w_lifetime),
            float(w_area),
            float(w_growth),
            float(w_center),
            float(w_stability),
        )
        self._mode = self.SEARCH_MODE
        self._mode_dwell_frames = 0
        self._pending_mode: str | None = None
        self._pending_count = 0
        self._scene_events: deque[tuple[int, int]] = deque(
            maxlen=max(1, int(self.config.scene_window_frames))
        )

    def reset(self) -> None:
        self._mode = self.SEARCH_MODE
        self._mode_dwell_frames = 0
        self._pending_mode = None
        self._pending_count = 0
        self._scene_events.clear()

    def observe_selection(self, selection: SelectionResult) -> None:
        lost = 1 if selection.event_type == EventType.TARGET_LOST else 0
        switched = 1 if selection.event_type == EventType.TARGET_SWITCHED else 0
        self._scene_events.append((lost, switched))

    def choose(
        self,
        candidates: Mapping[int, TrackCandidate],
        features: Mapping[int, FeatureVector],
        primary_target_id: int | None,
    ) -> AutoPolicyDecision:
        primary_candidate = candidates.get(primary_target_id) if primary_target_id is not None else None
        primary_feature = features.get(primary_target_id) if primary_target_id is not None else None

        scene_is_unstable = self._is_scene_unstable()
        center_dist = self._center_dist(primary_candidate)
        primary_margin = self._primary_margin(primary_target_id, features)
        primary_acceptable = self._is_primary_acceptable(primary_candidate, primary_feature)

        target_mode, target_reason = self._target_mode(
            candidates=candidates,
            primary_candidate=primary_candidate,
            primary_acceptable=primary_acceptable,
            primary_center_dist=center_dist,
            primary_margin_to_runner_up=primary_margin,
            scene_is_unstable=scene_is_unstable,
        )
        active_mode, active_reason = self._advance_mode(target_mode, target_reason)
        effective_policy = self._MODE_TO_POLICY.get(active_mode, "single_best")

        return AutoPolicyDecision(
            effective_policy_name=effective_policy,
            mode=active_mode,
            reason=active_reason,
            scene_is_unstable=scene_is_unstable,
            primary_is_acceptable=primary_acceptable,
            primary_center_dist=center_dist,
            primary_margin_to_runner_up=primary_margin,
        )

    def _target_mode(
        self,
        candidates: Mapping[int, TrackCandidate],
        primary_candidate: TrackCandidate | None,
        primary_acceptable: bool,
        primary_center_dist: float | None,
        primary_margin_to_runner_up: float | None,
        scene_is_unstable: bool,
    ) -> tuple[str, str]:
        if not candidates:
            return self.SEARCH_MODE, "no_candidates_search"

        if primary_candidate is None:
            return self.SEARCH_MODE, "no_primary_target"

        if not primary_acceptable:
            return self.SEARCH_MODE, "primary_unacceptable"

        if scene_is_unstable:
            return self.HOLD_MODE, "scene_unstable_hold_preferred"

        if self._should_center_assist(primary_center_dist, primary_margin_to_runner_up):
            return self.CENTER_ASSIST_MODE, "primary_off_center_assist"

        return self.HOLD_MODE, "primary_centered_or_stable_hold"

    def _advance_mode(self, target_mode: str, target_reason: str) -> tuple[str, str]:
        if self._mode_dwell_frames <= 0:
            self._mode = target_mode
            self._mode_dwell_frames = 1
            self._reset_pending()
            return self._mode, target_reason

        if target_mode == self._mode:
            self._mode_dwell_frames += 1
            self._reset_pending()
            return self._mode, target_reason

        if target_mode == self.SEARCH_MODE and target_reason in self._IMMEDIATE_SEARCH_REASONS:
            self._mode = target_mode
            self._mode_dwell_frames = 1
            self._reset_pending()
            return self._mode, target_reason

        min_dwell = max(0, int(self.config.min_mode_dwell_frames))
        if self._mode_dwell_frames < min_dwell:
            self._mode_dwell_frames += 1
            self._reset_pending()
            return self._mode, "mode_dwell_not_reached"

        if self._pending_mode == target_mode:
            self._pending_count += 1
        else:
            self._pending_mode = target_mode
            self._pending_count = 1

        required = max(1, int(self.config.transition_persistence_frames))
        if self._pending_count >= required:
            self._mode = target_mode
            self._mode_dwell_frames = 1
            self._reset_pending()
            return self._mode, target_reason

        self._mode_dwell_frames += 1
        return self._mode, "transition_persistence_not_reached"

    def _reset_pending(self) -> None:
        self._pending_mode = None
        self._pending_count = 0

    def _is_scene_unstable(self) -> bool:
        if not self._scene_events:
            return False
        lost_count = sum(item[0] for item in self._scene_events)
        switched_count = sum(item[1] for item in self._scene_events)
        return (
            lost_count >= max(1, int(self.config.scene_lost_threshold))
            or switched_count >= max(1, int(self.config.scene_switch_threshold))
        )

    @staticmethod
    def _center_dist(candidate: TrackCandidate | None) -> float | None:
        if candidate is None:
            return None
        cx, cy = candidate.smoothed_center
        return hypot(float(cx) - 0.5, float(cy) - 0.5)

    def _primary_margin(
        self,
        primary_target_id: int | None,
        features: Mapping[int, FeatureVector],
    ) -> float | None:
        if primary_target_id is None:
            return None
        primary_feature = features.get(primary_target_id)
        if primary_feature is None:
            return None

        primary_score = self._base_score(primary_feature)
        runner_up = None
        for track_id, feature in features.items():
            if int(track_id) == int(primary_target_id):
                continue
            score = self._base_score(feature)
            if runner_up is None or score > runner_up:
                runner_up = score
        if runner_up is None:
            return 1.0
        return primary_score - runner_up

    def _is_primary_acceptable(
        self,
        candidate: TrackCandidate | None,
        feature: FeatureVector | None,
    ) -> bool:
        if candidate is None or feature is None:
            return False
        if not bool(candidate.accepted):
            return False
        if not bool(candidate.current_visible):
            return False
        if float(candidate.avg_conf) < float(self.config.min_primary_conf):
            return False
        if int(candidate.lifetime_frames) < int(self.config.min_primary_lifetime):
            return False
        if float(feature.stability_score) < float(self.config.min_primary_stability):
            return False
        return True

    def _should_center_assist(
        self,
        center_dist: float | None,
        primary_margin_to_runner_up: float | None,
    ) -> bool:
        if center_dist is None:
            return False
        if center_dist <= float(self.config.center_assist_exit_dist):
            return False
        if center_dist < float(self.config.center_assist_enter_dist):
            return False
        if primary_margin_to_runner_up is None:
            return False
        if primary_margin_to_runner_up < float(self.config.min_margin_for_center_assist):
            return False
        return True

    def _base_score(self, feature: FeatureVector) -> float:
        w_conf, w_lifetime, w_area, w_growth, w_center, w_stability = self._weights
        return (
            w_conf * float(feature.conf_score)
            + w_lifetime * float(feature.lifetime_score)
            + w_area * float(feature.area_score)
            + w_growth * float(feature.growth_score)
            + w_center * float(feature.center_score)
            + w_stability * float(feature.stability_score)
        )
