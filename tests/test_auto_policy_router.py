from __future__ import annotations

from pts import PrimaryTargetSelection, SelectionTrack
from pts.target_selection.domain.enums import EventType, SelectionState
from pts.target_selection.domain.models import FeatureVector, SelectionResult, TrackCandidate
from pts.target_selection.pipeline.auto_policy import AutoPolicyConfig, AutoPolicyRouter


def _feature(
    track_id: int,
    conf: float = 0.7,
    lifetime: float = 0.7,
    area: float = 0.4,
    growth: float = 0.2,
    center: float = 0.6,
    stability: float = 0.8,
) -> FeatureVector:
    return FeatureVector(
        track_id=track_id,
        conf_score=conf,
        lifetime_score=lifetime,
        area_score=area,
        growth_score=growth,
        center_score=center,
        stability_score=stability,
    )


def _candidate(
    track_id: int,
    center: tuple[float, float] = (0.5, 0.5),
    conf: float = 0.7,
    lifetime_frames: int = 12,
    accepted: bool = True,
    visible: bool = True,
) -> TrackCandidate:
    return TrackCandidate(
        track_id=track_id,
        class_id=0,
        class_name="soldier",
        bbox=(10.0, 10.0, 40.0, 40.0),
        current_visible=visible,
        smoothed_center=center,
        smoothed_area_ratio=0.01,
        lifetime_frames=lifetime_frames,
        avg_conf=conf,
        miss_count=0,
        center_jitter=0.01,
        area_jitter=0.01,
        area_history=[0.009, 0.01],
        accepted=accepted,
    )


def _router(config: AutoPolicyConfig | None = None) -> AutoPolicyRouter:
    return AutoPolicyRouter(
        config=config or AutoPolicyConfig(),
        w_conf=0.25,
        w_lifetime=0.20,
        w_area=0.15,
        w_growth=0.10,
        w_center=0.15,
        w_stability=0.15,
    )


def test_auto_no_primary_uses_search_mode() -> None:
    router = _router()
    decision = router.choose(
        candidates={1: _candidate(1)},
        features={1: _feature(1)},
        primary_target_id=None,
    )
    assert decision.mode == "search_mode"
    assert decision.effective_policy_name == "single_best"
    assert decision.reason == "no_primary_target"


def test_auto_hold_mode_for_acceptable_primary() -> None:
    router = _router(config=AutoPolicyConfig(min_mode_dwell_frames=0, transition_persistence_frames=1))
    decision = router.choose(
        candidates={1: _candidate(1, center=(0.5, 0.5))},
        features={1: _feature(1, center=0.95, stability=0.9)},
        primary_target_id=1,
    )
    assert decision.mode == "hold_mode"
    assert decision.effective_policy_name == "stable_target"
    assert decision.primary_is_acceptable is True


def test_auto_center_assist_mode_for_off_center_primary() -> None:
    router = _router(config=AutoPolicyConfig(min_mode_dwell_frames=0, transition_persistence_frames=1))
    # Prime router in hold mode first.
    _ = router.choose(
        candidates={1: _candidate(1, center=(0.5, 0.5)), 2: _candidate(2, center=(0.9, 0.9))},
        features={1: _feature(1, center=0.95), 2: _feature(2, center=0.1)},
        primary_target_id=1,
    )
    decision = router.choose(
        candidates={1: _candidate(1, center=(0.10, 0.10)), 2: _candidate(2, center=(0.9, 0.9))},
        features={
            1: _feature(1, conf=0.9, lifetime=0.9, area=0.6, center=0.15, stability=0.9),
            2: _feature(2, conf=0.4, lifetime=0.3, area=0.1, center=0.9, stability=0.3),
        },
        primary_target_id=1,
    )
    assert decision.mode == "center_assist_mode"
    assert decision.effective_policy_name == "center_biased"
    assert decision.reason == "primary_off_center_assist"


def test_auto_scene_unstable_forces_hold() -> None:
    router = _router(
        config=AutoPolicyConfig(
            min_mode_dwell_frames=0,
            transition_persistence_frames=1,
            scene_window_frames=10,
            scene_lost_threshold=2,
            scene_switch_threshold=2,
        )
    )
    lost = SelectionResult(
        primary_target_id=None,
        primary_score=None,
        selection_state=SelectionState.NO_TARGET,
        switch_candidate_id=None,
        selection_reason="target_lost",
        event_type=EventType.TARGET_LOST,
        previous_target_id=1,
    )
    router.observe_selection(lost)
    router.observe_selection(lost)

    decision = router.choose(
        candidates={1: _candidate(1, center=(0.1, 0.1)), 2: _candidate(2, center=(0.9, 0.9))},
        features={
            1: _feature(1, conf=0.9, lifetime=0.9, area=0.6, center=0.15, stability=0.9),
            2: _feature(2, conf=0.4, lifetime=0.3, area=0.1, center=0.9, stability=0.3),
        },
        primary_target_id=1,
    )
    assert decision.scene_is_unstable is True
    assert decision.mode == "hold_mode"
    assert decision.effective_policy_name == "stable_target"
    assert decision.reason == "scene_unstable_hold_preferred"


def test_auto_unacceptable_primary_switches_to_search_immediately() -> None:
    router = _router(config=AutoPolicyConfig(min_mode_dwell_frames=50, transition_persistence_frames=10))
    _ = router.choose(
        candidates={1: _candidate(1)},
        features={1: _feature(1)},
        primary_target_id=1,
    )
    # Low stability should fail acceptability gate.
    decision = router.choose(
        candidates={1: _candidate(1)},
        features={1: _feature(1, stability=0.05)},
        primary_target_id=1,
    )
    assert decision.mode == "search_mode"
    assert decision.effective_policy_name == "single_best"
    assert decision.reason == "primary_unacceptable"


def test_selection_output_exposes_auto_fields() -> None:
    selector = PrimaryTargetSelection()
    out = selector.update(
        tracks=[
            SelectionTrack(
                track_id=101,
                bbox_xyxy=(100.0, 100.0, 150.0, 200.0),
                confidence=0.8,
                class_id=0,
                class_name="soldier",
                visible=True,
            )
        ],
        frame_size=(640, 480),
        frame_idx=0,
        timestamp_s=0.0,
        policy_name="auto",
    )
    assert out.effective_policy_name is not None
    assert out.auto_mode is not None
    assert out.auto_mode_reason is not None
