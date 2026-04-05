from __future__ import annotations

import math

from pts.target_selection.domain.models import FeatureVector, TrackCandidate
from pts.target_selection.scoring.scorer import ScoringConfig, TargetScorer


def _feature(
    track_id: int,
    conf: float = 0.5,
    lifetime: float = 0.5,
    area: float = 0.5,
    growth: float = 0.5,
    center: float = 0.5,
    stability: float = 0.5,
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


def _candidate(track_id: int, class_id: int = 0, class_name: str = "soldier") -> TrackCandidate:
    return TrackCandidate(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        bbox=(10.0, 10.0, 30.0, 30.0),
        current_visible=True,
        smoothed_center=(0.5, 0.5),
        smoothed_area_ratio=0.01,
        lifetime_frames=10,
        avg_conf=0.7,
        miss_count=0,
        center_jitter=0.01,
        area_jitter=0.01,
        area_history=[0.009, 0.01],
        accepted=True,
    )


def test_single_best_has_no_extra_contrib() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="single_best",
            policy_strength=0.3,
            max_policy_bonus_abs=0.2,
            max_external_bonus_abs=0.2,
            max_total_bonus_abs=0.2,
        )
    )
    feat = _feature(track_id=1, conf=0.9, center=0.2)
    score = scorer.score(features=feat, candidate=_candidate(1))

    base = (
        score.conf_contrib
        + score.lifetime_contrib
        + score.area_contrib
        + score.growth_contrib
        + score.center_contrib
        + score.stability_contrib
    )
    assert math.isclose(score.policy_contrib, 0.0, abs_tol=1e-12)
    assert math.isclose(score.external_contrib, 0.0, abs_tol=1e-12)
    assert math.isclose(score.final_score, base, rel_tol=1e-9, abs_tol=1e-9)


def test_center_policy_bonus_is_clipped() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="center_biased",
            policy_strength=0.5,
            w_conf=0.0,
            w_lifetime=0.0,
            w_area=0.0,
            w_growth=0.0,
            w_center=0.0,
            w_stability=0.0,
            max_policy_bonus_abs=0.10,
            max_external_bonus_abs=1.0,
            max_total_bonus_abs=1.0,
        )
    )
    score = scorer.score(features=_feature(track_id=1, center=1.0), candidate=_candidate(1))
    assert math.isclose(score.policy_contrib, 0.10, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(score.final_score, 0.10, rel_tol=1e-9, abs_tol=1e-9)
    assert score.policy_raw_contrib >= score.policy_contrib
    assert score.policy_clip_applied == 1.0


def test_external_preferred_bonus_is_clipped() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="single_best",
            preferred_track_bonus=0.50,
            w_conf=0.0,
            w_lifetime=0.0,
            w_area=0.0,
            w_growth=0.0,
            w_center=0.0,
            w_stability=0.0,
            max_policy_bonus_abs=1.0,
            max_external_bonus_abs=0.20,
            max_total_bonus_abs=1.0,
        )
    )
    score = scorer.score(
        features=_feature(track_id=7),
        candidate=_candidate(7),
        external_signals={"preferred_track_id": 7},
    )
    assert math.isclose(score.external_contrib, 0.20, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(score.final_score, 0.20, rel_tol=1e-9, abs_tol=1e-9)


def test_combined_bonus_is_capped_by_total_guardrail() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="center_biased",
            policy_strength=0.30,
            preferred_track_bonus=0.30,
            w_conf=0.0,
            w_lifetime=0.0,
            w_area=0.0,
            w_growth=0.0,
            w_center=0.0,
            w_stability=0.0,
            max_policy_bonus_abs=0.30,
            max_external_bonus_abs=0.30,
            max_total_bonus_abs=0.25,
        )
    )
    score = scorer.score(
        features=_feature(track_id=9, center=1.0),
        candidate=_candidate(9),
        external_signals={"preferred_track_id": 9},
    )
    assert score.policy_contrib > 0.0
    assert score.external_contrib > 0.0
    assert math.isclose(score.policy_contrib + score.external_contrib, 0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(score.final_score, 0.25, rel_tol=1e-9, abs_tol=1e-9)


def test_center_biased_ranks_center_target_higher() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="center_biased",
            policy_strength=0.2,
            max_policy_bonus_abs=1.0,
            max_external_bonus_abs=1.0,
            max_total_bonus_abs=1.0,
        )
    )
    center = _feature(track_id=1, center=1.0)
    edge = _feature(track_id=2, center=0.1)
    scores = scorer.score_many(
        features={1: center, 2: edge},
        candidates={1: _candidate(1), 2: _candidate(2)},
    )
    assert scores[1].final_score > scores[2].final_score
    assert scores[1].policy_contrib > scores[2].policy_contrib


def test_external_bias_is_noop_without_signals() -> None:
    scorer = TargetScorer(
        ScoringConfig(
            policy_name="single_best",
            preferred_track_bonus=0.2,
            track_bias_scale=1.0,
            class_bias_scale=1.0,
            external_hint_scale=1.0,
        )
    )
    score = scorer.score(features=_feature(track_id=1), candidate=_candidate(1))
    assert math.isclose(score.external_contrib, 0.0, abs_tol=1e-12)


def test_breakdown_fields_are_finite() -> None:
    scorer = TargetScorer(ScoringConfig(policy_name="stable_target", policy_strength=0.2))
    score = scorer.score(
        features=_feature(track_id=11, conf=0.7, lifetime=0.8, area=0.4, growth=0.6, center=0.9, stability=0.95),
        candidate=_candidate(11),
        external_signals={"track_score_bias": {11: 0.05}},
    )
    for value in (
        score.conf_contrib,
        score.lifetime_contrib,
        score.area_contrib,
        score.growth_contrib,
        score.center_contrib,
        score.stability_contrib,
        score.policy_contrib,
        score.external_contrib,
        score.final_score,
    ):
        assert math.isfinite(value)
