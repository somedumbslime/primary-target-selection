from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping
from dataclasses import replace

from ..domain.models import FeatureVector, ScoreBreakdown, TrackCandidate
from .policies import ExternalSignals, PolicyContext, apply_external_signals, resolve_policy


@dataclass
class ScoringConfig:
    w_conf: float = 0.25
    w_lifetime: float = 0.20
    w_area: float = 0.15
    w_growth: float = 0.10
    w_center: float = 0.15
    w_stability: float = 0.15
    policy_name: str = "single_best"
    policy_strength: float = 0.20
    class_priority_gain: float = 0.25
    class_priority_weights: dict[str, float] = field(default_factory=dict)
    preferred_track_bonus: float = 0.20
    track_bias_scale: float = 1.0
    class_bias_scale: float = 1.0
    external_hint_scale: float = 1.0
    # Guardrails to prevent policy/external bonuses from dominating base ranking.
    # Non-positive values disable corresponding clamps.
    max_policy_bonus_abs: float = 0.20
    max_external_bonus_abs: float = 0.25
    max_total_bonus_abs: float = 0.35
    # Optional config for stateful "auto" policy router (used in frame processor).
    auto_policy: dict[str, Any] = field(default_factory=dict)


def _clamp_abs(value: float, max_abs: float) -> float:
    if max_abs <= 0.0:
        return value
    if value > max_abs:
        return max_abs
    if value < -max_abs:
        return -max_abs
    return value


class TargetScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def score(
        self,
        features: FeatureVector,
        candidate: TrackCandidate | None = None,
        policy_name: str | None = None,
        external_signals: Mapping[str, object] | ExternalSignals | None = None,
        all_candidates: Mapping[int, TrackCandidate] | None = None,
        all_features: Mapping[int, FeatureVector] | None = None,
    ) -> ScoreBreakdown:
        conf_contrib = self.config.w_conf * features.conf_score
        lifetime_contrib = self.config.w_lifetime * features.lifetime_score
        area_contrib = self.config.w_area * features.area_score
        growth_contrib = self.config.w_growth * features.growth_score
        center_contrib = self.config.w_center * features.center_score
        stability_contrib = self.config.w_stability * features.stability_score

        base_final_score = (
            conf_contrib
            + lifetime_contrib
            + area_contrib
            + growth_contrib
            + center_contrib
            + stability_contrib
        )

        score = ScoreBreakdown(
            track_id=features.track_id,
            conf_contrib=conf_contrib,
            lifetime_contrib=lifetime_contrib,
            area_contrib=area_contrib,
            growth_contrib=growth_contrib,
            center_contrib=center_contrib,
            stability_contrib=stability_contrib,
            final_score=base_final_score,
        )

        policy = resolve_policy(
            name=(policy_name if policy_name is not None else self.config.policy_name),
            default_weight=self.config.policy_strength,
            class_priority_weights=self.config.class_priority_weights,
            class_priority_gain=self.config.class_priority_gain,
        )

        if isinstance(external_signals, ExternalSignals):
            signals = external_signals
        else:
            signals = ExternalSignals.from_dict(external_signals)

        context = PolicyContext(
            candidates=all_candidates or {},
            features=all_features or {},
            external_signals=signals,
        )

        score = policy.apply(score=score, feature=features, candidate=candidate, context=context)
        score = apply_external_signals(
            score=score,
            candidate=candidate,
            signals=signals,
            preferred_track_bonus=self.config.preferred_track_bonus,
            track_bias_scale=self.config.track_bias_scale,
            class_bias_scale=self.config.class_bias_scale,
            hint_scale=self.config.external_hint_scale,
        )
        score = self._apply_bonus_guardrails(score=score, base_final_score=base_final_score)
        return score

    def score_many(
        self,
        features: dict[int, FeatureVector],
        candidates: Mapping[int, TrackCandidate] | None = None,
        policy_name: str | None = None,
        external_signals: Mapping[str, object] | ExternalSignals | None = None,
    ) -> dict[int, ScoreBreakdown]:
        candidates = candidates or {}
        return {
            track_id: self.score(
                vec,
                candidate=candidates.get(track_id),
                policy_name=policy_name,
                external_signals=external_signals,
                all_candidates=candidates,
                all_features=features,
            )
            for track_id, vec in features.items()
        }

    def _apply_bonus_guardrails(self, score: ScoreBreakdown, base_final_score: float) -> ScoreBreakdown:
        policy_raw = float(score.policy_contrib)
        external_raw = float(score.external_contrib)

        policy = _clamp_abs(policy_raw, float(self.config.max_policy_bonus_abs))
        external = _clamp_abs(external_raw, float(self.config.max_external_bonus_abs))

        policy_clip_abs = abs(policy_raw - policy)
        external_clip_abs = abs(external_raw - external)
        policy_clip_applied = 1.0 if policy_clip_abs > 1e-12 else 0.0
        external_clip_applied = 1.0 if external_clip_abs > 1e-12 else 0.0

        pre_total_bonus = policy + external
        total_bonus = _clamp_abs(pre_total_bonus, float(self.config.max_total_bonus_abs))
        total_bonus_clip_abs = abs(pre_total_bonus - total_bonus)
        total_bonus_clip_applied = 1.0 if total_bonus_clip_abs > 1e-12 else 0.0

        if abs(pre_total_bonus) > 1e-12 and total_bonus_clip_applied > 0.0:
            ratio = total_bonus / pre_total_bonus if abs(pre_total_bonus) > 1e-12 else 0.0
            policy_scaled = policy * ratio
            external_scaled = external * ratio
            policy_clip_abs += abs(policy - policy_scaled)
            external_clip_abs += abs(external - external_scaled)
            policy = policy_scaled
            external = external_scaled
        elif abs(pre_total_bonus) <= 1e-12:
            policy = 0.0
            external = 0.0

        final_score = base_final_score + policy + external
        if not isfinite(final_score):
            final_score = base_final_score
            policy = 0.0
            external = 0.0
            policy_raw = 0.0
            external_raw = 0.0
            policy_clip_abs = 0.0
            external_clip_abs = 0.0
            total_bonus_clip_abs = 0.0
            policy_clip_applied = 0.0
            external_clip_applied = 0.0
            total_bonus_clip_applied = 0.0

        return replace(
            score,
            policy_contrib=policy,
            external_contrib=external,
            policy_raw_contrib=policy_raw,
            external_raw_contrib=external_raw,
            policy_clip_abs=policy_clip_abs,
            external_clip_abs=external_clip_abs,
            total_bonus_clip_abs=total_bonus_clip_abs,
            policy_clip_applied=policy_clip_applied,
            external_clip_applied=external_clip_applied,
            total_bonus_clip_applied=total_bonus_clip_applied,
            final_score=final_score,
        )
