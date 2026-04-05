from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from ..domain.models import FeatureVector, ScoreBreakdown, TrackCandidate


@dataclass
class ExternalSignals:
    preferred_track_id: int | None = None
    track_score_bias: dict[int, float] = field(default_factory=dict)
    class_score_bias: dict[str, float] = field(default_factory=dict)
    external_hint_score: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: Mapping[str, object] | None) -> "ExternalSignals":
        if not isinstance(raw, Mapping):
            return ExternalSignals()

        preferred_track_id = raw.get("preferred_track_id")
        preferred = None
        if isinstance(preferred_track_id, (int, float)):
            preferred = int(preferred_track_id)

        track_bias_raw = raw.get("track_score_bias")
        track_bias: dict[int, float] = {}
        if isinstance(track_bias_raw, Mapping):
            for key, value in track_bias_raw.items():
                try:
                    track_id = int(key)
                    track_bias[track_id] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue

        class_bias_raw = raw.get("class_score_bias")
        class_bias: dict[str, float] = {}
        if isinstance(class_bias_raw, Mapping):
            for key, value in class_bias_raw.items():
                try:
                    class_bias[str(key)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue

        hint_raw = raw.get("external_hint_score")
        hint_score: dict[int, float] = {}
        if isinstance(hint_raw, Mapping):
            for key, value in hint_raw.items():
                try:
                    track_id = int(key)
                    hint_score[track_id] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue

        return ExternalSignals(
            preferred_track_id=preferred,
            track_score_bias=track_bias,
            class_score_bias=class_bias,
            external_hint_score=hint_score,
        )


@dataclass(frozen=True)
class PolicyContext:
    candidates: Mapping[int, TrackCandidate]
    features: Mapping[int, FeatureVector]
    external_signals: ExternalSignals


class BaseScoringPolicy:
    name = "single_best"

    def apply(
        self,
        score: ScoreBreakdown,
        feature: FeatureVector,
        candidate: TrackCandidate | None,
        context: PolicyContext,
    ) -> ScoreBreakdown:
        return score


@dataclass
class SingleBestPolicy(BaseScoringPolicy):
    name: str = "single_best"


@dataclass
class CenterBiasedPolicy(BaseScoringPolicy):
    weight: float = 0.20
    name: str = "center_biased"

    def apply(self, score: ScoreBreakdown, feature: FeatureVector, candidate: TrackCandidate | None, context: PolicyContext) -> ScoreBreakdown:
        bonus = float(self.weight) * float(feature.center_score)
        if abs(bonus) < 1e-12:
            return score
        return replace(
            score,
            policy_contrib=score.policy_contrib + bonus,
            final_score=score.final_score + bonus,
        )


@dataclass
class StableTargetPolicy(BaseScoringPolicy):
    weight: float = 0.20
    lifetime_mix: float = 0.35
    name: str = "stable_target"

    def apply(self, score: ScoreBreakdown, feature: FeatureVector, candidate: TrackCandidate | None, context: PolicyContext) -> ScoreBreakdown:
        stability_component = (1.0 - self.lifetime_mix) * feature.stability_score
        lifetime_component = self.lifetime_mix * feature.lifetime_score
        bonus = float(self.weight) * float(stability_component + lifetime_component)
        if abs(bonus) < 1e-12:
            return score
        return replace(
            score,
            policy_contrib=score.policy_contrib + bonus,
            final_score=score.final_score + bonus,
        )


@dataclass
class LargestTargetPolicy(BaseScoringPolicy):
    weight: float = 0.20
    name: str = "largest_target"

    def apply(self, score: ScoreBreakdown, feature: FeatureVector, candidate: TrackCandidate | None, context: PolicyContext) -> ScoreBreakdown:
        bonus = float(self.weight) * float(feature.area_score)
        if abs(bonus) < 1e-12:
            return score
        return replace(
            score,
            policy_contrib=score.policy_contrib + bonus,
            final_score=score.final_score + bonus,
        )


@dataclass
class ClassPriorityPolicy(BaseScoringPolicy):
    weights: Mapping[str, float] = field(default_factory=dict)
    gain: float = 0.25
    name: str = "class_priority"

    def apply(self, score: ScoreBreakdown, feature: FeatureVector, candidate: TrackCandidate | None, context: PolicyContext) -> ScoreBreakdown:
        if candidate is None:
            return score

        class_name_key = str(candidate.class_name)
        class_id_key = str(candidate.class_id)
        weight = float(
            self.weights.get(
                class_name_key,
                self.weights.get(
                    class_id_key,
                    self.weights.get("*", 1.0),
                ),
            )
        )
        bonus = float(self.gain) * (weight - 1.0)
        if abs(bonus) < 1e-12:
            return score
        return replace(
            score,
            policy_contrib=score.policy_contrib + bonus,
            final_score=score.final_score + bonus,
        )


def resolve_policy(
    name: str,
    default_weight: float,
    class_priority_weights: Mapping[str, float],
    class_priority_gain: float,
) -> BaseScoringPolicy:
    policy_name = (name or "single_best").strip().lower()
    if policy_name == "center_biased":
        return CenterBiasedPolicy(weight=float(default_weight))
    if policy_name == "stable_target":
        return StableTargetPolicy(weight=float(default_weight))
    if policy_name == "largest_target":
        return LargestTargetPolicy(weight=float(default_weight))
    if policy_name == "class_priority":
        return ClassPriorityPolicy(
            weights=dict(class_priority_weights),
            gain=float(class_priority_gain),
        )
    return SingleBestPolicy()


def apply_external_signals(
    score: ScoreBreakdown,
    candidate: TrackCandidate | None,
    signals: ExternalSignals,
    preferred_track_bonus: float,
    track_bias_scale: float,
    class_bias_scale: float,
    hint_scale: float,
) -> ScoreBreakdown:
    if candidate is None:
        return score

    external_bonus = 0.0

    if signals.preferred_track_id is not None and candidate.track_id == signals.preferred_track_id:
        external_bonus += float(preferred_track_bonus)

    external_bonus += float(track_bias_scale) * float(signals.track_score_bias.get(candidate.track_id, 0.0))
    external_bonus += float(hint_scale) * float(signals.external_hint_score.get(candidate.track_id, 0.0))

    class_name_key = str(candidate.class_name)
    class_id_key = str(candidate.class_id)
    class_bias = float(
        signals.class_score_bias.get(
            class_name_key,
            signals.class_score_bias.get(class_id_key, signals.class_score_bias.get("*", 0.0)),
        )
    )
    external_bonus += float(class_bias_scale) * class_bias

    if abs(external_bonus) < 1e-12:
        return score

    return replace(
        score,
        external_contrib=score.external_contrib + external_bonus,
        final_score=score.final_score + external_bonus,
    )
