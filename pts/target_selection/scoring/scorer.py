from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import FeatureVector, ScoreBreakdown


@dataclass
class ScoringConfig:
    w_conf: float = 0.25
    w_lifetime: float = 0.20
    w_area: float = 0.15
    w_growth: float = 0.10
    w_center: float = 0.15
    w_stability: float = 0.15


class TargetScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def score(self, features: FeatureVector) -> ScoreBreakdown:
        conf_contrib = self.config.w_conf * features.conf_score
        lifetime_contrib = self.config.w_lifetime * features.lifetime_score
        area_contrib = self.config.w_area * features.area_score
        growth_contrib = self.config.w_growth * features.growth_score
        center_contrib = self.config.w_center * features.center_score
        stability_contrib = self.config.w_stability * features.stability_score

        final_score = (
            conf_contrib
            + lifetime_contrib
            + area_contrib
            + growth_contrib
            + center_contrib
            + stability_contrib
        )

        return ScoreBreakdown(
            track_id=features.track_id,
            conf_contrib=conf_contrib,
            lifetime_contrib=lifetime_contrib,
            area_contrib=area_contrib,
            growth_contrib=growth_contrib,
            center_contrib=center_contrib,
            stability_contrib=stability_contrib,
            final_score=final_score,
        )

    def score_many(self, features: dict[int, FeatureVector]) -> dict[int, ScoreBreakdown]:
        return {track_id: self.score(vec) for track_id, vec in features.items()}
