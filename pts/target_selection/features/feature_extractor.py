from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from ..domain.models import FeatureVector, TrackCandidate
from .normalization import clamp01, minmax, ratio


@dataclass
class FeatureExtractorConfig:
    conf_min: float = 0.25
    conf_max: float = 0.95
    lifetime_ref: float = 20.0
    area_ref: float = 0.02
    growth_window: int = 5
    growth_clip: float = 0.5
    center_max_dist: float = 0.7071  # distance from center to corner in normalized coords
    center_jitter_ref: float = 0.08
    area_jitter_ref: float = 0.5


class FeatureExtractor:
    def __init__(self, config: FeatureExtractorConfig):
        self.config = config

    def extract(self, candidate: TrackCandidate) -> FeatureVector:
        conf_score = minmax(candidate.avg_conf, self.config.conf_min, self.config.conf_max)
        lifetime_score = ratio(float(candidate.lifetime_frames), self.config.lifetime_ref)
        area_score = ratio(candidate.smoothed_area_ratio, self.config.area_ref)

        growth_score = self._growth_score(candidate.area_history)
        center_score = self._center_score(candidate.smoothed_center)
        stability_score = self._stability_score(candidate.center_jitter, candidate.area_jitter)

        return FeatureVector(
            track_id=candidate.track_id,
            conf_score=conf_score,
            lifetime_score=lifetime_score,
            area_score=area_score,
            growth_score=growth_score,
            center_score=center_score,
            stability_score=stability_score,
        )

    def _growth_score(self, area_history: list[float]) -> float:
        if len(area_history) < 2:
            return 0.0
        k = min(max(self.config.growth_window, 1), len(area_history) - 1)
        old = area_history[-1 - k]
        cur = area_history[-1]
        growth = (cur - old) / max(old, 1e-9)
        clipped = max(-self.config.growth_clip, min(self.config.growth_clip, growth))
        return clamp01((clipped + self.config.growth_clip) / (2 * self.config.growth_clip))

    def _center_score(self, center: tuple[float, float]) -> float:
        cx, cy = center
        dist = sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        return clamp01(1.0 - min(dist / max(self.config.center_max_dist, 1e-9), 1.0))

    def _stability_score(self, center_jitter: float, area_jitter: float) -> float:
        c = clamp01(1.0 - min(center_jitter / max(self.config.center_jitter_ref, 1e-9), 1.0))
        a = clamp01(1.0 - min(area_jitter / max(self.config.area_jitter_ref, 1e-9), 1.0))
        return 0.5 * (c + a)
