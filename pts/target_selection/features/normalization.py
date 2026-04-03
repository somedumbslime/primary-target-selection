from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def minmax(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def ratio(value: float, ref: float) -> float:
    if ref <= 1e-9:
        return 0.0
    return clamp01(value / ref)


def inverse_ratio(value: float, ref: float) -> float:
    if ref <= 1e-9:
        return 0.0
    return clamp01(1.0 - min(value / ref, 1.0))

