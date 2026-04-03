from __future__ import annotations

from math import sqrt


def ema(current: float, previous: float | None, alpha: float) -> float:
    if previous is None:
        return current
    a = max(0.0, min(1.0, float(alpha)))
    return a * current + (1.0 - a) * previous


def ema_point(
    current: tuple[float, float],
    previous: tuple[float, float] | None,
    alpha: float,
) -> tuple[float, float]:
    if previous is None:
        return current
    return (
        ema(current[0], previous[0], alpha),
        ema(current[1], previous[1], alpha),
    )


def center_jitter(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    return sum(sqrt((p[0] - mx) ** 2 + (p[1] - my) ** 2) for p in points) / len(points)


def area_jitter(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 1e-9:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance**0.5) / mean

