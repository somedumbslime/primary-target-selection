from __future__ import annotations

import math

from pts.target_selection.reporting import summarize_event_rows


def _row(
    frame_idx: int,
    event_type: str,
    selection_reason: str,
    track_id: int | None = None,
    score_breakdown: dict | None = None,
) -> dict:
    return {
        "frame_idx": frame_idx,
        "video_time_sec": frame_idx / 30.0,
        "event_type": event_type,
        "track_id": track_id,
        "selection_reason": selection_reason,
        "score_breakdown": score_breakdown,
    }


def test_summarize_event_rows_extended_metrics() -> None:
    rows = [
        _row(
            frame_idx=2,
            event_type="target_acquired",
            selection_reason="initial_acquire",
            track_id=10,
            score_breakdown={
                "conf_contrib": 0.2,
                "lifetime_contrib": 0.1,
                "area_contrib": 0.1,
                "growth_contrib": 0.0,
                "center_contrib": 0.1,
                "stability_contrib": 0.1,
                "policy_contrib": 0.05,
                "external_contrib": 0.02,
                "policy_clip_applied": 1.0,
                "external_clip_applied": 0.0,
                "total_bonus_clip_applied": 1.0,
                "policy_clip_abs": 0.01,
                "external_clip_abs": 0.0,
                "total_bonus_clip_abs": 0.02,
                "final_score": 0.67,
            },
        ),
        _row(
            frame_idx=10,
            event_type="target_switched",
            selection_reason="switched_to_better_candidate",
            track_id=12,
            score_breakdown={
                "conf_contrib": 0.18,
                "lifetime_contrib": 0.11,
                "area_contrib": 0.08,
                "growth_contrib": 0.03,
                "center_contrib": 0.09,
                "stability_contrib": 0.11,
                "policy_contrib": 0.04,
                "external_contrib": 0.01,
                "policy_clip_applied": 0.0,
                "external_clip_applied": 1.0,
                "total_bonus_clip_applied": 0.0,
                "policy_clip_abs": 0.0,
                "external_clip_abs": 0.03,
                "total_bonus_clip_abs": 0.0,
                "final_score": 0.65,
            },
        ),
        _row(
            frame_idx=20,
            event_type="target_lost",
            selection_reason="target_lost",
            track_id=None,
        ),
        _row(
            frame_idx=22,
            event_type="candidate_rejected",
            selection_reason="candidate_rejected_low_conf",
            track_id=42,
        ),
    ]

    summary = summarize_event_rows(rows, frames_total=30)

    assert summary["acquired"] == 1
    assert summary["switched"] == 1
    assert summary["lost"] == 1
    assert summary["candidate_rejected"] == 1
    assert summary["target_acquisition_delay_frames"] == 2
    assert math.isclose(summary["primary_target_presence_ratio"], 18 / 30, rel_tol=1e-9)
    assert math.isclose(summary["primary_target_stability_ratio"], 0.5, rel_tol=1e-9)
    assert summary["target_run_max_frames"] == 10
    assert summary["selection_reason_counts"]["initial_acquire"] == 1
    assert summary["reject_reason_counts"]["candidate_rejected_low_conf"] == 1
    assert summary["score_rows_count"] == 2
    assert summary["avg_policy_contrib"] > 0.0
    assert summary["avg_external_contrib"] > 0.0
    assert summary["policy_clip_count"] == 1
    assert summary["external_clip_count"] == 1
    assert summary["total_bonus_clip_count"] == 1
    assert summary["policy_clip_ratio"] > 0.0
    assert summary["external_clip_ratio"] > 0.0
