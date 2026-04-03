from __future__ import annotations

from ..domain.enums import EventType
from ..domain.models import EventRecord, ScoreBreakdown, TrackCandidate


def make_event_record(
    frame_idx: int,
    timestamp_s: float,
    event_type: EventType,
    candidate: TrackCandidate | None,
    previous_track_id: int | None,
    breakdown: ScoreBreakdown | None,
    reason: str,
) -> EventRecord:
    return EventRecord(
        frame_idx=frame_idx,
        video_time_sec=timestamp_s,
        event_type=event_type,
        track_id=None if candidate is None else candidate.track_id,
        previous_track_id=previous_track_id,
        final_score=None if breakdown is None else breakdown.final_score,
        score_breakdown=None if breakdown is None else breakdown.as_dict(),
        bbox=None if candidate is None else [float(v) for v in candidate.bbox],
        avg_conf=None if candidate is None else float(candidate.avg_conf),
        lifetime=None if candidate is None else int(candidate.lifetime_frames),
        selection_reason=reason,
    )
