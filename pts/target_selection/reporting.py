from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_event_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)

    rows.sort(key=lambda r: int(r.get("frame_idx", -1)))
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_event_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(str(r.get("event_type", "")) for r in rows)
    acquired = int(event_counts.get("target_acquired", 0))
    lost = int(event_counts.get("target_lost", 0))
    switched = int(event_counts.get("target_switched", 0))

    runs: list[int] = []
    current_target: int | None = None
    run_start: int | None = None
    last_frame: int | None = None

    for r in rows:
        et = str(r.get("event_type", ""))
        frame_idx = _safe_int(r.get("frame_idx"), -1)
        target_id = r.get("track_id")
        if isinstance(target_id, str) and target_id.isdigit():
            target_id = int(target_id)

        if et == "target_acquired":
            if current_target is not None and run_start is not None and frame_idx >= run_start:
                runs.append(frame_idx - run_start)
            current_target = target_id if isinstance(target_id, int) else None
            run_start = frame_idx
        elif et == "target_switched":
            if current_target is not None and run_start is not None and frame_idx >= run_start:
                runs.append(frame_idx - run_start)
            current_target = target_id if isinstance(target_id, int) else None
            run_start = frame_idx
        elif et == "target_lost":
            if current_target is not None and run_start is not None and frame_idx >= run_start:
                runs.append(frame_idx - run_start)
            current_target = None
            run_start = None

        last_frame = frame_idx

    if current_target is not None and run_start is not None and last_frame is not None and last_frame >= run_start:
        runs.append(last_frame - run_start)

    quick_drop_gaps: list[int] = []
    lost_to_next: list[tuple[int, int | None, int | None]] = []
    for i, row in enumerate(rows):
        et = str(row.get("event_type", ""))
        if et in {"target_acquired", "target_switched"}:
            start_frame = _safe_int(row.get("frame_idx"), -1)
            for j in range(i + 1, len(rows)):
                next_row = rows[j]
                next_et = str(next_row.get("event_type", ""))
                if next_et == "target_lost":
                    next_frame = _safe_int(next_row.get("frame_idx"), -1)
                    if start_frame >= 0 and next_frame >= start_frame:
                        quick_drop_gaps.append(next_frame - start_frame)
                    break
                if next_et in {"target_acquired", "target_switched"}:
                    break

        if et == "target_lost":
            lost_frame = _safe_int(row.get("frame_idx"), -1)
            prev_id_val = row.get("previous_track_id")
            prev_id = prev_id_val if isinstance(prev_id_val, int) else None
            if lost_frame < 0:
                continue
            for j in range(i + 1, len(rows)):
                next_row = rows[j]
                next_et = str(next_row.get("event_type", ""))
                if next_et in {"target_acquired", "target_switched"}:
                    next_frame = _safe_int(next_row.get("frame_idx"), -1)
                    next_id_val = next_row.get("track_id")
                    next_id = next_id_val if isinstance(next_id_val, int) else None
                    if next_frame >= lost_frame:
                        lost_to_next.append((next_frame - lost_frame, prev_id, next_id))
                    break

    acquire_like = [r for r in rows if str(r.get("event_type", "")) in {"target_acquired", "target_switched"}]
    lifetimes = [_safe_int(r.get("lifetime"), -1) for r in acquire_like if r.get("lifetime") is not None]
    lifetimes = [v for v in lifetimes if v >= 0]
    scores = [_safe_float(r.get("final_score"), -1.0) for r in acquire_like if r.get("final_score") is not None]
    scores = [v for v in scores if v >= 0.0]

    return {
        "events_total": len(rows),
        "acquired": acquired,
        "lost": lost,
        "switched": switched,
        "target_runs_count": len(runs),
        "target_run_avg_frames": (sum(runs) / len(runs)) if runs else 0.0,
        "target_run_min_frames": min(runs) if runs else 0,
        "target_run_max_frames": max(runs) if runs else 0,
        "flap_le_5_frames": sum(1 for d in quick_drop_gaps if d <= 5),
        "flap_le_10_frames": sum(1 for d in quick_drop_gaps if d <= 10),
        "flap_checks_total": len(quick_drop_gaps),
        "lost_to_next_count": len(lost_to_next),
        "lost_to_next_avg_frames": (sum(d for d, _, _ in lost_to_next) / len(lost_to_next)) if lost_to_next else 0.0,
        "lost_to_next_le_5_frames": sum(1 for d, _, _ in lost_to_next if d <= 5),
        "lost_to_next_same_id": sum(1 for _, prev_id, next_id in lost_to_next if prev_id is not None and prev_id == next_id),
        "acquire_like_count": len(acquire_like),
        "acquire_lifetime_eq3": sum(1 for v in lifetimes if v == 3),
        "acquire_lifetime_avg": (sum(lifetimes) / len(lifetimes)) if lifetimes else 0.0,
        "acquire_score_avg": (sum(scores) / len(scores)) if scores else 0.0,
        "acquire_score_min": min(scores) if scores else 0.0,
        "acquire_score_max": max(scores) if scores else 0.0,
    }


def _flatten_for_csv(file_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"event_file": file_name}
    row.update(metrics)
    return row


def write_event_report(
    event_files: list[Path],
    summary_json_path: Path,
    summary_csv_path: Path,
) -> dict[str, Any]:
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    files_summary: list[dict[str, Any]] = []
    aggregate_counter = Counter()
    aggregate_numeric: Counter[str] = Counter()

    for event_file in event_files:
        rows = load_event_rows(event_file)
        metrics = summarize_event_rows(rows)
        files_summary.append(
            {
                "event_file": str(event_file),
                "metrics": metrics,
            }
        )
        aggregate_counter["events_total"] += int(metrics["events_total"])
        aggregate_counter["acquired"] += int(metrics["acquired"])
        aggregate_counter["lost"] += int(metrics["lost"])
        aggregate_counter["switched"] += int(metrics["switched"])
        aggregate_numeric["target_run_avg_frames"] += float(metrics["target_run_avg_frames"])
        aggregate_numeric["lost_to_next_avg_frames"] += float(metrics["lost_to_next_avg_frames"])
        aggregate_numeric["acquire_lifetime_avg"] += float(metrics["acquire_lifetime_avg"])
        aggregate_numeric["acquire_score_avg"] += float(metrics["acquire_score_avg"])

    files_count = max(len(files_summary), 1)
    aggregate = {
        "files_count": len(files_summary),
        "events_total": int(aggregate_counter["events_total"]),
        "acquired": int(aggregate_counter["acquired"]),
        "lost": int(aggregate_counter["lost"]),
        "switched": int(aggregate_counter["switched"]),
        "target_run_avg_frames": float(aggregate_numeric["target_run_avg_frames"]) / files_count,
        "lost_to_next_avg_frames": float(aggregate_numeric["lost_to_next_avg_frames"]) / files_count,
        "acquire_lifetime_avg": float(aggregate_numeric["acquire_lifetime_avg"]) / files_count,
        "acquire_score_avg": float(aggregate_numeric["acquire_score_avg"]) / files_count,
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": files_summary,
        "aggregate": aggregate,
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_rows = [_flatten_for_csv(Path(item["event_file"]).name, item["metrics"]) for item in files_summary]
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_file"])

    return payload
