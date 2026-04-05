from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pts.target_selection.reporting import load_event_rows, summarize_event_rows


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _collect_event_files(source: Path, glob_pattern: str) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Events source not found: {source}")
    files = sorted(p for p in source.rglob(glob_pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No event files found in {source} by glob '{glob_pattern}'")
    return files


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _flatten_file_metrics(file_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_file": file_name,
        "events_total": _safe_int(metrics.get("events_total"), 0),
        "acquired": _safe_int(metrics.get("acquired"), 0),
        "lost": _safe_int(metrics.get("lost"), 0),
        "switched": _safe_int(metrics.get("switched"), 0),
        "candidate_rejected": _safe_int(metrics.get("candidate_rejected"), 0),
        "target_run_avg_frames": _safe_float(metrics.get("target_run_avg_frames"), 0.0),
        "primary_target_presence_ratio": _safe_float(metrics.get("primary_target_presence_ratio"), 0.0),
        "primary_target_stability_ratio": _safe_float(metrics.get("primary_target_stability_ratio"), 0.0),
        "flap_ratio": (
            _safe_int(metrics.get("flap_le_5_frames"), 0) / max(_safe_int(metrics.get("flap_checks_total"), 0), 1)
        ),
        "avg_policy_contrib": _safe_float(metrics.get("avg_policy_contrib"), 0.0),
        "avg_external_contrib": _safe_float(metrics.get("avg_external_contrib"), 0.0),
        "bonus_dominated_count": _safe_int(metrics.get("bonus_dominated_count"), 0),
        "policy_dominated_count": _safe_int(metrics.get("policy_dominated_count"), 0),
        "external_dominated_count": _safe_int(metrics.get("external_dominated_count"), 0),
        "policy_clip_count": _safe_int(metrics.get("policy_clip_count"), 0),
        "external_clip_count": _safe_int(metrics.get("external_clip_count"), 0),
        "total_bonus_clip_count": _safe_int(metrics.get("total_bonus_clip_count"), 0),
        "policy_clip_ratio": _safe_float(metrics.get("policy_clip_ratio"), 0.0),
        "external_clip_ratio": _safe_float(metrics.get("external_clip_ratio"), 0.0),
        "total_bonus_clip_ratio": _safe_float(metrics.get("total_bonus_clip_ratio"), 0.0),
        "avg_policy_clip_abs": _safe_float(metrics.get("avg_policy_clip_abs"), 0.0),
        "avg_external_clip_abs": _safe_float(metrics.get("avg_external_clip_abs"), 0.0),
        "avg_total_bonus_clip_abs": _safe_float(metrics.get("avg_total_bonus_clip_abs"), 0.0),
        "score_rows_count": _safe_int(metrics.get("score_rows_count"), 0),
    }


def _diagnose_failure_modes(
    files_summary: list[dict[str, Any]],
    switch_threshold: int,
    flap_ratio_threshold: float,
    presence_threshold: float,
) -> dict[str, Any]:
    high_switch_files: list[str] = []
    high_flap_files: list[str] = []
    low_presence_files: list[str] = []
    dominant_reject_reason_files: list[dict[str, Any]] = []

    for item in files_summary:
        file_name = str(item["event_file"])
        metrics = item["metrics"]
        switched = _safe_int(metrics.get("switched"), 0)
        flap_checks = _safe_int(metrics.get("flap_checks_total"), 0)
        flap_le_5 = _safe_int(metrics.get("flap_le_5_frames"), 0)
        flap_ratio = (flap_le_5 / flap_checks) if flap_checks > 0 else 0.0
        presence = _safe_float(metrics.get("primary_target_presence_ratio"), 0.0)

        if switched >= switch_threshold:
            high_switch_files.append(file_name)
        if flap_ratio >= flap_ratio_threshold:
            high_flap_files.append(file_name)
        if presence <= presence_threshold:
            low_presence_files.append(file_name)

        reject_counts = metrics.get("reject_reason_counts", {})
        if isinstance(reject_counts, dict) and reject_counts:
            total = sum(_safe_int(v, 0) for v in reject_counts.values())
            if total > 0:
                top_reason, top_count = max(reject_counts.items(), key=lambda kv: _safe_int(kv[1], 0))
                top_share = _safe_int(top_count, 0) / total
                if top_share >= 0.70:
                    dominant_reject_reason_files.append(
                        {
                            "event_file": file_name,
                            "reason": str(top_reason),
                            "share": top_share,
                            "count": _safe_int(top_count, 0),
                            "total_rejects": total,
                        }
                    )

    return {
        "high_switch_files": high_switch_files,
        "high_flap_files": high_flap_files,
        "low_presence_files": low_presence_files,
        "dominant_reject_reason_files": dominant_reject_reason_files,
    }


def _build_interpretation(
    aggregate: dict[str, Any],
    *,
    max_bonus_dominated_ratio: float,
    max_policy_dominated_ratio: float,
    max_external_dominated_ratio: float,
    max_policy_clip_ratio: float,
    max_external_clip_ratio: float,
    min_presence_ratio: float,
    min_stability_ratio: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _add_check(metric: str, value: float, threshold: float, direction: str, message: str) -> None:
        if direction == "max":
            passed = value <= threshold
            expr = f"{metric} <= {threshold:.3f}"
        else:
            passed = value >= threshold
            expr = f"{metric} >= {threshold:.3f}"
        checks.append(
            {
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": direction,
                "passed": passed,
                "rule": expr,
                "message": message if not passed else "ok",
            }
        )

    _add_check(
        metric="bonus_dominated_ratio",
        value=_safe_float(aggregate.get("bonus_dominated_ratio"), 0.0),
        threshold=max_bonus_dominated_ratio,
        direction="max",
        message="Base score influence is weak; reduce policy/external aggressiveness or tighten guardrails.",
    )
    _add_check(
        metric="policy_dominated_ratio",
        value=_safe_float(aggregate.get("policy_dominated_ratio"), 0.0),
        threshold=max_policy_dominated_ratio,
        direction="max",
        message="Policy dominates too often; reduce policy_strength or tune policy choice.",
    )
    _add_check(
        metric="external_dominated_ratio",
        value=_safe_float(aggregate.get("external_dominated_ratio"), 0.0),
        threshold=max_external_dominated_ratio,
        direction="max",
        message="External signals dominate too often; reduce external scales/biases.",
    )
    _add_check(
        metric="policy_clip_ratio",
        value=_safe_float(aggregate.get("policy_clip_ratio"), 0.0),
        threshold=max_policy_clip_ratio,
        direction="max",
        message="Policy guardrail clipping happens too often; reduce policy strength.",
    )
    _add_check(
        metric="external_clip_ratio",
        value=_safe_float(aggregate.get("external_clip_ratio"), 0.0),
        threshold=max_external_clip_ratio,
        direction="max",
        message="External guardrail clipping happens too often; reduce external biases/hints.",
    )
    _add_check(
        metric="primary_target_presence_ratio",
        value=_safe_float(aggregate.get("primary_target_presence_ratio"), 0.0),
        threshold=min_presence_ratio,
        direction="min",
        message="Too many no-target windows; relax acquire constraints or improve tracker stability.",
    )
    _add_check(
        metric="primary_target_stability_ratio",
        value=_safe_float(aggregate.get("primary_target_stability_ratio"), 0.0),
        threshold=min_stability_ratio,
        direction="min",
        message="Target lock is unstable; increase hysteresis/switch margin or use stable policy.",
    )

    passed = all(bool(c.get("passed")) for c in checks)
    failed = [c for c in checks if not c.get("passed")]
    return {
        "status": "PASS" if passed else "WARN",
        "checks": checks,
        "failed_checks": failed,
    }


def _build_human_conclusions(
    aggregate: dict[str, Any],
    failure_modes: dict[str, Any],
    interpretation: dict[str, Any],
) -> list[str]:
    conclusions: list[str] = []
    status = str(interpretation.get("status", "WARN"))
    if status == "PASS":
        conclusions.append("Selection behavior is within configured thresholds.")
    else:
        conclusions.append("Selection behavior violates at least one threshold; inspect failed checks.")

    switch_files = len(failure_modes.get("high_switch_files", []))
    flap_files = len(failure_modes.get("high_flap_files", []))
    low_presence_files = len(failure_modes.get("low_presence_files", []))
    if switch_files > 0:
        conclusions.append(f"High switching detected in {switch_files} file(s).")
    if flap_files > 0:
        conclusions.append(f"Potential flap behavior detected in {flap_files} file(s).")
    if low_presence_files > 0:
        conclusions.append(f"Low primary-target presence detected in {low_presence_files} file(s).")

    policy_dom = _safe_float(aggregate.get("policy_dominated_ratio"), 0.0)
    external_dom = _safe_float(aggregate.get("external_dominated_ratio"), 0.0)
    if policy_dom > 0.40:
        conclusions.append("Policy influence is too strong; reduce policy strength or switch policy.")
    if external_dom > 0.50:
        conclusions.append("External influence is too strong; reduce external biases/hints.")

    if not conclusions:
        conclusions.append("No significant issues detected.")
    return conclusions


def _aggregate(files_summary: list[dict[str, Any]]) -> dict[str, Any]:
    if not files_summary:
        return {"files_count": 0}

    event_counter = Counter()
    reason_counter = Counter()
    reject_reason_counter = Counter()
    scalar_sum = Counter()

    for item in files_summary:
        metrics = item["metrics"]
        event_counter["events_total"] += _safe_int(metrics.get("events_total"), 0)
        event_counter["acquired"] += _safe_int(metrics.get("acquired"), 0)
        event_counter["lost"] += _safe_int(metrics.get("lost"), 0)
        event_counter["switched"] += _safe_int(metrics.get("switched"), 0)
        event_counter["candidate_rejected"] += _safe_int(metrics.get("candidate_rejected"), 0)

        scalar_sum["target_run_avg_frames"] += _safe_float(metrics.get("target_run_avg_frames"), 0.0)
        scalar_sum["presence"] += _safe_float(metrics.get("primary_target_presence_ratio"), 0.0)
        scalar_sum["stability"] += _safe_float(metrics.get("primary_target_stability_ratio"), 0.0)
        scalar_sum["policy_contrib"] += _safe_float(metrics.get("avg_policy_contrib"), 0.0)
        scalar_sum["external_contrib"] += _safe_float(metrics.get("avg_external_contrib"), 0.0)
        scalar_sum["bonus_dominated_count"] += _safe_int(metrics.get("bonus_dominated_count"), 0)
        scalar_sum["policy_dominated_count"] += _safe_int(metrics.get("policy_dominated_count"), 0)
        scalar_sum["external_dominated_count"] += _safe_int(metrics.get("external_dominated_count"), 0)
        scalar_sum["policy_clip_count"] += _safe_int(metrics.get("policy_clip_count"), 0)
        scalar_sum["external_clip_count"] += _safe_int(metrics.get("external_clip_count"), 0)
        scalar_sum["total_bonus_clip_count"] += _safe_int(metrics.get("total_bonus_clip_count"), 0)
        scalar_sum["avg_policy_clip_abs"] += _safe_float(metrics.get("avg_policy_clip_abs"), 0.0)
        scalar_sum["avg_external_clip_abs"] += _safe_float(metrics.get("avg_external_clip_abs"), 0.0)
        scalar_sum["avg_total_bonus_clip_abs"] += _safe_float(metrics.get("avg_total_bonus_clip_abs"), 0.0)
        scalar_sum["score_rows_count"] += _safe_int(metrics.get("score_rows_count"), 0)

        selection_reason_counts = metrics.get("selection_reason_counts")
        if isinstance(selection_reason_counts, dict):
            for reason, count in selection_reason_counts.items():
                reason_counter[str(reason)] += _safe_int(count, 0)

        reject_reason_counts = metrics.get("reject_reason_counts")
        if isinstance(reject_reason_counts, dict):
            for reason, count in reject_reason_counts.items():
                reject_reason_counter[str(reason)] += _safe_int(count, 0)

    files_count = len(files_summary)
    score_rows_count = max(int(scalar_sum["score_rows_count"]), 1)
    return {
        "files_count": files_count,
        "events_total": int(event_counter["events_total"]),
        "acquired": int(event_counter["acquired"]),
        "lost": int(event_counter["lost"]),
        "switched": int(event_counter["switched"]),
        "candidate_rejected": int(event_counter["candidate_rejected"]),
        "target_run_avg_frames": float(scalar_sum["target_run_avg_frames"]) / files_count,
        "primary_target_presence_ratio": float(scalar_sum["presence"]) / files_count,
        "primary_target_stability_ratio": float(scalar_sum["stability"]) / files_count,
        "avg_policy_contrib": float(scalar_sum["policy_contrib"]) / files_count,
        "avg_external_contrib": float(scalar_sum["external_contrib"]) / files_count,
        "bonus_dominated_ratio": float(scalar_sum["bonus_dominated_count"]) / score_rows_count,
        "policy_dominated_ratio": float(scalar_sum["policy_dominated_count"]) / score_rows_count,
        "external_dominated_ratio": float(scalar_sum["external_dominated_count"]) / score_rows_count,
        "policy_clip_ratio": float(scalar_sum["policy_clip_count"]) / score_rows_count,
        "external_clip_ratio": float(scalar_sum["external_clip_count"]) / score_rows_count,
        "total_bonus_clip_ratio": float(scalar_sum["total_bonus_clip_count"]) / score_rows_count,
        "avg_policy_clip_abs": float(scalar_sum["avg_policy_clip_abs"]) / files_count,
        "avg_external_clip_abs": float(scalar_sum["avg_external_clip_abs"]) / files_count,
        "avg_total_bonus_clip_abs": float(scalar_sum["avg_total_bonus_clip_abs"]) / files_count,
        "selection_reason_counts": dict(reason_counter),
        "reject_reason_counts": dict(reject_reason_counter),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_file"])
        return

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selection diagnostics summary for pts event logs.")
    parser.add_argument("--events", type=str, default="data/output", help="Event file or directory with JSONL events.")
    parser.add_argument("--glob", type=str, default="*_events.jsonl", help="Glob to find event files in directory mode.")
    parser.add_argument("--output-json", type=str, default="reports/selection_diagnostics.json")
    parser.add_argument("--output-csv", type=str, default="reports/selection_diagnostics.csv")
    parser.add_argument("--switch-threshold", type=int, default=20, help="Warn when switched events per file >= threshold.")
    parser.add_argument("--flap-ratio-threshold", type=float, default=0.25, help="Warn when flap ratio >= threshold.")
    parser.add_argument("--presence-threshold", type=float, default=0.25, help="Warn when presence ratio <= threshold.")
    parser.add_argument("--max-bonus-dominated-ratio", type=float, default=0.40)
    parser.add_argument("--max-policy-dominated-ratio", type=float, default=0.40)
    parser.add_argument("--max-external-dominated-ratio", type=float, default=0.50)
    parser.add_argument("--max-policy-clip-ratio", type=float, default=0.50)
    parser.add_argument("--max-external-clip-ratio", type=float, default=0.50)
    parser.add_argument("--min-presence-ratio", type=float, default=0.30)
    parser.add_argument("--min-stability-ratio", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve(args.events)
    files = _collect_event_files(source, args.glob)

    files_summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for file_path in files:
        rows = load_event_rows(file_path)
        metrics = summarize_event_rows(rows)
        files_summary.append({"event_file": str(file_path), "metrics": metrics})
        csv_rows.append(_flatten_file_metrics(file_path.name, metrics))

    aggregate = _aggregate(files_summary)
    failure_modes = _diagnose_failure_modes(
        files_summary=files_summary,
        switch_threshold=max(1, int(args.switch_threshold)),
        flap_ratio_threshold=float(args.flap_ratio_threshold),
        presence_threshold=float(args.presence_threshold),
    )
    interpretation = _build_interpretation(
        aggregate=aggregate,
        max_bonus_dominated_ratio=float(args.max_bonus_dominated_ratio),
        max_policy_dominated_ratio=float(args.max_policy_dominated_ratio),
        max_external_dominated_ratio=float(args.max_external_dominated_ratio),
        max_policy_clip_ratio=float(args.max_policy_clip_ratio),
        max_external_clip_ratio=float(args.max_external_clip_ratio),
        min_presence_ratio=float(args.min_presence_ratio),
        min_stability_ratio=float(args.min_stability_ratio),
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_source": str(source),
        "glob": str(args.glob),
        "interpretation_thresholds": {
            "max_bonus_dominated_ratio": float(args.max_bonus_dominated_ratio),
            "max_policy_dominated_ratio": float(args.max_policy_dominated_ratio),
            "max_external_dominated_ratio": float(args.max_external_dominated_ratio),
            "max_policy_clip_ratio": float(args.max_policy_clip_ratio),
            "max_external_clip_ratio": float(args.max_external_clip_ratio),
            "min_presence_ratio": float(args.min_presence_ratio),
            "min_stability_ratio": float(args.min_stability_ratio),
        },
        "files_count": len(files_summary),
        "aggregate": aggregate,
        "failure_modes": failure_modes,
        "interpretation": interpretation,
        "conclusions": _build_human_conclusions(aggregate, failure_modes, interpretation),
        "files": files_summary,
    }

    output_json = _resolve(args.output_json)
    output_csv = _resolve(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_csv, csv_rows)

    print(f"[DIAG] files={len(files_summary)}")
    print(f"[DIAG] JSON: {output_json}")
    print(f"[DIAG] CSV:  {output_csv}")
    print(f"[DIAG] switched={aggregate.get('switched', 0)} lost={aggregate.get('lost', 0)}")
    print(
        f"[DIAG] presence={_safe_float(aggregate.get('primary_target_presence_ratio', 0.0)):.3f} "
        f"stability={_safe_float(aggregate.get('primary_target_stability_ratio', 0.0)):.3f}"
    )
    print(f"[DIAG] interpretation={interpretation.get('status')}")
    for line in payload["conclusions"][:3]:
        print(f"[DIAG] conclusion: {line}")


if __name__ == "__main__":
    main()
