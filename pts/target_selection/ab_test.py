from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class GateThresholds:
    max_switched_increase_pct: float = 10.0
    max_lost_increase_pct: float = 10.0
    max_flap_ratio_increase_pct: float = 10.0
    max_run_avg_drop_pct: float = 5.0
    max_proc_fps_drop_pct: float = 20.0
    max_flap_ratio_if_baseline_zero: float = 0.02


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


def _resolve_manifest_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        candidate = path / "run_manifest.json"
        if not candidate.exists():
            raise FileNotFoundError(f"run_manifest.json not found in: {path}")
        return candidate.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_events_summary_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any] | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    summary_path_raw = artifacts.get("events_summary_json")
    if not summary_path_raw:
        return None

    summary_path = Path(str(summary_path_raw))
    if not summary_path.is_absolute():
        summary_path = (manifest_path.parent / summary_path).resolve()
    if not summary_path.exists():
        return None

    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    return summary


def _aggregate_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    files = summary.get("files")
    if not isinstance(files, list):
        return {}

    flap_le_5_total = 0
    flap_checks_total = 0
    run_weighted_sum = 0.0
    run_count_total = 0

    for item in files:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            continue
        flap_le_5_total += _safe_int(metrics.get("flap_le_5_frames"), 0)
        flap_checks_total += _safe_int(metrics.get("flap_checks_total"), 0)
        run_count = _safe_int(metrics.get("target_runs_count"), 0)
        run_avg = _safe_float(metrics.get("target_run_avg_frames"), 0.0)
        run_weighted_sum += run_avg * run_count
        run_count_total += run_count

    run_avg_weighted = (run_weighted_sum / run_count_total) if run_count_total > 0 else 0.0
    flap_ratio = (flap_le_5_total / flap_checks_total) if flap_checks_total > 0 else 0.0
    return {
        "flap_le_5_total": float(flap_le_5_total),
        "flap_checks_total": float(flap_checks_total),
        "flap_ratio": float(flap_ratio),
        "run_avg_weighted": float(run_avg_weighted),
        "target_runs_total": float(run_count_total),
    }


def load_run_snapshot(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    totals = manifest.get("totals") if isinstance(manifest.get("totals"), dict) else {}
    events_summary = _read_events_summary_from_manifest(manifest, manifest_path)
    summary_agg = _aggregate_from_summary(events_summary or {})

    frames_total = _safe_int(totals.get("frames_total"), 0)
    switched = _safe_int(totals.get("target_switched"), 0)
    lost = _safe_int(totals.get("target_lost"), 0)
    avg_processing_fps = _safe_float(totals.get("avg_processing_fps"), 0.0)

    switched_per_1k = (1000.0 * switched / frames_total) if frames_total > 0 else 0.0
    lost_per_1k = (1000.0 * lost / frames_total) if frames_total > 0 else 0.0

    snapshot = {
        "manifest_path": str(manifest_path),
        "run_id": str(manifest.get("run_id", manifest_path.parent.name)),
        "created_at": str(manifest.get("created_at", "")),
        "weights": str(manifest.get("weights", "")),
        "device": str(manifest.get("device", "")),
        "frames_total": frames_total,
        "target_switched": switched,
        "target_lost": lost,
        "switched_per_1k_frames": switched_per_1k,
        "lost_per_1k_frames": lost_per_1k,
        "avg_processing_fps": avg_processing_fps,
        "flap_ratio": _safe_float(summary_agg.get("flap_ratio"), 0.0),
        "flap_le_5_total": _safe_int(summary_agg.get("flap_le_5_total"), 0),
        "flap_checks_total": _safe_int(summary_agg.get("flap_checks_total"), 0),
        "target_run_avg_frames": _safe_float(summary_agg.get("run_avg_weighted"), 0.0),
        "target_runs_total": _safe_int(summary_agg.get("target_runs_total"), 0),
    }
    return snapshot


def _rel_change_pct(baseline: float, candidate: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0 if abs(candidate) < 1e-12 else float("inf")
    return 100.0 * ((candidate - baseline) / baseline)


def _check_upper_is_better(
    metric: str,
    baseline: float,
    candidate: float,
    max_increase_pct: float,
    note: str = "",
) -> dict[str, Any]:
    allowed = baseline * (1.0 + max_increase_pct / 100.0)
    passed = candidate <= allowed + 1e-12
    return {
        "metric": metric,
        "direction": "lower_is_better",
        "baseline": baseline,
        "candidate": candidate,
        "delta_pct": _rel_change_pct(baseline, candidate),
        "threshold": f"candidate <= baseline * (1 + {max_increase_pct:.2f}%)",
        "passed": passed,
        "note": note,
    }


def _check_lower_bound(
    metric: str,
    baseline: float,
    candidate: float,
    max_drop_pct: float,
    note: str = "",
) -> dict[str, Any]:
    allowed = baseline * (1.0 - max_drop_pct / 100.0)
    passed = candidate + 1e-12 >= allowed
    return {
        "metric": metric,
        "direction": "higher_is_better",
        "baseline": baseline,
        "candidate": candidate,
        "delta_pct": _rel_change_pct(baseline, candidate),
        "threshold": f"candidate >= baseline * (1 - {max_drop_pct:.2f}%)",
        "passed": passed,
        "note": note,
    }


def run_ab_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    checks.append(
        _check_upper_is_better(
            metric="switched_per_1k_frames",
            baseline=_safe_float(baseline.get("switched_per_1k_frames")),
            candidate=_safe_float(candidate.get("switched_per_1k_frames")),
            max_increase_pct=thresholds.max_switched_increase_pct,
        )
    )
    checks.append(
        _check_upper_is_better(
            metric="lost_per_1k_frames",
            baseline=_safe_float(baseline.get("lost_per_1k_frames")),
            candidate=_safe_float(candidate.get("lost_per_1k_frames")),
            max_increase_pct=thresholds.max_lost_increase_pct,
        )
    )

    base_flap_ratio = _safe_float(baseline.get("flap_ratio"), 0.0)
    cand_flap_ratio = _safe_float(candidate.get("flap_ratio"), 0.0)
    if base_flap_ratio <= 1e-12:
        checks.append(
            {
                "metric": "flap_ratio",
                "direction": "lower_is_better",
                "baseline": base_flap_ratio,
                "candidate": cand_flap_ratio,
                "delta_pct": _rel_change_pct(base_flap_ratio, cand_flap_ratio),
                "threshold": f"candidate <= {thresholds.max_flap_ratio_if_baseline_zero:.4f} when baseline is 0",
                "passed": cand_flap_ratio <= thresholds.max_flap_ratio_if_baseline_zero + 1e-12,
                "note": "baseline flap ratio is zero; absolute threshold applied",
            }
        )
    else:
        checks.append(
            _check_upper_is_better(
                metric="flap_ratio",
                baseline=base_flap_ratio,
                candidate=cand_flap_ratio,
                max_increase_pct=thresholds.max_flap_ratio_increase_pct,
            )
        )

    checks.append(
        _check_lower_bound(
            metric="target_run_avg_frames",
            baseline=_safe_float(baseline.get("target_run_avg_frames")),
            candidate=_safe_float(candidate.get("target_run_avg_frames")),
            max_drop_pct=thresholds.max_run_avg_drop_pct,
        )
    )
    checks.append(
        _check_lower_bound(
            metric="avg_processing_fps",
            baseline=_safe_float(baseline.get("avg_processing_fps")),
            candidate=_safe_float(candidate.get("avg_processing_fps")),
            max_drop_pct=thresholds.max_proc_fps_drop_pct,
        )
    )

    passed_all = all(bool(c.get("passed")) for c in checks)
    return {
        "passed": passed_all,
        "checks": checks,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, checks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not checks:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"])
        return

    fields = ["metric", "direction", "baseline", "candidate", "delta_pct", "threshold", "passed", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in checks:
            writer.writerow({k: c.get(k, "") for k in fields})


def _fmt_pct(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:+.2f}%"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    gate = payload.get("gate", {})
    checks = gate.get("checks", [])
    status = "PASS" if gate.get("passed") else "FAIL"

    lines: list[str] = []
    lines.append(f"# A/B Quality Gate: {status}")
    lines.append("")
    lines.append(f"- Generated at: `{payload.get('generated_at', '')}`")
    lines.append(f"- Baseline run: `{baseline.get('run_id', '')}`")
    lines.append(f"- Candidate run: `{candidate.get('run_id', '')}`")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| Run | Frames | Switched | Lost | Switch/1k | Lost/1k | Flap ratio | Run avg (frames) | Proc FPS |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Baseline | {baseline.get('frames_total', 0)} | {baseline.get('target_switched', 0)} | "
        f"{baseline.get('target_lost', 0)} | {baseline.get('switched_per_1k_frames', 0.0):.3f} | "
        f"{baseline.get('lost_per_1k_frames', 0.0):.3f} | {baseline.get('flap_ratio', 0.0):.4f} | "
        f"{baseline.get('target_run_avg_frames', 0.0):.2f} | {baseline.get('avg_processing_fps', 0.0):.2f} |"
    )
    lines.append(
        f"| Candidate | {candidate.get('frames_total', 0)} | {candidate.get('target_switched', 0)} | "
        f"{candidate.get('target_lost', 0)} | {candidate.get('switched_per_1k_frames', 0.0):.3f} | "
        f"{candidate.get('lost_per_1k_frames', 0.0):.3f} | {candidate.get('flap_ratio', 0.0):.4f} | "
        f"{candidate.get('target_run_avg_frames', 0.0):.2f} | {candidate.get('avg_processing_fps', 0.0):.2f} |"
    )
    lines.append("")
    lines.append("## Gate Checks")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta | Threshold | Status |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for c in checks:
        metric = c.get("metric", "")
        base = _safe_float(c.get("baseline"), 0.0)
        cand = _safe_float(c.get("candidate"), 0.0)
        delta = _fmt_pct(_safe_float(c.get("delta_pct"), 0.0))
        threshold = str(c.get("threshold", ""))
        state = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"| `{metric}` | {base:.6f} | {cand:.6f} | {delta} | {threshold} | **{state}** |")
    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two inference runs and apply quality gate checks.")
    parser.add_argument("--baseline", required=True, help="Path to baseline run_manifest.json or run directory.")
    parser.add_argument("--candidate", required=True, help="Path to candidate run_manifest.json or run directory.")
    parser.add_argument("--output-dir", default="", help="Where to save comparison artifacts. Default: candidate reports dir.")
    parser.add_argument("--name", default="", help="Optional suffix for output file names.")

    parser.add_argument("--max-switched-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-lost-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-flap-ratio-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-run-avg-drop-pct", type=float, default=5.0)
    parser.add_argument("--max-proc-fps-drop-pct", type=float, default=20.0)
    parser.add_argument("--max-flap-ratio-if-baseline-zero", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_manifest_path = _resolve_manifest_path(args.baseline)
    candidate_manifest_path = _resolve_manifest_path(args.candidate)

    baseline_snapshot = load_run_snapshot(baseline_manifest_path)
    candidate_snapshot = load_run_snapshot(candidate_manifest_path)

    thresholds = GateThresholds(
        max_switched_increase_pct=float(args.max_switched_increase_pct),
        max_lost_increase_pct=float(args.max_lost_increase_pct),
        max_flap_ratio_increase_pct=float(args.max_flap_ratio_increase_pct),
        max_run_avg_drop_pct=float(args.max_run_avg_drop_pct),
        max_proc_fps_drop_pct=float(args.max_proc_fps_drop_pct),
        max_flap_ratio_if_baseline_zero=float(args.max_flap_ratio_if_baseline_zero),
    )
    gate = run_ab_gate(baseline_snapshot, candidate_snapshot, thresholds)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_manifest": str(baseline_manifest_path),
        "candidate_manifest": str(candidate_manifest_path),
        "baseline": baseline_snapshot,
        "candidate": candidate_snapshot,
        "thresholds": thresholds.__dict__,
        "gate": gate,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = candidate_manifest_path.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.name.strip()}" if str(args.name).strip() else ""
    json_path = output_dir / f"ab_quality_gate{suffix}.json"
    csv_path = output_dir / f"ab_quality_gate{suffix}.csv"
    md_path = output_dir / f"ab_quality_gate{suffix}.md"

    _write_json(json_path, payload)
    _write_csv(csv_path, gate.get("checks", []))
    _write_markdown(md_path, payload)

    status = "PASS" if gate.get("passed") else "FAIL"
    print(f"[A/B] Status: {status}")
    print(f"[A/B] Baseline: {baseline_snapshot.get('run_id')} | Candidate: {candidate_snapshot.get('run_id')}")
    print(f"[A/B] JSON: {json_path}")
    print(f"[A/B] CSV:  {csv_path}")
    print(f"[A/B] MD:   {md_path}")

    for check in gate.get("checks", []):
        mark = "PASS" if check.get("passed") else "FAIL"
        delta = _fmt_pct(_safe_float(check.get("delta_pct"), 0.0))
        print(
            f"  - [{mark}] {check.get('metric')}: baseline={_safe_float(check.get('baseline')):.6f}, "
            f"candidate={_safe_float(check.get('candidate')):.6f}, delta={delta}"
        )

    if not gate.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
