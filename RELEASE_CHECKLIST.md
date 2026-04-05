# `pts` 0.1.0 Pre-release Checklist

## 1) Scope freeze
- [ ] No new algorithms added in this release cycle.
- [ ] No tracker backend expansion in core.
- [ ] Core/API changes are backward-compatible.

## 2) Benchmark validation
- [ ] Benchmark run on at least 3-5 videos with mixed scene complexity.
- [ ] Reduced comparison grid used (`botsort|bytetrack`, `single_best|stable_target`, `fast|balanced`).
- [ ] Aggregate report exported (`benchmark_aggregate.csv`, `benchmark_summary.json`, `benchmark_report.md`).

## 3) Interpretation quality
- [ ] Threshold-based interpretation is present in diagnostics output.
- [ ] Recommended defaults are supported by benchmark evidence.
- [ ] Experimental policies are clearly marked as non-default.

## 4) Guardrail impact analysis
- [ ] `policy_clip_ratio` and `external_clip_ratio` reported.
- [ ] Bonus dominance metrics reported (`bonus/policy/external_dominated_ratio`).
- [ ] No persistent excessive clipping in default config runs.

## 5) Documentation
- [ ] README explains problem/solution and scope boundaries.
- [ ] README contains practical default configurations and trade-offs.
- [ ] README includes benchmark and diagnostics commands.

## 6) Packaging and tests
- [ ] `pip install -e .` works in clean environment.
- [ ] `pytest` is green.
- [ ] Optional example dependencies are documented (`.[examples]`).
- [ ] Version and metadata in `pyproject.toml` are ready.

## 7) Release decision
- [ ] Publish `0.1.0` now
- [ ] Or hold release with explicit blockers listed

