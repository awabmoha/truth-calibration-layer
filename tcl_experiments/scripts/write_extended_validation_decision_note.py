from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


READY_DECISION_MAP = {
    "supports_continuing_tcl_v0": {
        "required_choice": "continue TCL-v0 research",
        "final_status": "ready",
        "interpretation": (
            "The completed metric and manual-review gate supports continuing TCL-v0 research. "
            "This does not by itself justify claiming full TCL validation."
        ),
    },
    "mixed_continue_cautiously": {
        "required_choice": "continue TCL-v0 research",
        "final_status": "ready",
        "interpretation": (
            "The completed gate shows mixed but useful TCL-v0 signal. Continue TCL-v0 research cautiously "
            "rather than moving directly to TCL-v1."
        ),
    },
    "revise_or_weaken_hypothesis": {
        "required_choice": "revise or weaken the TCL hypothesis",
        "final_status": "ready",
        "interpretation": (
            "The completed gate does not provide enough support for the current TCL-v0 hypothesis. "
            "The hidden-state-probe path should be revised or weakened before larger claims."
        ),
    },
}

NOT_READY_REASONS = {
    "incomplete": "The planned model/benchmark matrix is incomplete.",
    "manual_review_pending": "Targeted manual review is incomplete for one or more runs.",
    "manual_review_changes_require_reanalysis": (
        "Manual review marked at least one aggregate-impacting case; reviewed-label metrics must be recomputed "
        "or inspected before a final decision."
    ),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_rows(report: dict):
    rows = []
    for run in report.get("runs", []):
        conservative = run.get("signals", {}).get("tcl_v0_conservative_confidence", {})
        rows.append({
            "run_id": run.get("run_id", ""),
            "model": run.get("model_name") or run.get("model_key") or run.get("model_family", ""),
            "benchmark": run.get("benchmark", ""),
            "records": run.get("n_records", ""),
            "test": run.get("n_test", ""),
            "review": run.get("targeted_manual_review", {}).get("status", "missing"),
            "primary_wins": conservative.get("primary_wins", ""),
            "primary_losses": conservative.get("primary_losses", ""),
            "primary_ties": conservative.get("primary_ties", ""),
        })
    return rows


def build_note(report: dict):
    if "decision_gate" not in report:
        raise ValueError("Decision JSON is missing required key: decision_gate")
    gate = report["decision_gate"]
    if "decision" not in gate:
        raise ValueError("Decision gate is missing required key: decision")
    decision = gate["decision"]
    ready = READY_DECISION_MAP.get(decision)
    final_status = ready["final_status"] if ready else "not_ready"
    required_choice = ready["required_choice"] if ready else "no final choice yet"
    interpretation = ready["interpretation"] if ready else NOT_READY_REASONS.get(decision, gate.get("rationale", "Not ready."))

    lines = [
        "# TCL-v0 Extended Validation Final Decision Note",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Decision",
        "",
        f"- Status: `{final_status}`",
        f"- Aggregator decision: `{decision}`",
        f"- Required stopping-rule choice: `{required_choice}`",
        "",
        interpretation,
        "",
        "## Gate Summary",
        "",
        f"- Matrix complete: {gate.get('matrix_complete')}",
        f"- Completed runs: {gate.get('completed_runs')}",
        f"- Manual-review completed runs: {gate.get('manual_review_completed_runs')}",
        f"- Aggregate review changes: {gate.get('aggregate_review_changes')}",
        f"- Benchmarks: {', '.join(gate.get('benchmark_types', [])) or 'none'}",
        f"- Models: {', '.join(gate.get('models', [])) or 'none'}",
        f"- Primary supported runs: {gate.get('primary_supported_runs')}",
        f"- Primary mixed runs: {gate.get('primary_mixed_runs')}",
        f"- Primary failed runs: {gate.get('primary_failed_runs')}",
        f"- High-confidence regressions: {gate.get('high_confidence_regressions')}",
        "",
        "## Run Table",
        "",
        "| Run | Model | Benchmark | Records | Test | Review | Conservative primary W/L/T |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in run_rows(report):
        lines.append(
            f"| `{row['run_id']}` | `{row['model']}` | `{row['benchmark']}` | "
            f"{row['records']} | {row['test']} | `{row['review']}` | "
            f"{row['primary_wins']}/{row['primary_losses']}/{row['primary_ties']} |"
        )

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This note concerns TCL-v0 only: frozen LLM hidden states, a small probe, and confidence calibration for answer correctness. It does not validate the full Truth Calibration Layer theory, does not prove model truthfulness, and does not establish deployment readiness.",
        "",
    ])
    if final_status != "ready":
        lines.extend([
            "## Next Required Action",
            "",
            "Do not use this as a final extended-validation conclusion yet. Complete the missing matrix, manual review, or reviewed-label reanalysis, then regenerate the aggregator output and this decision note.",
            "",
        ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = load_json(Path(args.decision_json))
    note = build_note(report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(note, encoding="utf-8")
    print(json.dumps({
        "out_md": str(out_md),
        "aggregator_decision": report["decision_gate"]["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
