from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PRIMARY_METRICS = ["brier", "ece", "wrong_conf_ge_0_8", "wrong_conf_ge_0_9"]
SECONDARY_METRICS = ["mce", "accuracy_at_0_5", "auc"]
LOWER_IS_BETTER = {"brier", "ece", "mce", "wrong_conf_ge_0_8", "wrong_conf_ge_0_9"}
HIGHER_IS_BETTER = {"accuracy_at_0_5", "auc"}
SIGNALS = [
    "tcl_v0_probe_confidence",
    "tcl_v0_calibrated_confidence",
    "tcl_v0_conservative_confidence",
]
PREFERRED_ANALYSIS_DIRS = ["analysis_v2labels", "analysis_strict", "analysis_conservative", "analysis"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_first(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def infer_benchmark(dataset: str, run_id: str):
    text = f"{dataset} {run_id}".lower()
    if "squad" in text:
        return "squad"
    if "triviaqa" in text or "trivia_qa" in text:
        return "triviaqa"
    if "nq" in text:
        return "nq_open"
    return "unknown"


def infer_model_family(model_name: str, run_id: str):
    text = f"{model_name} {run_id}".lower()
    if "qwen" in text:
        return "qwen"
    if "smollm" in text:
        return "smollm"
    return model_name or "unknown"


def infer_model_key(model_name: str, run_id: str):
    if model_name:
        return model_name
    text = run_id.lower()
    if "qwen2.5-1.5b" in text or "qwen15" in text:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    if "qwen2.5-0.5b" in text or "qwen05" in text or "qwen" in text:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    if "smollm2-360m" in text or "smollm" in text:
        return "HuggingFaceTB/SmolLM2-360M-Instruct"
    return "unknown"


def choose_analysis_dir(run_dir: Path, requested: str | None):
    if requested:
        return run_dir / requested
    for name in PREFERRED_ANALYSIS_DIRS:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return run_dir / "analysis"


def compare_metric(raw_value, candidate_value, metric_name: str):
    if raw_value is None or candidate_value is None:
        return None
    if metric_name in LOWER_IS_BETTER:
        if candidate_value < raw_value:
            winner = "candidate"
        elif candidate_value > raw_value:
            winner = "raw"
        else:
            winner = "tie"
    elif metric_name in HIGHER_IS_BETTER:
        if candidate_value > raw_value:
            winner = "candidate"
        elif candidate_value < raw_value:
            winner = "raw"
        else:
            winner = "tie"
    else:
        winner = "unknown"
    return {
        "raw": raw_value,
        "candidate": candidate_value,
        "delta_candidate_minus_raw": candidate_value - raw_value,
        "winner": winner,
    }


def compare_signal(metrics: dict, signal: str):
    raw = metrics.get("raw_generation_confidence", {})
    candidate = metrics.get(signal, {})
    comparisons = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS:
        result = compare_metric(raw.get(metric_name), candidate.get(metric_name), metric_name)
        if result is not None:
            comparisons[metric_name] = result

    primary = {name: comparisons[name] for name in PRIMARY_METRICS if name in comparisons}
    secondary = {name: comparisons[name] for name in SECONDARY_METRICS if name in comparisons}
    return {
        "primary_wins": sum(1 for item in primary.values() if item["winner"] == "candidate"),
        "primary_losses": sum(1 for item in primary.values() if item["winner"] == "raw"),
        "primary_ties": sum(1 for item in primary.values() if item["winner"] == "tie"),
        "secondary_wins": sum(1 for item in secondary.values() if item["winner"] == "candidate"),
        "secondary_losses": sum(1 for item in secondary.values() if item["winner"] == "raw"),
        "secondary_ties": sum(1 for item in secondary.values() if item["winner"] == "tie"),
        "metrics": comparisons,
    }


def load_run(run_dir: Path, method: str, analysis_dir_name: str | None):
    analysis_dir = choose_analysis_dir(run_dir, analysis_dir_name)
    metrics_path = analysis_dir / method / "metrics.json"
    summary_path = analysis_dir / "summary.json"
    config_path = run_dir / f"records_{method}.config.json"
    method_summary_path = analysis_dir / "method_summary.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = load_json(metrics_path)
    config = load_json(config_path) if config_path.exists() else {}
    method_summary = load_csv_first(method_summary_path) if method_summary_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}

    run_id = config.get("run_id") or run_dir.name
    model_name = config.get("model_name") or method_summary.get("model_name") or ""
    dataset = config.get("dataset") or ""
    benchmark = infer_benchmark(dataset, run_id)
    model_family = infer_model_family(model_name, run_id)
    model_key = infer_model_key(model_name, run_id)

    signal_reports = {}
    for signal in SIGNALS:
        if signal in metrics:
            signal_reports[signal] = compare_signal(metrics, signal)

    return {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "run_id": run_id,
        "model_name": model_name,
        "model_key": model_key,
        "model_family": model_family,
        "dataset": dataset,
        "benchmark": benchmark,
        "hidden_state_method": metrics.get("hidden_state_method"),
        "n_records": metrics.get("n_records"),
        "n_test": metrics.get("n_test"),
        "split_mode": metrics.get("split_mode"),
        "calibration_status": metrics.get("calibration_status"),
        "analysis_errors": summary.get("errors", []),
        "signals": signal_reports,
    }


def decide(run_reports: list[dict]):
    conservative_runs = [
        run for run in run_reports if "tcl_v0_conservative_confidence" in run["signals"]
    ]
    benchmarks = {run["benchmark"] for run in conservative_runs if run["benchmark"] != "unknown"}
    models = {run["model_key"] for run in conservative_runs if run["model_key"] != "unknown"}
    model_families = {run["model_family"] for run in conservative_runs if run["model_family"] != "unknown"}
    completed_runs = len(conservative_runs)
    primary_supported = 0
    primary_mixed = 0
    primary_failed = 0
    high_conf_regressions = 0

    for run in conservative_runs:
        comparison = run["signals"]["tcl_v0_conservative_confidence"]
        wins = comparison["primary_wins"]
        losses = comparison["primary_losses"]
        if wins > losses:
            primary_supported += 1
        elif losses > wins:
            primary_failed += 1
        else:
            primary_mixed += 1

        metrics = comparison["metrics"]
        for metric_name in ["wrong_conf_ge_0_8", "wrong_conf_ge_0_9"]:
            if metrics.get(metric_name, {}).get("winner") == "raw":
                high_conf_regressions += 1

    matrix_complete = len(benchmarks) >= 2 and len(models) >= 2 and completed_runs >= 4
    if not matrix_complete:
        decision = "incomplete"
        rationale = "The tested matrix does not yet include at least two distinct models and two benchmark types."
    elif primary_supported >= 3 and high_conf_regressions == 0:
        decision = "supports_continuing_tcl_v0"
        rationale = "Conservative TCL-v0 wins most primary criteria on most runs without high-confidence error regressions."
    elif primary_supported >= 2 and primary_failed <= 1:
        decision = "mixed_continue_cautiously"
        rationale = "Conservative TCL-v0 shows useful signal, but the win pattern is not strong enough for TCL-v1."
    else:
        decision = "revise_or_weaken_hypothesis"
        rationale = "Raw confidence beats or matches conservative TCL-v0 too often under the primary criteria."

    return {
        "decision": decision,
        "rationale": rationale,
        "matrix_complete": matrix_complete,
        "completed_runs": completed_runs,
        "benchmark_types": sorted(benchmarks),
        "models": sorted(models),
        "model_families": sorted(model_families),
        "primary_supported_runs": primary_supported,
        "primary_mixed_runs": primary_mixed,
        "primary_failed_runs": primary_failed,
        "high_confidence_regressions": high_conf_regressions,
        "claim_boundary": "This is a TCL-v0 decision gate only. It does not validate full TCL.",
    }


def build_markdown(report: dict):
    decision = report["decision_gate"]
    lines = [
        "# TCL-v0 Extended Validation Decision Note",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        decision["rationale"],
        "",
        "## Matrix Coverage",
        "",
        f"- Completed runs: {decision['completed_runs']}",
        f"- Benchmark types: {', '.join(decision['benchmark_types']) or 'none'}",
        f"- Models: {', '.join(decision['models']) or 'none'}",
        f"- Model families: {', '.join(decision['model_families']) or 'none'}",
        f"- Matrix complete: {decision['matrix_complete']}",
        "",
        "## Run-Level Results",
        "",
    ]
    for run in report["runs"]:
        lines.extend([
            f"### {run['run_id']}",
            "",
            f"- Model: `{run['model_name'] or run['model_family']}`",
            f"- Benchmark: `{run['benchmark']}`",
            f"- Records/test: {run['n_records']} / {run['n_test']}",
            f"- Analysis dir: `{run['analysis_dir']}`",
        ])
        conservative = run["signals"].get("tcl_v0_conservative_confidence")
        if conservative:
            lines.append(
                f"- Conservative TCL-v0 primary metrics: {conservative['primary_wins']} wins, "
                f"{conservative['primary_losses']} losses, {conservative['primary_ties']} ties vs raw"
            )
            for metric_name in PRIMARY_METRICS:
                metric = conservative["metrics"].get(metric_name)
                if metric:
                    lines.append(
                        f"  - {metric_name}: raw={metric['raw']:.6g}, "
                        f"conservative={metric['candidate']:.6g}, winner={metric['winner']}"
                    )
        else:
            lines.append("- Conservative TCL-v0 metrics: missing")
        lines.append("")

    lines.extend([
        "## Claim Boundary",
        "",
        "This decision note only evaluates TCL-v0 confidence diagnostics. It does not validate the full Truth Calibration Layer theory, does not prove truthfulness, and does not show deployment readiness.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory to include. Repeatable.")
    parser.add_argument("--method", default="answer_mean")
    parser.add_argument("--analysis-dir-name", help="Force one analysis directory name for all run dirs.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    run_reports = [
        load_run(Path(run_dir), method=args.method, analysis_dir_name=args.analysis_dir_name)
        for run_dir in args.run_dir
    ]
    report = {
        "decision_gate": decide(run_reports),
        "runs": run_reports,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps(report["decision_gate"], indent=2))


if __name__ == "__main__":
    main()
