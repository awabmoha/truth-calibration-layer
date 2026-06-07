from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from import_run_artifact import find_run_dirs, safe_members


SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command: list[str]):
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def import_zip(zip_path: Path, extract_dir: Path):
    target_dir = extract_dir / zip_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        safe_members(zf)
        zf.extractall(target_dir)
    run_dirs = find_run_dirs(target_dir)
    if not run_dirs:
        raise SystemExit(f"No run directory found in imported artifact: {zip_path}")
    if len(run_dirs) > 1:
        raise SystemExit(f"Multiple run directories found in {zip_path}: {[str(path) for path in run_dirs]}")
    return run_dirs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="append", required=True, help="Downloaded run artifact zip. Repeatable.")
    parser.add_argument("--extract-dir", default="imported_artifacts")
    parser.add_argument("--out-dir", default="runs/post_cloud_decision")
    parser.add_argument("--method", default="answer_mean")
    parser.add_argument("--min-records", type=int, default=200)
    parser.add_argument("--strict", action="store_true", help="Require review/calibration artifacts during import verification.")
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    out_dir = Path(args.out_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for zip_text in args.zip:
        zip_path = Path(zip_text)
        if not zip_path.exists():
            raise SystemExit(f"Artifact zip does not exist: {zip_path}")
        run_dir = import_zip(zip_path, extract_dir)
        run_dirs.append(run_dir)
        verify_command = [
            sys.executable,
            str(SCRIPT_DIR / "verify_run_artifact.py"),
            "--run-dir",
            str(run_dir),
            "--method",
            args.method,
            "--min-records",
            str(args.min_records),
            "--out-json",
            str(run_dir / "artifact_verification_import.json"),
            "--out-md",
            str(run_dir / "ARTIFACT_VERIFICATION_IMPORT.md"),
        ]
        if args.strict:
            verify_command.extend(["--require-manual-review", "--require-targeted-review", "--require-calibrated"])
        run_command(verify_command)

    decision_json = out_dir / "extended_validation_decision.json"
    decision_md = out_dir / "EXTENDED_VALIDATION_DECISION.md"
    final_note_md = out_dir / "TCL-v0-EXTENDED-VALIDATION-FINAL-DECISION.md"

    aggregate_command = [
        sys.executable,
        str(SCRIPT_DIR / "summarize_extended_validation.py"),
        "--method",
        args.method,
        "--out-json",
        str(decision_json),
        "--out-md",
        str(decision_md),
    ]
    for run_dir in run_dirs:
        aggregate_command.extend(["--run-dir", str(run_dir)])
    run_command(aggregate_command)

    run_command([
        sys.executable,
        str(SCRIPT_DIR / "write_extended_validation_decision_note.py"),
        "--decision-json",
        str(decision_json),
        "--out-md",
        str(final_note_md),
    ])

    print(json.dumps({
        "imported_run_dirs": [str(path) for path in run_dirs],
        "decision_json": str(decision_json),
        "decision_md": str(decision_md),
        "final_decision_note": str(final_note_md),
        "claim_boundary": "This pipeline orchestrates TCL-v0 artifact import and decision reporting only. It does not validate full TCL.",
    }, indent=2))


if __name__ == "__main__":
    main()
