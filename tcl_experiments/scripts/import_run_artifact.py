from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def safe_members(zip_file: zipfile.ZipFile):
    members = []
    for info in zip_file.infolist():
        name = info.filename.replace("\\", "/")
        parts = Path(name).parts
        if name.startswith("/") or ".." in parts:
            raise ValueError(f"Unsafe zip member path: {info.filename}")
        members.append(info)
    return members


def find_run_dirs(extract_dir: Path):
    candidates = []
    for path in extract_dir.iterdir():
        if not path.is_dir() or path.name == "benchmark_data":
            continue
        if any(path.glob("records_*.jsonl")) and (path / "analysis").exists():
            candidates.append(path)
    return candidates


def run_verifier(run_dir: Path, method: str, min_records: int, strict: bool):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "verify_run_artifact.py"),
        "--run-dir",
        str(run_dir),
        "--method",
        method,
        "--min-records",
        str(min_records),
        "--out-json",
        str(run_dir / "artifact_verification_import.json"),
        "--out-md",
        str(run_dir / "ARTIFACT_VERIFICATION_IMPORT.md"),
    ]
    if strict:
        command.extend(["--require-manual-review", "--require-targeted-review", "--require-calibrated"])
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Downloaded run artifact zip.")
    parser.add_argument("--extract-dir", default="imported_artifacts")
    parser.add_argument("--method", default="answer_mean")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Artifact zip does not exist: {zip_path}")

    extract_dir = Path(args.extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        safe_members(zf)
        zf.extractall(extract_dir)

    run_dirs = find_run_dirs(extract_dir)
    if not run_dirs:
        raise SystemExit(f"No run directory found after extracting {zip_path} into {extract_dir}")
    if len(run_dirs) > 1:
        raise SystemExit(f"Multiple run directories found; choose manually: {[str(path) for path in run_dirs]}")

    run_dir = run_dirs[0]
    verification_returncode = None
    if args.verify or args.strict:
        verification_returncode = run_verifier(
            run_dir=run_dir,
            method=args.method,
            min_records=args.min_records,
            strict=args.strict,
        )
        if verification_returncode != 0:
            raise SystemExit(verification_returncode)

    print(json.dumps({
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "run_dir": str(run_dir),
        "verification_returncode": verification_returncode,
        "next_decision_gate_arg": f"--run-dir {run_dir}",
    }, indent=2))


if __name__ == "__main__":
    main()
