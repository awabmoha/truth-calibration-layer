from __future__ import annotations

import argparse
import glob
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INCLUDE_PATTERNS = [
    "records_*.jsonl",
    "records_*.config.json",
    "analysis/**",
    "manual_review_all.csv",
    "targeted_manual_review_candidates.csv",
    "artifact_verification.json",
    "ARTIFACT_VERIFICATION.md",
]


def collect_files(run_dir: Path):
    files: dict[Path, str] = {}
    for pattern in DEFAULT_INCLUDE_PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                files[path] = str(path.relative_to(run_dir.parent))
    return files


def add_if_exists(files: dict[Path, str], path: Path, arcname_root: str):
    if path.exists() and path.is_file():
        files[path] = f"{arcname_root}/{path.name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark-glob", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    files = collect_files(run_dir)
    for pattern in args.benchmark_glob:
        for path_text in glob.glob(pattern):
            add_if_exists(files, Path(path_text), "benchmark_data")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_out) if args.manifest_out else run_dir / "artifact_manifest.json"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "zip_path": str(out_path),
        "n_files": len(files),
        "files": [
            {"source": str(source), "archive_name": archive_name}
            for source, archive_name in sorted(files.items(), key=lambda item: item[1])
        ],
        "claim_boundary": "This package preserves TCL-v0 run artifacts only. It does not validate the full TCL theory.",
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    files[manifest_path] = str(manifest_path.relative_to(run_dir.parent)) if manifest_path.is_relative_to(run_dir.parent) else manifest_path.name

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, archive_name in sorted(files.items(), key=lambda item: item[1]):
            zf.write(source, archive_name)

    print(json.dumps({
        "zip_path": str(out_path),
        "manifest_path": str(manifest_path),
        "n_files": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
