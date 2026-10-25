"""Run one part of the Phase 2 trainings for one training dataset, spreading jobs over the available GPUs.

Protocol: docs/phase2/TCL-Phase2-Protocol.md (amendment of 2026-10-25). The 14 trainings per dataset are split
into three parts so that each Kaggle session finishes well inside its time limit and saves its own outputs:

  part 1: arm C with lambda in {0.1, 0.5, 1.0}, seed 0; arm A seeds 0-2 (seed 0 regenerates). Writes lambda_selection.json.
  part 2: arm C with the chosen lambda, seeds 1-2; arm B seeds 0-2. Needs --lam from part 1's lambda_selection.json.
  part 3: arm D seeds 0-2.

Each job is one call of train_trust_head.py, pinned to one GPU with CUDA_VISIBLE_DEVICES. Jobs whose metrics.json
already exists are skipped. The exit code is non-zero if any job failed; finished jobs keep their outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LAMBDAS = (0.1, 0.5, 1.0)
MICRO_BATCH = {"triviaqa": 8, "squad": 2}  # squad inputs include the passage (amendment of 2026-10-21)


def jobs_for(part: int, lam: float | None) -> list[dict]:
    if part == 1:
        return ([{"arm": "C", "seed": 0, "lam": l} for l in LAMBDAS]
                + [{"arm": "A", "seed": s, "regenerate": s == 0} for s in (0, 1, 2)])
    if part == 2:
        if lam is None:
            raise SystemExit("part 2 needs --lam (the lambda chosen in part 1)")
        if lam not in LAMBDAS:
            raise SystemExit(f"--lam must be one of {LAMBDAS}")
        return [{"arm": "C", "seed": s, "lam": lam} for s in (1, 2)] + [{"arm": "B", "seed": s} for s in (0, 1, 2)]
    if part == 3:
        return [{"arm": "D", "seed": s} for s in (0, 1, 2)]
    raise SystemExit("--part must be 1, 2, or 3")


def out_dir(root: Path, job: dict) -> Path:
    name = f"C-lam{job['lam']}" if job["arm"] == "C" else job["arm"]
    return root / name / f"seed{job['seed']}"


def command(job: dict, args, out: Path) -> list[str]:
    cmd = [sys.executable, str(SCRIPTS / "train_trust_head.py"), "--arm", job["arm"],
           "--train-dataset", args.train_dataset, "--seed", str(job["seed"]), "--out-dir", str(out),
           "--micro-batch", str(args.micro_batch or MICRO_BATCH[args.train_dataset])]
    if "lam" in job:
        cmd += ["--lam", str(job["lam"])]
    if job.get("regenerate"):
        cmd += ["--regenerate"]
    if args.model:
        cmd += ["--model", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit), "--wikitext-limit", "5"]
    return cmd


def choose_lambda(root: Path) -> dict:
    """Validation AUROC of arm C, seed 0, for each lambda; the highest wins, ties go to the smaller lambda."""
    val_auc = {}
    for l in LAMBDAS:
        m = json.loads((root / f"C-lam{l}" / "seed0" / "metrics.json").read_text(encoding="utf-8"))
        val_auc[l] = m["validation_auroc"]
    chosen = max(val_auc, key=lambda k: val_auc[k] if val_auc[k] is not None else -1)
    return {"validation_auroc": val_auc, "chosen": chosen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dataset", choices=["triviaqa", "squad"], required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out-root", type=Path, required=True, help="jobs write to <out-root>/<train-dataset>/<arm>/seed<k>/")
    ap.add_argument("--slots", default=None, help="comma-separated GPU ids; default: all visible GPUs (or one CPU slot)")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows per split, for quick checks only")
    args = ap.parse_args()

    root = args.out_root / args.train_dataset
    root.mkdir(parents=True, exist_ok=True)
    if args.slots is not None:
        slots = [s for s in args.slots.split(",") if s != ""]
    else:
        import torch
        slots = [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]

    todo: queue.Queue = queue.Queue()
    for job in jobs_for(args.part, args.lam):
        todo.put(job)
    results, lock = [], threading.Lock()

    def worker(slot: str) -> None:
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if slot != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = slot
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            out = out_dir(root, job)
            if (out / "metrics.json").exists():
                with lock:
                    print(f"[{slot}] skip {out.relative_to(args.out_root)}", flush=True)
                    results.append((job, 0, 0.0))
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = command(job, args, out)
            with lock:
                print(f"[{slot}] start {' '.join(cmd[1:])}", flush=True)
            t0 = time.time()
            with (out / "train.log").open("w", encoding="utf-8") as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            secs = time.time() - t0
            with lock:
                status = "done" if code == 0 else f"FAILED (exit {code}, see {out / 'train.log'})"
                print(f"[{slot}] {status} {out.relative_to(args.out_root)} in {secs / 60:.1f} min", flush=True)
                results.append((job, code, secs))

    threads = [threading.Thread(target=worker, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [j for j, code, _ in results if code != 0]
    if args.part == 1 and not failed:
        sel = choose_lambda(root)
        (root / "lambda_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        print("validation AUROC by lambda:", sel["validation_auroc"], "-> chosen", sel["chosen"], flush=True)
    print(f"{len(results) - len(failed)} of {len(results)} jobs finished", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
