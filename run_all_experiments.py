"""Run reproducible A--E intra-domain and LODO experiments.

The runner skips an existing complete output unless ``--force`` is passed.  It
always writes ``outputs/run_all_summary.json`` after every protocol so an
interrupted CPU campaign can resume without losing finished metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_PROTOCOLS = [
    ("intra_C", "splits/intra_C.json"),
    ("intra_I", "splits/intra_I.json"),
    ("intra_M", "splits/intra_M.json"),
    ("lodo_C", "splits/lodo_C.json"),
    ("lodo_I", "splits/lodo_I.json"),
    ("lodo_M", "splits/lodo_M.json"),
]
PHASES = ("A", "B", "C", "D", "E")


def protocol_manifests() -> list[tuple[str, str]]:
    """Use OULU only when its manifest is ready; never run placeholders."""
    protocols = list(BASE_PROTOCOLS)
    for name in ("intra_O", "lodo_O"):
        path = Path("splits") / f"{name}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ready":
            protocols.append((name, str(path)))
    return protocols


def _row(phase: str, protocol: str, output: Path, exit_code: int, seconds: float) -> dict:
    row = {"phase": phase, "protocol": protocol, "exit": int(exit_code), "seconds": round(seconds, 1)}
    metrics_path = output / "metrics_test.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        macro = metrics["macro"]
        row.update(
            {
                "auc": round(float(macro["auc"]), 4),
                "hter": round(float(macro["hter"]), 4),
                "apcer": round(float(macro["apcer"]), 4),
                "bpcer": round(float(macro["bpcer"]), 4),
                "threshold": round(float(metrics["threshold"]), 4),
                "best_epoch": metrics.get("best_epoch"),
            }
        )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="ABCDE", help="subset of A/B/C/D/E to run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="rerun even if metrics_test.json exists")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    phases = tuple(dict.fromkeys(character.upper() for character in args.phases if character.upper() in PHASES))
    if not phases:
        raise ValueError("--phases must contain one of A/B/C/D/E")
    results: list[dict] = []
    started = time.time()
    root = Path(__file__).resolve().parent
    for phase in phases:
        for protocol, manifest in protocol_manifests():
            output = Path("outputs") / f"phase{phase}" / protocol
            metrics_path = output / "metrics_test.json"
            if metrics_path.is_file() and not args.force:
                row = _row(phase, protocol, output, 0, 0.0)
                row["skipped"] = True
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                continue
            augment = "strong" if protocol.startswith("lodo_") else "standard"
            command = [
                sys.executable, "train_fas.py",
                "--manifest", manifest,
                "--phase", phase,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--augment", augment,
                "--device", args.device,
                "--output-dir", str(output),
            ]
            print(f"\n=== {phase} {protocol} ({augment}) start ===", flush=True)
            run_started = time.time()
            result = subprocess.run(command, cwd=root)
            row = _row(phase, protocol, output, result.returncode, time.time() - run_started)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (Path("outputs")).mkdir(parents=True, exist_ok=True)
            (Path("outputs") / "run_all_summary.json").write_text(
                json.dumps({"phases": phases, "results": results, "total_seconds": round(time.time() - started, 1)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(json.dumps({"phases": phases, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all(row["exit"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
