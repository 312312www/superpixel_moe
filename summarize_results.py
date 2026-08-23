"""Summarize A--E experiment outputs as Markdown or JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOLS = {
    "intra_C": "CASIA → CASIA", "intra_I": "Replay → Replay", "intra_M": "MSU → MSU",
    "lodo_C": "Replay+MSU → CASIA", "lodo_I": "CASIA+MSU → Replay", "lodo_M": "CASIA+Replay → MSU",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = []
    for phase in "ABCDE":
        for protocol, train_test in PROTOCOLS.items():
            path = Path("outputs") / f"phase{phase}" / protocol / "metrics_test.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            macro = metrics["macro"]
            rows.append(
                {
                    "phase": phase, "protocol": protocol, "train_test": train_test,
                    "auc": round(float(macro["auc"]), 4), "hter": round(float(macro["hter"]), 4),
                    "apcer": round(float(macro["apcer"]), 4), "bpcer": round(float(macro["bpcer"]), 4),
                    "best_epoch": metrics.get("best_epoch"), "threshold": round(float(metrics["threshold"]), 4),
                }
            )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print("| 阶段 | 协议 | 训练 → 测试 | AUC | HTER | APCER | BPCER |")
        print("|---|---|---|---|---|---|---|")
        for row in rows:
            print(f"| {row['phase']} | {row['protocol']} | {row['train_test']} | {row['auc']:.4f} | {row['hter']:.4f} | {row['apcer']:.4f} | {row['bpcer']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
