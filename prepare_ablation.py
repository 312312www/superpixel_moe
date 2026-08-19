"""Generate locked subject-disjoint LODO manifests and a duplicate-image audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fas_moe.datasets import audit_duplicate_images, generate_lodo_manifests, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation_splits"))
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--duplicate-audit", action=argparse.BooleanOptionalAction, default=True,
        help="hash all 6670 NPY rows and report cross-identity/label/split duplicates",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = generate_lodo_manifests(
        args.dataset_root,
        args.output_dir,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
    )
    print("Generated manifests:")
    for protocol, path in sorted(paths.items()):
        payload = load_manifest(path)
        print(f"  {protocol}: {payload['counts']} -> {path}")
    if args.duplicate_audit:
        # Every protocol contains each of the four domains exactly once; one is
        # sufficient to audit all unique underlying RGB samples.
        first = load_manifest(next(iter(paths.values())))
        report = audit_duplicate_images(args.dataset_root, first["records"])
        audit_path = args.output_dir / "duplicate_audit.json"
        audit_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "Duplicate audit: "
            f"groups={report['duplicate_groups']} "
            f"cross_identity={report['cross_identity_groups']} "
            f"cross_label={report['cross_label_groups']} "
            f"cross_split={report['cross_split_groups']}"
        )
        print(f"Audit: {audit_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
