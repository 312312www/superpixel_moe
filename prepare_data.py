"""Generate subject-disjoint manifests for the three current domains plus OULU-NPU placeholder.

When OULU-NPU files are absent, ``intra_O.json``, ``lodo_O.json`` and
``four_domain_protocol.json`` are emitted with ``status=pending_dataset``.
Adding the OULU NPY files and rerunning this command converts those placeholders
into real four-domain manifests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fas_moe.data import generate_manifests, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--no-placeholders", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = generate_manifests(
        args.dataset_root,
        args.output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        include_placeholders=not args.no_placeholders,
    )
    for name, path in sorted(paths.items()):
        payload = load_manifest(path)
        print(f"{name}: status={payload.get('status', 'ready')} counts={payload.get('counts')} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
