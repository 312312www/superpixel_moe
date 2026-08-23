"""Dataset discovery, subject-disjoint manifests, and lazy NPY loading.

The current workspace contains CASIA-FASD, Idiap Replay-Attack and MSU-MFSD.
OULU-NPU is declared as a fourth-domain placeholder: if its files are absent,
manifest generation emits a pending placeholder instead of silently pretending
that the domain was evaluated.  When the files are added, the same command
regenerates all four-domain intra/LODO manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS: dict[str, str] = {
    "CASIA-FASD": "casia",
    "Idiap Replay-Attack": "replay",
    "MSU-MFSD": "MSU",
    "OULU-NPU": "Oulu",
}
DATASET_CODES: dict[str, str] = {
    "CASIA-FASD": "C",
    "Idiap Replay-Attack": "I",
    "MSU-MFSD": "M",
    "OULU-NPU": "O",
}


@dataclass(frozen=True)
class DatasetSpec:
    code: str
    folder: str
    prefix: str
    subjects: int


SPECS: dict[str, DatasetSpec] = {
    "CASIA-FASD": DatasetSpec("C", "CASIA-FASD", "casia", 50),
    "Idiap Replay-Attack": DatasetSpec("I", "Idiap Replay-Attack", "replay", 35),
    "MSU-MFSD": DatasetSpec("M", "MSU-MFSD", "MSU", 35),
    "OULU-NPU": DatasetSpec("O", "OULU-NPU", "Oulu", 55),
}


def _folder(dataset_root: str | Path, dataset: str) -> Path:
    return Path(dataset_root) / "domain-generalization" / SPECS[dataset].folder


def _paths(dataset_root: str | Path, dataset: str, class_name: str) -> tuple[Path, Path]:
    spec = SPECS[dataset]
    folder = _folder(dataset_root, dataset)
    return folder / f"{spec.prefix}_images_{class_name}.npy", folder / f"{spec.prefix}_subject_{class_name}.npy"


def dataset_available(dataset_root: str | Path, dataset: str) -> bool:
    if dataset not in SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(SPECS)}")
    return all(path.is_file() for class_name in ("live", "spoof") for path in _paths(dataset_root, dataset, class_name))


def available_datasets(dataset_root: str | Path) -> list[str]:
    return [dataset for dataset in SPECS if dataset_available(dataset_root, dataset)]


def _load_array_meta(path: Path) -> tuple[tuple[int, ...], np.dtype[Any]]:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tuple(mapped.shape), np.dtype(mapped.dtype)
    finally:
        if isinstance(mapped, np.memmap):
            mapped._mmap.close()


def _read_row(path: Path, index: int) -> np.ndarray:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return np.array(mapped[index], copy=True)
    finally:
        if isinstance(mapped, np.memmap):
            mapped._mmap.close()


def _load_subjects(path: Path, expected: int) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.shape != (expected,) or values.dtype.kind not in "iu":
        raise ValueError(f"subject array must be ({expected},) integer: {path}")
    return values.astype(np.int64, copy=False)


def dataset_records(dataset_root: str | Path, dataset: str) -> list[dict[str, Any]]:
    if dataset not in SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(SPECS)}")
    if not dataset_available(dataset_root, dataset):
        missing = [str(path) for class_name in ("live", "spoof") for path in _paths(dataset_root, dataset, class_name) if not path.is_file()]
        raise FileNotFoundError(f"dataset {dataset} is not available; missing: {missing}")
    spec = SPECS[dataset]
    records: list[dict[str, Any]] = []
    for class_name, label in (("live", 1), ("spoof", 0)):
        image_path, subject_path = _paths(dataset_root, dataset, class_name)
        shape, dtype = _load_array_meta(image_path)
        if len(shape) != 4 or shape[-1] != 3 or not np.issubdtype(dtype, np.number):
            raise ValueError(f"RGB array must be numeric NHWC: {image_path} has {shape}/{dtype}")
        subjects = _load_subjects(subject_path, int(shape[0]))
        for index, subject_id in enumerate(subjects):
            records.append(
                {
                    "sample_id": f"{spec.code}:{class_name}:{index}",
                    "dataset": dataset,
                    "class": class_name,
                    "label": int(label),
                    "array_file": str(image_path.resolve()),
                    "sample_index": int(index),
                    "subject_id": int(subject_id),
                }
            )
    return records


def subject_split(
    records: Sequence[Mapping[str, Any]], *, seed: int, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> tuple[set[int], set[int], set[int]]:
    subjects = sorted({int(record["subject_id"]) for record in records})
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required for train/val/test")
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(subjects, dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    while val_count + test_count >= len(shuffled):
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("split fractions leave no training subject")
    val_subjects = set(int(value) for value in shuffled[:val_count])
    test_subjects = set(int(value) for value in shuffled[val_count : val_count + test_count])
    train_subjects = set(int(value) for value in shuffled[val_count + test_count :])
    return train_subjects, val_subjects, test_subjects


def _placeholder(destination: Path, dataset: str, protocol: str, protocol_name: str, required_codes: list[str]) -> Path:
    path = destination / f"intra_{DATASET_CODES[dataset]}.json" if protocol == "intra" else destination / f"lodo_{DATASET_CODES[dataset]}.json"
    payload = {
        "schema": 3,
        "status": "pending_dataset",
        "protocol": protocol,
        "protocol_name": protocol_name,
        "dataset": dataset,
        "required_codes": required_codes,
        "counts": {"train": 0, "val": 0, "test": 0},
        "records": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def generate_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260819,
    val_fraction: float = 0.15,
    include_placeholders: bool = True,
) -> dict[str, Path]:
    """Generate actual manifests for available domains and OULU placeholders."""
    root = Path(dataset_root).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, tuple[set[int], set[int], set[int]]] = {}
    for dataset in SPECS:
        if dataset_available(root, dataset):
            records_by_dataset[dataset] = dataset_records(root, dataset)
            splits[dataset] = subject_split(records_by_dataset[dataset], seed=seed, val_fraction=val_fraction)

    manifests: dict[str, Path] = {}
    for dataset in SPECS:
        if dataset not in records_by_dataset:
            if include_placeholders:
                manifests[f"intra_{DATASET_CODES[dataset]}"] = _placeholder(
                    destination, dataset, "intra", f"{dataset} pending local files", [DATASET_CODES[dataset]]
                )
            continue
        records = [dict(record) for record in records_by_dataset[dataset]]
        train_subjects, val_subjects, test_subjects = splits[dataset]
        for record in records:
            subject = int(record["subject_id"])
            record["split"] = "train" if subject in train_subjects else "val" if subject in val_subjects else "test"
        payload = {
            "schema": 3,
            "status": "ready",
            "protocol": "intra",
            "protocol_name": f"{dataset} subject-disjoint 70/15/15",
            "dataset_root": str(root),
            "seed": int(seed),
            "sources": [DATASET_CODES[dataset]],
            "target": DATASET_CODES[dataset],
            "counts": {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")},
            "records": records,
        }
        path = destination / f"intra_{DATASET_CODES[dataset]}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifests[f"intra_{DATASET_CODES[dataset]}"] = path

    available = list(records_by_dataset)
    for target in available:
        sources = [dataset for dataset in available if dataset != target]
        if len(sources) < 2:
            continue
        combined: list[dict[str, Any]] = []
        for source in sources:
            train_subjects, val_subjects, _ = splits[source]
            for record in records_by_dataset[source]:
                item = dict(record)
                item["split"] = "train" if int(item["subject_id"]) in train_subjects else "val"
                combined.append(item)
        combined.extend({**record, "split": "test"} for record in records_by_dataset[target])
        payload = {
            "schema": 3,
            "status": "ready",
            "protocol": "lodo",
            "protocol_name": f"{'+'.join(DATASET_CODES[source] for source in sources)} -> {DATASET_CODES[target]}",
            "dataset_root": str(root),
            "seed": int(seed),
            "sources": [DATASET_CODES[source] for source in sources],
            "target": DATASET_CODES[target],
            "counts": {split: sum(item["split"] == split for item in combined) for split in ("train", "val", "test")},
            "records": combined,
        }
        path = destination / f"lodo_{DATASET_CODES[target]}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifests[f"lodo_{DATASET_CODES[target]}"] = path

    if include_placeholders and "OULU-NPU" not in records_by_dataset:
        manifests["lodo_O"] = _placeholder(destination, "OULU-NPU", "lodo", "C+I+M -> OULU-NPU pending files", ["C", "I", "M", "O"])
    four_domain = destination / "four_domain_protocol.json"
    four_domain.write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "pending_dataset" if "OULU-NPU" not in records_by_dataset else "ready",
                "required_domains": ["C", "I", "M", "O"],
                "instructions": "Add OULU-NPU NPY/subject files, rerun prepare_data.py, then run lodo_O and the four-domain manifests.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifests["four_domain_protocol"] = four_domain
    return manifests


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") == "pending_dataset":
        return payload
    if payload.get("schema") not in (2, 3) or not payload.get("records"):
        raise ValueError(f"unsupported or empty manifest: {path}")
    counts = {split: 0 for split in ("train", "val", "test")}
    for record in payload["records"]:
        if record.get("split") not in counts:
            raise ValueError(f"invalid split in {path}: {record.get('split')!r}")
        counts[record["split"]] += 1
    if any(counts[split] < 1 for split in counts):
        raise ValueError(f"manifest {path} needs train/val/test samples: {counts}")
    return payload


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FASDataset(Dataset[dict[str, Any]]):
    """Lazy RGB reader; optional cached superpixels are returned per sample."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        transform=None,
        superpixel_config=None,
    ) -> None:
        self.records = [dict(record) for record in records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"no {split!r} records")
        self.transform = transform
        self.superpixel_config = superpixel_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        array = _read_row(Path(record["array_file"]), int(record["sample_index"]))
        image = np.clip(np.round(array * (255.0 * 255.0)), 0, 255).astype(np.uint8)
        item: dict[str, Any] = {
            "image": self.transform(image) if self.transform is not None else torch.from_numpy(image).permute(2, 0, 1).float(),
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "dataset": str(record["dataset"]),
            "subject_id": torch.tensor(int(record["subject_id"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
        }
        if self.superpixel_config is not None:
            from .superpixels import cached_superpixels

            views = cached_superpixels(
                image,
                source=record["array_file"],
                index=int(record["sample_index"]),
                config=self.superpixel_config,
            )
            item["superpixel_labels"] = {
                str(level): torch.from_numpy(views[f"labels_{level}"].astype(np.int64, copy=False))
                for level in self.superpixel_config.levels
            }
            item["superpixel_positions"] = {
                str(level): torch.from_numpy(views[f"positions_{level}"].astype(np.float32, copy=False))
                for level in self.superpixel_config.levels
            }
            item["superpixel_valid"] = {
                str(level): torch.from_numpy(views[f"valid_{level}"].astype(np.bool_, copy=False))
                for level in self.superpixel_config.levels
            }
        return item


__all__ = [
    "DATASETS", "DATASET_CODES", "DatasetSpec", "FASDataset", "SPECS",
    "available_datasets", "dataset_available", "dataset_records", "generate_manifests",
    "load_manifest", "manifest_sha256", "subject_split",
]
