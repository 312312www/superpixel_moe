"""Subject-disjoint RGB datasets and deterministic LODO batch plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .io import IMAGE_RANGE_CHOICES, restore_image_range


@dataclass(frozen=True)
class DatasetSpec:
    code: str
    folder: str
    prefix: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "CASIA-FASD": DatasetSpec("C", "CASIA-FASD", "casia"),
    "Idiap Replay-Attack": DatasetSpec("I", "Idiap Replay-Attack", "replay"),
    "MSU-MFSD": DatasetSpec("M", "MSU-MFSD", "MSU"),
    "OULU-NPU": DatasetSpec("O", "OULU-NPU", "Oulu"),
}
SPECS_BY_CODE = {spec.code: spec for spec in DATASET_SPECS.values()}
LODO_PROTOCOLS: dict[str, tuple[tuple[str, ...], str]] = {
    "OCI_M": (("O", "C", "I"), "M"),
    "OMI_C": (("O", "M", "I"), "C"),
    "OCM_I": (("O", "C", "M"), "I"),
    "ICM_O": (("I", "C", "M"), "O"),
}


def _domain_folder(dataset_root: str | Path, spec: DatasetSpec) -> Path:
    return Path(dataset_root) / "domain-generalization" / spec.folder


def _array_metadata(path: Path) -> tuple[tuple[int, ...], np.dtype[Any]]:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tuple(mapped.shape), np.dtype(mapped.dtype)
    finally:
        if isinstance(mapped, np.memmap):
            mapped._mmap.close()


def _read_sample(path: Path, index: int) -> np.ndarray:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return np.array(mapped[index], copy=True)
    finally:
        if isinstance(mapped, np.memmap):
            mapped._mmap.close()


def _validate_image_array(path: Path) -> int:
    shape, dtype = _array_metadata(path)
    if len(shape) != 4 or shape[-1] != 3 or shape[0] < 1:
        raise ValueError(f"RGB array must have non-empty NHWC shape: {path} has {shape}")
    if not np.issubdtype(dtype, np.number):
        raise TypeError(f"RGB array must be numeric: {path} has {dtype}")
    return int(shape[0])


def _subject_values(path: Path, expected: int) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    values = np.asarray(values)
    if values.shape != (expected,) or values.dtype.kind not in "iu":
        raise ValueError(f"subject array must have shape {(expected,)} and integer dtype: {path}")
    return values.astype(np.int64, copy=False)


def domain_records(
    dataset_root: str | Path,
    dataset_code: str,
    *,
    split_seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create one deterministic subject split and all records for a domain."""

    if dataset_code not in SPECS_BY_CODE:
        raise ValueError(f"unknown domain code {dataset_code!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    spec = SPECS_BY_CODE[dataset_code]
    folder = _domain_folder(dataset_root, spec)
    class_data: list[tuple[str, int, Path, np.ndarray]] = []
    subject_sets: list[set[int]] = []
    for class_name, label in (("live", 1), ("spoof", 0)):
        image_path = folder / f"{spec.prefix}_images_{class_name}.npy"
        subject_path = folder / f"{spec.prefix}_subject_{class_name}.npy"
        count = _validate_image_array(image_path)
        subjects = _subject_values(subject_path, count)
        class_data.append((class_name, label, image_path, subjects))
        subject_sets.append(set(int(value) for value in np.unique(subjects)))
    if subject_sets[0] != subject_sets[1]:
        raise ValueError(f"live/spoof subject sets differ in {spec.folder}")
    subjects = np.asarray(sorted(subject_sets[0]), dtype=np.int64)
    validation_count = max(1, min(len(subjects) - 1, int(round(len(subjects) * val_fraction))))
    # A stable domain-specific offset avoids coupling split order to dict/hash behavior.
    domain_offset = sum(ord(character) for character in dataset_code) * 1009
    rng = np.random.default_rng(int(split_seed) + domain_offset)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    validation_subjects = set(int(value) for value in shuffled[:validation_count])
    training_subjects = set(int(value) for value in shuffled[validation_count:])

    records: list[dict[str, Any]] = []
    root = Path(dataset_root).resolve()
    for class_name, label, image_path, subject_array in class_data:
        relative = image_path.resolve().relative_to(root).as_posix()
        for sample_index, subject_value in enumerate(subject_array):
            subject_id = int(subject_value)
            base_split = "val" if subject_id in validation_subjects else "train"
            records.append(
                {
                    "sample_id": f"{dataset_code}:{class_name}:{sample_index}",
                    "domain": dataset_code,
                    "dataset": spec.folder,
                    "class": class_name,
                    "label": int(label),
                    "array_file": relative,
                    "sample_index": int(sample_index),
                    "subject_id": subject_id,
                    "base_split": base_split,
                }
            )
    summary = {
        "domain": dataset_code,
        "dataset": spec.folder,
        "subjects": int(len(subjects)),
        "train_subjects": sorted(training_subjects),
        "val_subjects": sorted(validation_subjects),
        "train_samples": sum(record["base_split"] == "train" for record in records),
        "val_samples": sum(record["base_split"] == "val" for record in records),
    }
    return records, summary


def generate_lodo_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int = 20260819,
    val_fraction: float = 0.2,
) -> dict[str, Path]:
    """Generate and lock all four local subject-disjoint LODO manifests."""

    root = Path(dataset_root).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records_by_domain: dict[str, list[dict[str, Any]]] = {}
    domain_summaries: dict[str, dict[str, Any]] = {}
    for code in sorted(SPECS_BY_CODE):
        records, summary = domain_records(
            root, code, split_seed=split_seed, val_fraction=val_fraction
        )
        records_by_domain[code] = records
        domain_summaries[code] = summary

    paths: dict[str, Path] = {}
    manifest_hashes: dict[str, str] = {}
    for protocol, (sources, target) in LODO_PROTOCOLS.items():
        protocol_records: list[dict[str, Any]] = []
        for source in sources:
            for source_record in records_by_domain[source]:
                record = dict(source_record)
                record["split"] = record["base_split"]
                record["protocol"] = protocol
                protocol_records.append(record)
        for target_record in records_by_domain[target]:
            record = dict(target_record)
            record["split"] = "test"
            record["protocol"] = protocol
            protocol_records.append(record)
        counts = {
            split: sum(record["split"] == split for record in protocol_records)
            for split in ("train", "val", "test")
        }
        payload = {
            "schema": 1,
            "protocol": protocol,
            "protocol_name": "+".join(sources) + " -> " + target,
            "dataset_root": str(root),
            "split_seed": int(split_seed),
            "val_fraction": float(val_fraction),
            "sources": list(sources),
            "target": target,
            "counts": counts,
            "domain_summaries": {code: domain_summaries[code] for code in (*sources, target)},
            "records": protocol_records,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        path = destination / f"{protocol}.json"
        path.write_text(serialized, encoding="utf-8")
        paths[protocol] = path.resolve()
        manifest_hashes[protocol] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    split_manifest = {
        "schema": 1,
        "dataset_root": str(root),
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "domain_summaries": domain_summaries,
        "protocol_sha256": manifest_hashes,
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get("protocol") not in LODO_PROTOCOLS:
        raise ValueError(f"unsupported LODO manifest: {manifest_path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records must be a non-empty list")
    sample_ids = [record.get("sample_id") for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest contains duplicate sample_id entries")
    for split in ("train", "val", "test"):
        expected = int(payload.get("counts", {}).get(split, -1))
        actual = sum(record.get("split") == split for record in records)
        if expected != actual or actual < 1:
            raise ValueError(f"manifest {split} count mismatch: expected={expected}, actual={actual}")
    train_subjects = {
        (record["domain"], int(record["subject_id"]))
        for record in records
        if record["split"] == "train"
    }
    val_subjects = {
        (record["domain"], int(record["subject_id"]))
        for record in records
        if record["split"] == "val"
    }
    if train_subjects & val_subjects:
        raise ValueError("manifest leaks subjects between train and validation")
    expected_sources, expected_target = LODO_PROTOCOLS[str(payload["protocol"])]
    source_domains = {
        str(record["domain"]) for record in records if record["split"] in ("train", "val")
    }
    test_domains = {str(record["domain"]) for record in records if record["split"] == "test"}
    if source_domains != set(expected_sources) or test_domains != {expected_target}:
        raise ValueError(
            f"manifest domain assignment mismatch: sources={source_domains}, target={test_domains}"
        )
    return payload


class ManifestFASDataset(Dataset[dict[str, Any]]):
    """Lazy RGB reader for one split of a locked LODO manifest."""

    def __init__(
        self,
        dataset_root: str | Path,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        image_range: str = "0-1/255",
    ) -> None:
        if image_range not in IMAGE_RANGE_CHOICES:
            raise ValueError(f"unsupported image range {image_range!r}")
        self.dataset_root = Path(dataset_root)
        self.records = [dict(record) for record in records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"manifest contains no {split!r} records")
        self.split = split
        self.image_range = image_range

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.dataset_root / Path(str(record["array_file"]))
        array = _read_sample(path, int(record["sample_index"]))
        restored, _ = restore_image_range(array, self.image_range)
        image = torch.from_numpy(restored).permute(2, 0, 1).contiguous()
        return {
            "image": image,
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "domain": str(record["domain"]),
            "subject_id": torch.tensor(int(record["subject_id"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
        }


class NpyBinaryFASDataset(Dataset[dict[str, torch.Tensor]]):
    """Backward-compatible lazy reader for one live/spoof NPY pair."""

    def __init__(
        self,
        dataset_root: str | Path,
        dataset: str,
        limit_samples: int | None = None,
        image_range: str = "auto",
    ) -> None:
        if dataset not in DATASET_SPECS:
            raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(DATASET_SPECS)}")
        if limit_samples is not None and int(limit_samples) < 0:
            raise ValueError("limit_samples must be nonnegative when provided")
        if image_range not in IMAGE_RANGE_CHOICES:
            raise ValueError(f"unsupported image range {image_range!r}")
        spec = DATASET_SPECS[dataset]
        folder = _domain_folder(dataset_root, spec)
        self.live_path = folder / f"{spec.prefix}_images_live.npy"
        self.spoof_path = folder / f"{spec.prefix}_images_spoof.npy"
        live = np.load(self.live_path, mmap_mode="r", allow_pickle=False)
        spoof = np.load(self.spoof_path, mmap_mode="r", allow_pickle=False)
        try:
            live_shape, live_dtype, live_max = live.shape, live.dtype, float(np.asarray(live.max()))
            spoof_shape, spoof_dtype, spoof_max = spoof.shape, spoof.dtype, float(np.asarray(spoof.max()))
        finally:
            for mapped in (live, spoof):
                if isinstance(mapped, np.memmap):
                    mapped._mmap.close()
        if len(live_shape) != 4 or live_shape[-1] != 3 or live_shape[1:] != spoof_shape[1:]:
            raise ValueError("cached live/spoof RGB arrays must have matching NHWC shape")
        if live_shape[0] < 1 or spoof_shape[0] < 1:
            raise ValueError("cached RGB arrays must contain at least one sample per class")
        if np.issubdtype(live_dtype, np.integer) != np.issubdtype(spoof_dtype, np.integer):
            raise ValueError("live/spoof arrays must use compatible numeric dtypes")
        self.image_range = image_range
        self.detected_range: str | None = None
        if image_range == "auto":
            combined_max = max(live_max, spoof_max)
            if np.issubdtype(live_dtype, np.integer):
                self.detected_range = "0-255" if combined_max <= 255.0 else "auto"
            elif combined_max <= (1.0 / 255.0) + 1e-6:
                self.detected_range = "0-1/255"
            elif combined_max <= 1.0 + 1e-6:
                self.detected_range = "0-1"
            elif combined_max <= 255.0 + 1e-6:
                self.detected_range = "0-255"
            else:
                raise ValueError(f"unsupported dataset image range: max={combined_max}")
        source_live = int(live_shape[0])
        source_spoof = int(spoof_shape[0])
        total = source_live + source_spoof
        requested = min(total, int(limit_samples)) if limit_samples is not None else total
        if requested < total:
            selected_live = min(source_live, (requested + 1) // 2)
            selected_spoof = min(source_spoof, requested // 2)
            remaining = requested - selected_live - selected_spoof
            extra_live = min(source_live - selected_live, remaining)
            selected_live += extra_live
            remaining -= extra_live
            selected_spoof += min(source_spoof - selected_spoof, remaining)
        else:
            selected_live, selected_spoof = source_live, source_spoof
        self.live_count = int(selected_live)
        self.spoof_count = int(selected_spoof)
        self.length = self.live_count + self.spoof_count

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < self.live_count:
            array, label = _read_sample(self.live_path, index), 1
        else:
            spoof_index = index - self.live_count
            if spoof_index >= self.spoof_count:
                raise IndexError(index)
            array, label = _read_sample(self.spoof_path, spoof_index), 0
        restored, _ = restore_image_range(array, self.detected_range or self.image_range)
        return {
            "image": torch.from_numpy(restored).permute(2, 0, 1).contiguous(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def make_balanced_epoch_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    num_batches: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Build model-independent, nearly exact domain/class-balanced batches."""

    if batch_size < 1 or num_batches < 1:
        raise ValueError("batch_size and num_batches must be positive")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record["domain"]), int(record["label"]))].append(index)
    keys = sorted(groups)
    if not keys or any(not groups[key] for key in keys):
        raise ValueError("every domain/class sampling group must be non-empty")
    rng = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    pools: dict[tuple[str, int], list[int]] = {}
    offsets: dict[tuple[str, int], int] = {}

    def draw(key: tuple[str, int]) -> int:
        pool = pools.get(key)
        offset = offsets.get(key, 0)
        if pool is None or offset >= len(pool):
            pool = list(groups[key])
            rng.shuffle(pool)
            pools[key] = pool
            offset = 0
        value = pool[offset]
        offsets[key] = offset + 1
        return value

    batches: list[list[int]] = []
    base, remainder = divmod(batch_size, len(keys))
    for batch_index in range(num_batches):
        quotas = {key: base for key in keys}
        for offset in range(remainder):
            quotas[keys[(batch_index * max(1, remainder) + offset) % len(keys)]] += 1
        batch = [draw(key) for key in keys for _ in range(quotas[key])]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


class FixedDomainClassBatchSampler(Sampler[list[int]]):
    """Epoch-addressable sampler whose sequence is independent of model RNG use."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        seed: int,
        num_batches: int | None = None,
    ) -> None:
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_batches = int(num_batches or math.ceil(len(self.records) / self.batch_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def batches(self) -> list[list[int]]:
        return make_balanced_epoch_batches(
            self.records,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            seed=self.seed,
            epoch=self.epoch,
        )

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches())

    def __len__(self) -> int:
        return self.num_batches


def audit_duplicate_images(
    dataset_root: str | Path, records: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Hash unique NPY rows and report content duplicates and split leakage."""

    root = Path(dataset_root)
    unique_records: dict[str, dict[str, Any]] = {}
    for source in records:
        record = dict(source)
        unique_records.setdefault(str(record["sample_id"]), record)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in unique_records.values():
        path = root / Path(str(record["array_file"]))
        array = np.ascontiguousarray(_read_sample(path, int(record["sample_index"])))
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
        groups[digest.hexdigest()].append(record)
    duplicates = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        identities = {(item["domain"], int(item["subject_id"])) for item in members}
        labels = {int(item["label"]) for item in members}
        splits = {str(item.get("split", item.get("base_split"))) for item in members}
        duplicates.append(
            {
                "sha256": digest,
                "count": len(members),
                "cross_identity": len(identities) > 1,
                "cross_label": len(labels) > 1,
                "cross_split": len(splits) > 1,
                "members": members,
            }
        )
    return {
        "schema": 1,
        "unique_samples": len(unique_records),
        "duplicate_groups": len(duplicates),
        "cross_identity_groups": sum(item["cross_identity"] for item in duplicates),
        "cross_label_groups": sum(item["cross_label"] for item in duplicates),
        "cross_split_groups": sum(item["cross_split"] for item in duplicates),
        "duplicates": duplicates,
    }


__all__ = [
    "DATASET_SPECS",
    "LODO_PROTOCOLS",
    "DatasetSpec",
    "FixedDomainClassBatchSampler",
    "ManifestFASDataset",
    "NpyBinaryFASDataset",
    "audit_duplicate_images",
    "domain_records",
    "generate_lodo_manifests",
    "load_manifest",
    "make_balanced_epoch_batches",
]
