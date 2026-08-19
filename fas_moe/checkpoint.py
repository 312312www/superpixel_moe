"""Checkpoint compatibility helpers for Superpixel-MoE models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


# These fields change parameter names/shapes or alter the meaning of the
# forward pass.  Other saved options (for example ``freeze_backbone``) are
# runtime choices and do not make an otherwise complete state dict invalid.
STRUCTURAL_CONFIG_FIELDS = (
    "experiment",
    "use_superpixel",
    "use_landmarks",
    "moe_mode",
    "levels",
    "feature_channels",
    "position_dim",
    "expert_hidden_dim",
    "num_experts",
    "num_classes",
    "freeze_batch_norm",
    "image_range",
)


def _normalise_config_value(value: Any) -> Any:
    """Make JSON/list/tuple values comparable without changing their meaning."""

    if isinstance(value, (list, tuple)):
        return tuple(_normalise_config_value(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_config_value(item) for key, item in value.items()}
    return value


def checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    """Extract and validate the state-dict portion of a checkpoint payload."""

    state = payload.get("model_state", payload) if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a mapping named 'model_state'")
    non_tensor = [str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor:
        raise ValueError(f"checkpoint state contains non-tensor values: {non_tensor[:8]}")
    return state


def validate_checkpoint(
    model: nn.Module,
    payload: Any,
    *,
    source: str | Path | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint against ``model`` and return a load report.

    Validation is performed before the actual load, including exact key and
    tensor-shape checks and the architecture-affecting fields saved by
    ``train_moe.py``.  A descriptive ``ValueError`` is raised for any mismatch;
    callers can then avoid producing logits from a partially initialized model.
    """

    state = checkpoint_state(payload)
    expected_state = model.state_dict()
    missing = sorted(str(key) for key in expected_state.keys() - state.keys())
    unexpected = sorted(str(key) for key in state.keys() - expected_state.keys())
    shape_mismatches: list[str] = []
    for key in expected_state.keys() & state.keys():
        expected_shape = tuple(expected_state[key].shape)
        actual_shape = tuple(state[key].shape)
        if expected_shape != actual_shape:
            shape_mismatches.append(f"{key}: checkpoint {actual_shape}, model {expected_shape}")

    checkpoint_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    config_mismatches: list[str] = []
    metadata_mismatches: list[str] = []
    model_config = getattr(model, "config", None)
    if checkpoint_config is not None and not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping when present")
    if isinstance(checkpoint_config, Mapping) and model_config is not None:
        for field in STRUCTURAL_CONFIG_FIELDS:
            if not hasattr(model_config, field):
                continue
            if field not in checkpoint_config:
                config_mismatches.append(f"{field}: missing from checkpoint")
                continue
            saved = _normalise_config_value(checkpoint_config[field])
            current = _normalise_config_value(getattr(model_config, field))
            if saved != current:
                config_mismatches.append(f"{field}: checkpoint={saved!r}, model={current!r}")
    if expected_metadata is not None:
        if not isinstance(payload, Mapping):
            metadata_mismatches.append("checkpoint payload is not a mapping")
        else:
            for field, expected in expected_metadata.items():
                actual = payload.get(field, "<missing>")
                if _normalise_config_value(actual) != _normalise_config_value(expected):
                    metadata_mismatches.append(
                        f"{field}: checkpoint={actual!r}, expected={expected!r}"
                    )

    if missing or unexpected or shape_mismatches or config_mismatches or metadata_mismatches:
        origin = f" from {Path(source)}" if source is not None else ""
        details = []
        if missing:
            details.append(f"missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            details.append(f"unexpected keys ({len(unexpected)}): {unexpected[:8]}")
        if shape_mismatches:
            details.append(f"shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:8]}")
        if config_mismatches:
            details.append(f"model_config mismatches ({len(config_mismatches)}): {config_mismatches[:8]}")
        if metadata_mismatches:
            details.append(f"metadata mismatches ({len(metadata_mismatches)}): {metadata_mismatches[:8]}")
        raise ValueError("incompatible Superpixel-MoE checkpoint" + origin + ": " + "; ".join(details))

    # The strict load is intentional: the preflight above makes the error
    # actionable while guaranteeing no parameter is silently skipped.
    model.load_state_dict(state, strict=True)
    return {
        "source": str(Path(source).resolve()) if source is not None else None,
        "state_keys": len(state),
        "model_config_present": isinstance(checkpoint_config, Mapping),
        "metadata_validated": sorted(expected_metadata) if expected_metadata is not None else [],
        "validated_strict": True,
    }


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a serialized checkpoint into ``model``."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return validate_checkpoint(
        model, payload, source=checkpoint_path, expected_metadata=expected_metadata
    )


__all__ = ["STRUCTURAL_CONFIG_FIELDS", "checkpoint_state", "load_checkpoint", "validate_checkpoint"]
