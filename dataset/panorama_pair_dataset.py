"""Manifest-based panorama pair loading for cross-scene training and inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _first(record: Mapping, keys: Iterable[str], required: bool = False):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    if required:
        raise ValueError(f"pair record is missing one of the required fields: {tuple(keys)}")
    return None


def _resolve_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _candidate_intervals(record: Mapping, keys: Iterable[str]) -> Tuple[Tuple[int, int], ...]:
    candidates = _first(record, keys) or []
    output = []
    for candidate in candidates:
        interval = candidate
        if isinstance(candidate, Mapping):
            interval = _first(candidate, ("token_interval", "tokenInterval"), required=True)
        if not isinstance(interval, Sequence) or len(interval) != 2:
            raise ValueError("each candidate must contain a two-value token interval")
        output.append((int(interval[0]), int(interval[1])))
    return tuple(output)


def _resolve_data_root(value: Optional[str], manifest_dir: Path) -> Path:
    if value is None:
        return manifest_dir
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = manifest_dir / root
    return root.resolve()


@dataclass(frozen=True)
class PanoramaPairRecord:
    pair_id: str
    image_a: str
    image_b: str
    layout_a: Optional[str] = None
    layout_b: Optional[str] = None
    scene_id: str = ""
    floor_id: str = ""
    token_count: int = 256
    candidate_intervals_a: Tuple[Tuple[int, int], ...] = ()
    candidate_intervals_b: Tuple[Tuple[int, int], ...] = ()
    is_match: bool = False
    target_candidate_a: int = -1
    target_candidate_b: int = -1
    relative_transform_b_to_a: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    relative_yaw_radians: float = 0.0
    pose_valid: bool = False

    @classmethod
    def from_mapping(
        cls,
        record: Mapping,
        base_dir: Path,
        index: int,
        default_token_count: int = 256,
    ):
        image_a = _first(record, ("image_a", "imageA", "path_A", "pathA"), required=True)
        image_b = _first(record, ("image_b", "imageB", "path_B", "pathB"), required=True)
        pair_id = str(_first(record, ("id", "pair_id", "pairId")) or f"pair_{index:06d}")
        supervision = record.get("supervision") or record.get("ground_truth") or {}
        if not isinstance(supervision, Mapping):
            raise ValueError(f"pair {pair_id} supervision must be an object")
        token_count = int(
            _first(record, ("token_count", "tokenCount")) or default_token_count
        )
        if token_count <= 0:
            raise ValueError(f"pair {pair_id} token_count must be positive")
        intervals_a = _candidate_intervals(record, ("candidates_a", "candidatesA"))
        intervals_b = _candidate_intervals(record, ("candidates_b", "candidatesB"))
        target_a_value = _first(
            supervision, ("target_candidate_a", "targetCandidateA")
        )
        target_b_value = _first(
            supervision, ("target_candidate_b", "targetCandidateB")
        )
        target_a = -1 if target_a_value is None else int(target_a_value)
        target_b = -1 if target_b_value is None else int(target_b_value)
        transform = _first(
            supervision,
            ("relative_transform_b_to_a", "relativeTransformBToA", "transformBToA"),
        )
        pose_valid_value = _first(supervision, ("pose_valid", "poseValid"))
        pose_valid = (
            bool(transform is not None)
            if pose_valid_value is None
            else bool(pose_valid_value)
        )
        if transform is None:
            transform_array = np.eye(3, dtype=np.float32)
        else:
            transform_array = np.asarray(transform, dtype=np.float32)
            if transform_array.shape != (3, 3):
                raise ValueError(f"pair {pair_id} relative transform must have shape [3, 3]")
        yaw = _first(supervision, ("relative_yaw_radians", "relativeYawRadians"))
        if yaw is None:
            yaw = math.atan2(float(transform_array[1, 0]), float(transform_array[0, 0]))
        is_match_value = _first(supervision, ("is_match", "isMatch"))
        if is_match_value is None:
            is_match_value = bool(
                (target_a >= 0 and target_b >= 0)
                or "interfaceLocalA" in supervision
            )
        return cls(
            pair_id=pair_id,
            image_a=_resolve_path(str(image_a), base_dir),
            image_b=_resolve_path(str(image_b), base_dir),
            layout_a=_resolve_path(
                _first(record, ("layout_a", "layoutA", "label_a", "labelA")), base_dir
            ),
            layout_b=_resolve_path(
                _first(record, ("layout_b", "layoutB", "label_b", "labelB")), base_dir
            ),
            scene_id=str(_first(record, ("scene_id", "sceneId", "house_id")) or ""),
            floor_id=str(_first(record, ("floor_id", "floorId")) or ""),
            token_count=token_count,
            candidate_intervals_a=intervals_a,
            candidate_intervals_b=intervals_b,
            is_match=bool(is_match_value),
            target_candidate_a=target_a,
            target_candidate_b=target_b,
            relative_transform_b_to_a=tuple(
                tuple(float(value) for value in row) for row in transform_array
            ),
            relative_yaw_radians=float(yaw),
            pose_valid=pose_valid,
        )


def load_pair_manifest(
    path: str, data_root: Optional[str] = None
) -> List[PanoramaPairRecord]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pair manifest does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".jsonl":
        raw_records = []
        manifest_data_root = _resolve_data_root(data_root, manifest_path.parent)
        default_token_count = 256
        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL record at {manifest_path}:{line_number}"
                    ) from exc
    else:
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        raw_records = payload.get("pairs", []) if isinstance(payload, dict) else payload
        declared_root = None
        default_token_count = 256
        if isinstance(payload, dict):
            declared_root = _first(payload, ("data_root", "dataRoot"))
            default_token_count = int(
                _first(payload, ("token_count", "tokenCount")) or 256
            )
        manifest_data_root = _resolve_data_root(
            data_root if data_root is not None else declared_root,
            manifest_path.parent,
        )

    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("pair manifest must contain a non-empty list or a {'pairs': [...]} object")
    return [
        PanoramaPairRecord.from_mapping(
            record, manifest_data_root, index, default_token_count=default_token_count
        )
        for index, record in enumerate(raw_records)
    ]


def _interval_masks(
    intervals: Sequence[Tuple[int, int]], token_count: int
) -> torch.Tensor:
    masks = torch.zeros((len(intervals), token_count), dtype=torch.bool)
    for index, (start, end) in enumerate(intervals):
        start %= token_count
        end %= token_count
        if start <= end:
            masks[index, start : end + 1] = True
        else:
            masks[index, start:] = True
            masks[index, : end + 1] = True
    return masks


class PanoramaPairDataset(Dataset):
    """Load paired RGB panoramas using a stable manifest contract."""

    def __init__(
        self,
        manifest_path: str,
        image_shape: Tuple[int, int] = (512, 1024),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        validate_paths: bool = True,
        data_root: Optional[str] = None,
    ):
        self.records = load_pair_manifest(manifest_path, data_root=data_root)
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must contain positive (height, width)")
        self.mean = None if mean is None else torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = None if std is None else torch.tensor(std, dtype=torch.float32)[:, None, None]
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be provided together")
        if self.std is not None and (self.std <= 0).any():
            raise ValueError("normalization std values must be positive")
        token_counts = {record.token_count for record in self.records}
        if len(token_counts) != 1:
            raise ValueError("all records in one pair dataset must use the same token_count")
        self.token_count = next(iter(token_counts))

        if validate_paths:
            missing = []
            for record in self.records:
                for path in (record.image_a, record.image_b, record.layout_a, record.layout_b):
                    if path and not Path(path).is_file():
                        missing.append(path)
            if missing:
                preview = ", ".join(missing[:3])
                raise FileNotFoundError(f"pair manifest references missing files: {preview}")

    def __len__(self):
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        height, width = self.image_shape
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        if self.mean is not None:
            tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        candidate_masks_a = _interval_masks(
            record.candidate_intervals_a, record.token_count
        )
        candidate_masks_b = _interval_masks(
            record.candidate_intervals_b, record.token_count
        )
        opening_target_a = candidate_masks_a.any(dim=0).to(torch.float32)
        opening_target_b = candidate_masks_b.any(dim=0).to(torch.float32)
        affinity_target = torch.zeros(
            (record.token_count, record.token_count), dtype=torch.float32
        )
        target_indices_valid = (
            record.is_match
            and 0 <= record.target_candidate_a < len(candidate_masks_a)
            and 0 <= record.target_candidate_b < len(candidate_masks_b)
        )
        if target_indices_valid:
            affinity_target = torch.outer(
                candidate_masks_a[record.target_candidate_a].to(torch.float32),
                candidate_masks_b[record.target_candidate_b].to(torch.float32),
            )
        return {
            "pair_id": record.pair_id,
            "image_A": self._load_image(record.image_a),
            "image_B": self._load_image(record.image_b),
            "image_path_A": record.image_a,
            "image_path_B": record.image_b,
            "layout_path_A": record.layout_a or "",
            "layout_path_B": record.layout_b or "",
            "scene_id": record.scene_id,
            "floor_id": record.floor_id,
            "candidate_masks_A": candidate_masks_a,
            "candidate_masks_B": candidate_masks_b,
            "opening_target_A": opening_target_a,
            "opening_target_B": opening_target_b,
            "affinity_target_AB": affinity_target,
            "is_match": torch.tensor(record.is_match, dtype=torch.bool),
            "target_candidate_pair": torch.tensor(
                [record.target_candidate_a, record.target_candidate_b], dtype=torch.long
            ),
            "relative_transform_B_to_A": torch.tensor(
                record.relative_transform_b_to_a, dtype=torch.float32
            ),
            "relative_yaw_radians": torch.tensor(
                record.relative_yaw_radians, dtype=torch.float32
            ),
            "pose_valid": torch.tensor(record.pose_valid, dtype=torch.bool),
        }


class ZInDPairDataset(PanoramaPairDataset):
    """ZInD pair dataset using an explicit topology/covisibility manifest."""


def _pad_candidate_masks(samples: Sequence[Mapping[str, Any]], key: str):
    batch_size = len(samples)
    token_count = samples[0][key].shape[-1]
    max_candidates = max(sample[key].shape[0] for sample in samples)
    masks = torch.zeros(
        (batch_size, max_candidates, token_count), dtype=torch.bool
    )
    valid = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample[key].shape[0]
        masks[index, :count] = sample[key]
        valid[index, :count] = True
    return masks, valid


def collate_panorama_pairs(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable candidate counts while stacking dense training targets."""
    if not samples:
        raise ValueError("cannot collate an empty panorama pair batch")
    candidate_masks_a, candidate_valid_a = _pad_candidate_masks(
        samples, "candidate_masks_A"
    )
    candidate_masks_b, candidate_valid_b = _pad_candidate_masks(
        samples, "candidate_masks_B"
    )
    string_keys = (
        "pair_id",
        "image_path_A",
        "image_path_B",
        "layout_path_A",
        "layout_path_B",
        "scene_id",
        "floor_id",
    )
    tensor_keys = (
        "image_A",
        "image_B",
        "opening_target_A",
        "opening_target_B",
        "affinity_target_AB",
        "is_match",
        "target_candidate_pair",
        "relative_transform_B_to_A",
        "relative_yaw_radians",
        "pose_valid",
    )
    output: Dict[str, Any] = {
        key: [sample[key] for sample in samples] for key in string_keys
    }
    output.update({key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys})
    output.update(
        candidate_masks_A=candidate_masks_a,
        candidate_masks_B=candidate_masks_b,
        candidate_valid_A=candidate_valid_a,
        candidate_valid_B=candidate_valid_b,
    )
    return output


def build_pair_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    dataset = PanoramaPairDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_panorama_pairs,
    )
