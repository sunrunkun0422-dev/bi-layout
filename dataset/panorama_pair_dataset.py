"""Manifest-based panorama pair loading for cross-scene training and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class PanoramaPairRecord:
    pair_id: str
    image_a: str
    image_b: str
    layout_a: Optional[str] = None
    layout_b: Optional[str] = None
    scene_id: str = ""
    floor_id: str = ""

    @classmethod
    def from_mapping(cls, record: Mapping, base_dir: Path, index: int):
        image_a = _first(record, ("image_a", "imageA", "path_A", "pathA"), required=True)
        image_b = _first(record, ("image_b", "imageB", "path_B", "pathB"), required=True)
        pair_id = str(_first(record, ("id", "pair_id", "pairId")) or f"pair_{index:06d}")
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
        )


def load_pair_manifest(path: str) -> List[PanoramaPairRecord]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pair manifest does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".jsonl":
        raw_records = []
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

    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("pair manifest must contain a non-empty list or a {'pairs': [...]} object")
    return [
        PanoramaPairRecord.from_mapping(record, manifest_path.parent, index)
        for index, record in enumerate(raw_records)
    ]


class PanoramaPairDataset(Dataset):
    """Load paired RGB panoramas using a stable manifest contract."""

    def __init__(
        self,
        manifest_path: str,
        image_shape: Tuple[int, int] = (512, 1024),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        validate_paths: bool = True,
    ):
        self.records = load_pair_manifest(manifest_path)
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must contain positive (height, width)")
        self.mean = None if mean is None else torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = None if std is None else torch.tensor(std, dtype=torch.float32)[:, None, None]
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be provided together")
        if self.std is not None and (self.std <= 0).any():
            raise ValueError("normalization std values must be positive")

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
        }


class ZInDPairDataset(PanoramaPairDataset):
    """ZInD pair dataset using an explicit topology/covisibility manifest."""


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
    )
