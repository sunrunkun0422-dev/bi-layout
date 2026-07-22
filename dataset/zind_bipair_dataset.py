"""PyTorch loader for generated ZInD-BiPair-v1 JSONL/NPZ data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"manifest contains no pair records: {path}")
    return records


class ZInDBiPairDataset(Dataset):
    """Load paired panoramas and dense labels produced by build_zind_bipair_v1."""

    def __init__(
        self,
        manifest_path: str,
        data_root: Optional[str] = None,
        image_shape: Tuple[int, int] = (512, 1024),
        load_images: bool = True,
        validate_paths: bool = True,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {self.manifest_path}")
        self.dataset_root = self.manifest_path.parent.parent
        info_path = self.dataset_root / "dataset_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"dataset_info.json does not exist: {info_path}")
        with info_path.open("r", encoding="utf-8") as file:
            self.dataset_info = json.load(file)
        declared_root = self.dataset_info.get("source", {}).get("dataRoot")
        if data_root is None and not declared_root:
            raise ValueError("data_root is not provided and dataset_info has no source.dataRoot")
        self.data_root = Path(data_root or declared_root).expanduser().resolve()
        self.records = _read_jsonl(self.manifest_path)
        self.load_images = bool(load_images)
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must contain positive (height, width)")
        if validate_paths:
            missing = []
            for record in self.records:
                paths = (
                    self.data_root / record["view_A"]["image_path"],
                    self.data_root / record["view_B"]["image_path"],
                    self.dataset_root / record["label_cache"],
                )
                missing.extend(str(path) for path in paths if not path.is_file())
                if len(missing) >= 3:
                    break
            if missing:
                raise FileNotFoundError(
                    "ZInD-BiPair manifest references missing files: "
                    + ", ".join(missing[:3])
                )

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> torch.Tensor:
        height, width = self.image_shape
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        label_path = self.dataset_root / record["label_cache"]
        with np.load(label_path, allow_pickle=False) as cache:
            labels = {
                key: torch.from_numpy(np.asarray(cache[key]).copy()) for key in cache.files
            }
        output: Dict[str, Any] = {
            **labels,
            "pair_id": record["pair_id"],
            "pair_type": record["pair_type"],
            "house_id": record["house_id"],
            "floor_id": record["floor_id"],
            "complete_room_id": record["complete_room_id"],
            "image_path_A": str(self.data_root / record["view_A"]["image_path"]),
            "image_path_B": str(self.data_root / record["view_B"]["image_path"]),
            "partial_room_id_A": record["view_A"]["partial_room_id"],
            "partial_room_id_B": record["view_B"]["partial_room_id"],
            "pano_id_A": record["view_A"]["pano_id"],
            "pano_id_B": record["view_B"]["pano_id"],
            "is_positive": torch.tensor(record["is_positive"], dtype=torch.bool),
        }
        if self.load_images:
            output["image_A"] = self._load_image(Path(output["image_path_A"]))
            output["image_B"] = self._load_image(Path(output["image_path_B"]))
        return output


VARIABLE_LABEL_KEYS = (
    "corners_enclosed_A",
    "corners_extended_A",
    "corners_enclosed_B",
    "corners_extended_B",
    "joint_layout_global",
    "joint_layout_global_A",
    "joint_layout_global_B",
    "shared_portal_global",
)


def collate_zind_bipair(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty ZInD-BiPair batch")
    string_keys = (
        "pair_id",
        "pair_type",
        "house_id",
        "floor_id",
        "complete_room_id",
        "image_path_A",
        "image_path_B",
        "partial_room_id_A",
        "partial_room_id_B",
        "pano_id_A",
        "pano_id_B",
    )
    output: Dict[str, Any] = {
        key: [sample[key] for sample in samples] for key in string_keys
    }
    for key in samples[0]:
        if key in string_keys:
            continue
        if key in VARIABLE_LABEL_KEYS:
            output[key] = [sample[key] for sample in samples]
            continue
        if torch.is_tensor(samples[0][key]):
            output[key] = torch.stack([sample[key] for sample in samples])
    return output


def build_zind_bipair_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    dataset = ZInDBiPairDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_zind_bipair,
    )
