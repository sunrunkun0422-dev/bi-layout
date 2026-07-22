"""Single-view ZInD opening datasets and synchronized token augmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_CACHE_ARRAY_FILES = {
    "features": "features.npy",
    "enclosed_depth": "enclosed_depth.npy",
    "extended_depth": "extended_depth.npy",
    "targets": "targets.npy",
}


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON: {}".format(path)) from exc
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSONL at {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    "record at {}:{} is not an object".format(path, line_number)
                )
            records.append(value)
    if not records:
        raise ValueError("manifest contains no pair records: {}".format(path))
    return records


def _positive_int(metadata: Mapping[str, Any], key: str) -> int:
    if key not in metadata or type(metadata[key]) is not int:
        raise ValueError("cache metadata requires integer {!r}".format(key))
    value = metadata[key]
    if value <= 0:
        raise ValueError("cache metadata {!r} must be a positive integer".format(key))
    return value


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("{} must be in [0, 1]".format(name))
    return value


def synchronize_opening_augmentation(
    feature: torch.Tensor,
    enclosed_depth: torch.Tensor,
    extended_depth: torch.Tensor,
    target: torch.Tensor,
    shift: int = 0,
    flip: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same circular roll and horizontal flip to one cached sample.

    The token dimension is dimension zero for all four tensors.  The circular
    roll is applied first, followed by an optional flip.  Inputs are not
    modified in place.
    """

    if not torch.is_tensor(feature) or feature.ndim != 2:
        raise ValueError("feature must be a tensor with shape [N, C]")
    token_count = int(feature.shape[0])
    if token_count <= 0 or int(feature.shape[1]) <= 0:
        raise ValueError("feature must have non-empty token and channel dimensions")
    for name, value in (
        ("enclosed_depth", enclosed_depth),
        ("extended_depth", extended_depth),
        ("target", target),
    ):
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError("{} must be a tensor with shape [N]".format(name))
        if int(value.shape[0]) != token_count:
            raise ValueError("{} token count does not match feature".format(name))
    if isinstance(shift, bool):
        raise ValueError("shift must be an integer")
    try:
        normalized_shift = int(shift)
    except (TypeError, ValueError) as exc:
        raise ValueError("shift must be an integer") from exc
    if normalized_shift != shift:
        raise ValueError("shift must be an integer")
    normalized_shift %= token_count

    values: Sequence[torch.Tensor] = (
        feature,
        enclosed_depth,
        extended_depth,
        target,
    )
    if normalized_shift:
        values = tuple(torch.roll(value, normalized_shift, dims=0) for value in values)
    if bool(flip):
        values = tuple(torch.flip(value, dims=(0,)) for value in values)
    return values[0], values[1], values[2], values[3]


class ZInDOpeningViewDataset(Dataset):
    """Load every panorama in a ZInD-BiPair manifest exactly once.

    Pair records are traversed in manifest order and side ``A`` before side
    ``B``.  The first occurrence of each literal ``image_path`` is retained,
    which makes both the sample order and returned ``index`` stable.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: Optional[str] = None,
        image_shape: Tuple[int, int] = (512, 1024),
        validate_paths: bool = True,
        max_views: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError("manifest does not exist: {}".format(self.manifest_path))
        self.dataset_root = self.manifest_path.parent.parent
        info_path = self.dataset_root / "dataset_info.json"
        if not info_path.is_file():
            raise FileNotFoundError("dataset_info.json does not exist: {}".format(info_path))
        self.dataset_info = dict(_read_json_object(info_path))

        token_value = self.dataset_info.get(
            "tokenCount", self.dataset_info.get("token_count")
        )
        if type(token_value) is not int or token_value <= 0:
            raise ValueError("dataset_info tokenCount must be a positive integer")
        self.token_count = token_value

        source = self.dataset_info.get("source", {})
        if not isinstance(source, Mapping):
            raise ValueError("dataset_info source must be an object")
        declared_root = source.get("dataRoot")
        if data_root is None and not declared_root:
            raise ValueError(
                "data_root is not provided and dataset_info has no source.dataRoot"
            )
        self.data_root = Path(data_root or declared_root).expanduser().resolve()
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must contain positive (height, width)")
        if max_views is not None:
            if type(max_views) is not int or max_views <= 0:
                raise ValueError("max_views must be a positive integer when provided")
        self.max_views = max_views

        records = _read_jsonl(self.manifest_path)
        views: List[Dict[str, Any]] = []
        seen_image_paths = set()
        for line_index, record in enumerate(records):
            label_cache = record.get("label_cache")
            if not isinstance(label_cache, str) or not label_cache:
                raise ValueError(
                    "manifest record {} has no valid label_cache".format(line_index)
                )
            for side in ("A", "B"):
                view = record.get("view_{}".format(side))
                if not isinstance(view, Mapping):
                    raise ValueError(
                        "manifest record {} has no valid view_{}".format(
                            line_index, side
                        )
                    )
                relative_image_path = view.get("image_path")
                if not isinstance(relative_image_path, str) or not relative_image_path:
                    raise ValueError(
                        "manifest record {} view_{} has no valid image_path".format(
                            line_index, side
                        )
                    )
                if relative_image_path in seen_image_paths:
                    continue
                seen_image_paths.add(relative_image_path)
                views.append(
                    {
                        "index": len(views),
                        "relative_image_path": relative_image_path,
                        "image_path": str(self.data_root / relative_image_path),
                        "relative_label_cache": label_cache,
                        "label_cache": str(self.dataset_root / label_cache),
                        "side": side,
                        "pair_id": str(record.get("pair_id", "")),
                        "house_id": str(record.get("house_id", "")),
                        "floor_id": str(record.get("floor_id", "")),
                        "complete_room_id": str(record.get("complete_room_id", "")),
                        "partial_room_id": str(view.get("partial_room_id", "")),
                        "pano_id": str(view.get("pano_id", "")),
                    }
                )
        if not views:
            raise ValueError("manifest contains no panorama views: {}".format(self.manifest_path))
        if self.max_views is not None:
            views = views[: self.max_views]
        self.views = views

        if validate_paths:
            missing: List[str] = []
            checked_label_paths = set()
            for view in self.views:
                image_path = Path(view["image_path"])
                if not image_path.is_file():
                    missing.append(str(image_path))
                label_path = Path(view["label_cache"])
                if label_path not in checked_label_paths:
                    checked_label_paths.add(label_path)
                    if not label_path.is_file():
                        missing.append(str(label_path))
                if len(missing) >= 3:
                    break
            if missing:
                raise FileNotFoundError(
                    "ZInD opening manifest references missing files: {}".format(
                        ", ".join(missing[:3])
                    )
                )

    def __len__(self) -> int:
        return len(self.views)

    def _load_image(self, path: Path) -> torch.Tensor:
        height, width = self.image_shape
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        expected_shape = (height, width, 3)
        if tuple(array.shape) != expected_shape or not np.isfinite(array).all():
            raise ValueError(
                "invalid image array at {}: expected {}, got {}".format(
                    path, expected_shape, tuple(array.shape)
                )
            )
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def _load_target(self, view: Mapping[str, Any]) -> torch.Tensor:
        path = Path(view["label_cache"])
        key = "opening_mask_all_{}".format(view["side"])
        with np.load(str(path), allow_pickle=False) as cache:
            if key not in cache.files:
                raise ValueError("{} is missing label {}".format(path, key))
            target = np.asarray(cache[key]).copy()
        if tuple(target.shape) != (self.token_count,):
            raise ValueError(
                "{} {} has shape {}, expected ({},)".format(
                    path, key, tuple(target.shape), self.token_count
                )
            )
        if not np.issubdtype(target.dtype, np.number) and target.dtype != np.bool_:
            raise ValueError("{} {} must be numeric or boolean".format(path, key))
        if not np.isfinite(target).all() or not np.logical_or(target == 0, target == 1).all():
            raise ValueError("{} {} must contain only binary finite values".format(path, key))
        return torch.from_numpy(target.astype(np.float32, copy=False).copy())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        view = self.views[index]
        return {
            "index": int(view["index"]),
            "image": self._load_image(Path(view["image_path"])),
            "target": self._load_target(view),
            **dict(view),
        }


class OpeningFeatureCacheDataset(Dataset):
    """Memory-map a complete frozen-Bi-Layout feature cache."""

    def __init__(
        self,
        cache_dir: str,
        augment: bool = False,
        roll_probability: float = 0.0,
        flip_probability: float = 0.0,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError("cache directory does not exist: {}".format(self.cache_dir))
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("cache metadata does not exist: {}".format(metadata_path))
        self.metadata = dict(_read_json_object(metadata_path))
        if self.metadata.get("complete") is not True:
            raise ValueError("opening feature cache is incomplete: {}".format(self.cache_dir))
        self.sample_count = _positive_int(self.metadata, "sample_count")
        self.token_count = _positive_int(self.metadata, "token_count")
        self.feature_dim = _positive_int(self.metadata, "feature_dim")
        views = self.metadata.get("views")
        if not isinstance(views, list) or len(views) != self.sample_count:
            raise ValueError(
                "cache metadata views must be a list with sample_count entries"
            )
        if any(not isinstance(view, Mapping) for view in views):
            raise ValueError("every cache metadata view must be an object")
        self.views = [dict(view) for view in views]
        for index, view in enumerate(self.views):
            declared_index = view.get("index")
            if declared_index is not None and (
                type(declared_index) is not int or declared_index != index
            ):
                raise ValueError(
                    "cache metadata view {} has an invalid index".format(index)
                )

        self.augment = bool(augment)
        self.roll_probability = _validate_probability(
            roll_probability, "roll_probability"
        )
        self.flip_probability = _validate_probability(
            flip_probability, "flip_probability"
        )
        self._array_paths = {
            name: self.cache_dir / filename
            for name, filename in _CACHE_ARRAY_FILES.items()
        }
        missing = [str(path) for path in self._array_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "opening feature cache is missing files: {}".format(
                    ", ".join(missing)
                )
            )

        self._features = None
        self._enclosed_depth = None
        self._extended_depth = None
        self._targets = None
        self._open_arrays_and_validate()

    def _open_arrays_and_validate(self) -> None:
        try:
            self._features = np.load(
                str(self._array_paths["features"]), mmap_mode="r", allow_pickle=False
            )
            self._enclosed_depth = np.load(
                str(self._array_paths["enclosed_depth"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            self._extended_depth = np.load(
                str(self._array_paths["extended_depth"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            self._targets = np.load(
                str(self._array_paths["targets"]), mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                "failed to memory-map opening feature cache: {}".format(self.cache_dir)
            ) from exc

        expected_feature_shape = (
            self.sample_count,
            self.token_count,
            self.feature_dim,
        )
        expected_token_shape = (self.sample_count, self.token_count)
        shapes = {
            "features": (self._features, expected_feature_shape),
            "enclosed_depth": (self._enclosed_depth, expected_token_shape),
            "extended_depth": (self._extended_depth, expected_token_shape),
            "targets": (self._targets, expected_token_shape),
        }
        for name, (array, expected_shape) in shapes.items():
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    "{} has shape {}, expected {}".format(
                        self._array_paths[name], tuple(array.shape), expected_shape
                    )
                )
        for name, array in (
            ("features", self._features),
            ("enclosed_depth", self._enclosed_depth),
            ("extended_depth", self._extended_depth),
        ):
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError("{} must have a floating dtype".format(self._array_paths[name]))
        if not (
            np.issubdtype(self._targets.dtype, np.number)
            or self._targets.dtype == np.bool_
        ):
            raise ValueError("{} must be numeric or boolean".format(self._array_paths["targets"]))

    def _ensure_arrays_open(self) -> None:
        if self._features is None:
            self._open_arrays_and_validate()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_features"] = None
        state["_enclosed_depth"] = None
        state["_extended_depth"] = None
        state["_targets"] = None
        return state

    def __len__(self) -> int:
        return self.sample_count

    @staticmethod
    def _float_tensor(row: np.ndarray, name: str, index: int) -> torch.Tensor:
        value = np.array(row, dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("cache {} sample {} contains non-finite values".format(name, index))
        return torch.from_numpy(value)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_arrays_open()
        feature = self._float_tensor(self._features[index], "features", index)
        enclosed_depth = self._float_tensor(
            self._enclosed_depth[index], "enclosed_depth", index
        )
        extended_depth = self._float_tensor(
            self._extended_depth[index], "extended_depth", index
        )
        target = self._float_tensor(self._targets[index], "targets", index)
        if not torch.logical_or(target == 0, target == 1).all():
            raise ValueError("cache targets sample {} is not binary".format(index))

        if self.augment:
            shift = 0
            if self.roll_probability > 0.0 and bool(
                torch.rand(()) < self.roll_probability
            ):
                shift = int(torch.randint(self.token_count, (1,)).item())
            flip = self.flip_probability > 0.0 and bool(
                torch.rand(()) < self.flip_probability
            )
            feature, enclosed_depth, extended_depth, target = (
                synchronize_opening_augmentation(
                    feature,
                    enclosed_depth,
                    extended_depth,
                    target,
                    shift=shift,
                    flip=flip,
                )
            )

        return {
            "index": int(index),
            "feature": feature,
            "enclosed_depth": enclosed_depth,
            "extended_depth": extended_depth,
            "target": target,
        }


__all__ = [
    "OpeningFeatureCacheDataset",
    "ZInDOpeningViewDataset",
    "synchronize_opening_augmentation",
]
