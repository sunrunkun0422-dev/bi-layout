"""Pair-level view of frozen Bi-Layout features for ZInD-BiPair training.

The single-view Opening Head training cache stores each panorama exactly once.
This module joins those cached predictions back to every pair record so the
cross-scene matcher can be trained without recomputing the frozen backbone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.zind_bipair_dataset import ZInDBiPairDataset


_CACHE_FILES = {
    "features": "features.npy",
    "enclosed_depth": "enclosed_depth.npy",
    "extended_depth": "extended_depth.npy",
}


class ZInDBiPairFeatureDataset(Dataset):
    """Join dense pair supervision with frozen single-view model outputs.

    Returned ``depth_enclosed_*`` and ``depth_extended_*`` fields are the
    frozen Bi-Layout predictions used to train the Opening Head.  The original
    dataset depths are retained as ``*_gt_*`` fields for diagnostics only.
    """

    def __init__(
        self,
        manifest_path: str,
        feature_cache_dir: str,
        data_root: Optional[str] = None,
        max_pairs: Optional[int] = None,
        validate_paths: bool = True,
    ) -> None:
        self.pairs = ZInDBiPairDataset(
            manifest_path,
            data_root=data_root,
            load_images=False,
            validate_paths=validate_paths,
        )
        if max_pairs is not None and (type(max_pairs) is not int or max_pairs <= 0):
            raise ValueError("max_pairs must be a positive integer when provided")
        self.max_pairs = max_pairs

        self.cache_dir = Path(feature_cache_dir).expanduser().resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "frozen feature cache metadata does not exist: {}".format(metadata_path)
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, Mapping) or metadata.get("complete") is not True:
            raise ValueError("frozen feature cache is incomplete: {}".format(self.cache_dir))
        self.metadata = dict(metadata)
        self.sample_count = int(self.metadata.get("sample_count", 0))
        self.token_count = int(self.metadata.get("token_count", 0))
        self.feature_dim = int(self.metadata.get("feature_dim", 0))
        if min(self.sample_count, self.token_count, self.feature_dim) <= 0:
            raise ValueError("feature cache metadata has invalid tensor dimensions")
        if self.metadata.get("branch_order") not in (None, "extended_first"):
            raise ValueError(
                "matcher training requires the extended_first opening cache contract"
            )

        views = self.metadata.get("views")
        if not isinstance(views, list) or len(views) != self.sample_count:
            raise ValueError("feature cache metadata views do not match sample_count")
        self._view_index: Dict[str, int] = {}
        for index, view in enumerate(views):
            if not isinstance(view, Mapping):
                raise ValueError("feature cache view metadata must be objects")
            for key in ("image_path", "relative_image_path"):
                value = view.get(key)
                if value:
                    normalized = self._normalize_path(value)
                    previous = self._view_index.setdefault(normalized, index)
                    if previous != index:
                        raise ValueError(
                            "feature cache contains duplicate image path: {}".format(value)
                        )

        # The builder stores one record per matched portal.  A small number of
        # panorama pairs share multiple portals, so treating those records as
        # independent would incorrectly supervise the other true portal to
        # the dustbin.  Keep one directed view-pair sample and merge all of its
        # portal/affinity labels at read time.
        grouped: Dict[Any, list] = {}
        for record_index, record in enumerate(self.pairs.records):
            key = (
                self._normalize_path(record["view_A"]["image_path"]),
                self._normalize_path(record["view_B"]["image_path"]),
                bool(record["is_positive"]),
            )
            grouped.setdefault(key, []).append(record_index)
        self._pair_groups = list(grouped.values())

        self._array_paths = {
            name: self.cache_dir / filename for name, filename in _CACHE_FILES.items()
        }
        missing = [str(path) for path in self._array_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "frozen feature cache is missing arrays: {}".format(", ".join(missing))
            )
        self._features = None
        self._enclosed_depth = None
        self._extended_depth = None
        self._open_arrays_and_validate()
        self._validate_pair_coverage()

    def _normalize_path(self, value: Any) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.pairs.data_root / path
        return str(path.resolve())

    def _open_arrays_and_validate(self) -> None:
        self._features = np.load(
            str(self._array_paths["features"]), mmap_mode="r", allow_pickle=False
        )
        self._enclosed_depth = np.load(
            str(self._array_paths["enclosed_depth"]), mmap_mode="r", allow_pickle=False
        )
        self._extended_depth = np.load(
            str(self._array_paths["extended_depth"]), mmap_mode="r", allow_pickle=False
        )
        expected_features = (self.sample_count, self.token_count, self.feature_dim)
        expected_depth = (self.sample_count, self.token_count)
        if tuple(self._features.shape) != expected_features:
            raise ValueError(
                "cached features have shape {}, expected {}".format(
                    tuple(self._features.shape), expected_features
                )
            )
        for name, value in (
            ("enclosed_depth", self._enclosed_depth),
            ("extended_depth", self._extended_depth),
        ):
            if tuple(value.shape) != expected_depth:
                raise ValueError(
                    "cached {} has shape {}, expected {}".format(
                        name, tuple(value.shape), expected_depth
                    )
                )

    def _validate_pair_coverage(self) -> None:
        missing = []
        for group in self._pair_groups[: len(self)]:
            record = self.pairs.records[group[0]]
            for side in ("A", "B"):
                path = self._normalize_path(record["view_{}".format(side)]["image_path"])
                if path not in self._view_index:
                    missing.append(path)
                    if len(missing) >= 3:
                        break
            if len(missing) >= 3:
                break
        if missing:
            raise ValueError(
                "feature cache does not cover pair manifest views: {}".format(
                    ", ".join(missing)
                )
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_features"] = None
        state["_enclosed_depth"] = None
        state["_extended_depth"] = None
        return state

    def _ensure_arrays_open(self) -> None:
        if self._features is None:
            self._open_arrays_and_validate()

    def __len__(self) -> int:
        length = len(self._pair_groups)
        return length if self.max_pairs is None else min(length, self.max_pairs)

    @staticmethod
    def _tensor(row: np.ndarray, name: str, pair_id: str) -> torch.Tensor:
        value = np.array(row, dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError(
                "cached {} for pair {} contains non-finite values".format(name, pair_id)
            )
        return torch.from_numpy(value)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_arrays_open()
        group = self._pair_groups[index]
        sample = dict(self.pairs[group[0]])
        source_pair_ids = [str(sample["pair_id"])]
        for record_index in group[1:]:
            other = self.pairs[record_index]
            source_pair_ids.append(str(other["pair_id"]))
            for key in ("opening_mask_all_A", "opening_mask_all_B"):
                if not torch.equal(sample[key], other[key]):
                    raise ValueError(
                        "duplicate view pair has inconsistent {} labels: {}".format(
                            key, source_pair_ids
                        )
                    )
            if not torch.allclose(sample["T_B_to_A"], other["T_B_to_A"], atol=1e-5):
                raise ValueError(
                    "duplicate view pair has inconsistent T_B_to_A: {}".format(
                        source_pair_ids
                    )
                )
            sample["portal_mask_A"] = sample["portal_mask_A"].bool().logical_or(
                other["portal_mask_A"].bool()
            ).to(sample["portal_mask_A"].dtype)
            sample["portal_mask_B"] = sample["portal_mask_B"].bool().logical_or(
                other["portal_mask_B"].bool()
            ).to(sample["portal_mask_B"].dtype)
            sample["affinity_gt"] = sample["affinity_gt"].bool().logical_or(
                other["affinity_gt"].bool()
            ).to(sample["affinity_gt"].dtype)
        if len(group) > 1:
            sample["pair_id"] = "{}__merged_{}".format(sample["pair_id"], len(group))
        sample["source_pair_ids"] = tuple(source_pair_ids)
        sample["merged_pair_count"] = torch.tensor(len(group), dtype=torch.long)
        pair_id = str(sample["pair_id"])
        sample["depth_enclosed_gt_A"] = sample["depth_enclosed_A"]
        sample["depth_enclosed_gt_B"] = sample["depth_enclosed_B"]
        sample["depth_extended_gt_A"] = sample["depth_extended_A"]
        sample["depth_extended_gt_B"] = sample["depth_extended_B"]
        for side in ("A", "B"):
            image_path = self._normalize_path(sample["image_path_{}".format(side)])
            view_index = self._view_index[image_path]
            sample["feature_{}".format(side)] = self._tensor(
                self._features[view_index], "features", pair_id
            )
            sample["depth_enclosed_{}".format(side)] = self._tensor(
                self._enclosed_depth[view_index], "enclosed_depth", pair_id
            )
            sample["depth_extended_{}".format(side)] = self._tensor(
                self._extended_depth[view_index], "extended_depth", pair_id
            )
            sample["feature_cache_index_{}".format(side)] = torch.tensor(
                view_index, dtype=torch.long
            )
        return sample


__all__ = ["ZInDBiPairFeatureDataset"]
