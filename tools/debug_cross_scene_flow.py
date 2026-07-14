#!/usr/bin/env python3
"""Run one panorama pair through every cross-scene module and report formats."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.defaults import merge_from_file
from dataset.panorama_pair_dataset import build_pair_dataloader
from models.bi_layout import Bi_Layout
from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    candidate_intervals_to_mask,
)
from models.geometry_consistency_selector import (
    GeometryConsistencySelector,
    candidate_metrics_to_tensor,
)
from postprocessing.post_process import post_process
from utils.conversion import depth2xyz
from utils.cross_scene_estimator import (
    estimate_wall_pair_candidates,
    extract_opening_candidates,
    polygon_validity,
)
from utils.cross_scene_pipeline import (
    CrossScenePipeline,
    CrossScenePipelineConfig,
    atomic_write_json,
)
from utils.writer import xyz2json


DEFAULT_ZIND_PANO_DIR = REPO_ROOT.parent / "zind/data/0000/panos"
DEFAULT_IMAGE_A = DEFAULT_ZIND_PANO_DIR / "floor_01_partial_room_04_pano_32.jpg"
DEFAULT_IMAGE_B = DEFAULT_ZIND_PANO_DIR / "floor_01_partial_room_08_pano_31.jpg"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save the data contract at every cross-scene pipeline module."
    )
    parser.add_argument("--image_a", default=str(DEFAULT_IMAGE_A))
    parser.add_argument("--image_b", default=str(DEFAULT_IMAGE_B))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default="src/output/cross_scene_format_smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--random_bi_layout",
        action="store_true",
        help="skip the Bi-Layout checkpoint; useful only for interface debugging",
    )
    return parser.parse_args()


def _tensor_format(value: torch.Tensor) -> Dict[str, Any]:
    return {
        "pythonType": "torch.Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "requiresGrad": bool(value.requires_grad),
    }


def describe_format(value: Any, depth: int = 0, max_depth: int = 4) -> Dict[str, Any]:
    """Describe structure without serializing tensor contents."""
    if isinstance(value, torch.Tensor):
        return _tensor_format(value)
    if isinstance(value, np.ndarray):
        return {
            "pythonType": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        output = {
            "pythonType": type(value).__name__,
            "keys": [str(key) for key in value.keys()],
        }
        if depth < max_depth:
            output["fields"] = {
                str(key): describe_format(item, depth + 1, max_depth)
                for key, item in value.items()
            }
        return output
    if isinstance(value, (list, tuple)):
        output = {"pythonType": type(value).__name__, "length": len(value)}
        if value and depth < max_depth:
            output["itemFormat"] = describe_format(value[0], depth + 1, max_depth)
        return output
    if value is None:
        return {"pythonType": "NoneType"}
    return {"pythonType": type(value).__name__}


class FlowReporter:
    def __init__(self):
        self.modules = []

    def add(
        self,
        module_id: str,
        name: str,
        inputs: Any,
        outputs: Any,
        note: str = "",
    ) -> None:
        entry = OrderedDict(
            module=module_id,
            name=name,
            status="PASS",
            inputFormat=describe_format(inputs),
            outputFormat=describe_format(outputs),
            note=note,
        )
        self.modules.append(entry)
        print("\n" + "=" * 88)
        print("[{}] {}: PASS".format(module_id, name))
        if note:
            print("说明: {}".format(note))
        print("输入格式:")
        print(json.dumps(entry["inputFormat"], ensure_ascii=False, indent=2))
        print("输出格式:")
        print(json.dumps(entry["outputFormat"], ensure_ascii=False, indent=2))

    def payload(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "formatVersion": 1,
            "status": "PASS",
            "moduleCount": len(self.modules),
            "metadata": dict(metadata),
            "modules": self.modules,
        }


def load_bi_layout(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    load_checkpoint: bool,
) -> Tuple[Bi_Layout, Dict[str, Any]]:
    config = merge_from_file(config_path)
    model_args = dict(config.MODEL.ARGS[0])

    # torchvision resolves pretrained ResNet weights from <hub_dir>/checkpoints.
    torch.hub.set_dir(str(REPO_ROOT))
    model = Bi_Layout(**model_args)
    checkpoint_report = {
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded": False,
        "modelArgs": model_args,
    }
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("net", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        checkpoint_report.update(
            loaded=True,
            epoch=checkpoint.get("epoch"),
            missingKeys=list(incompatible.missing_keys),
            unexpectedKeys=list(incompatible.unexpected_keys),
        )
        del checkpoint, state_dict
        gc.collect()

    model.to(device).eval()
    return model, checkpoint_report


def _top_intervals(probability: torch.Tensor, count: int = 4, radius: int = 4):
    values = probability.detach().cpu().reshape(-1)
    token_count = len(values)
    selected = []
    suppressed = torch.zeros(token_count, dtype=torch.bool)
    for _ in range(min(count, token_count)):
        score = values.masked_fill(suppressed, -1.0)
        center = int(score.argmax().item())
        if float(score[center]) < 0:
            break
        selected.append(((center - radius) % token_count, (center + radius) % token_count))
        offsets = torch.arange(-2 * radius, 2 * radius + 1)
        suppressed[(center + offsets) % token_count] = True
    return selected


def _convex_layout(depth: np.ndarray, ratio: float) -> Dict[str, Any]:
    xyz = depth2xyz(np.maximum(np.abs(depth), 1e-3))
    hull = cv2.convexHull(xyz[..., ::2].astype(np.float32))[:, 0, :]
    if len(hull) < 3:
        extent = max(float(np.max(np.abs(xyz[..., ::2]))), 1.0)
        hull = np.asarray(
            [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]],
            dtype=np.float32,
        )
    hull_xyz = np.insert(hull, 1, 1.0, axis=1)
    return xyz2json(hull_xyz, max(abs(float(ratio)), 1e-3))


def prediction_to_layout(depth: torch.Tensor, ratio: torch.Tensor) -> Tuple[Dict, Dict]:
    depth_np = depth.detach().cpu().numpy().reshape(1, -1)
    ratio_value = float(ratio.detach().cpu().reshape(-1)[0])
    report = {"postProcessing": "manhattan", "fallbackUsed": False}
    try:
        processed_xyz = post_process(np.maximum(np.abs(depth_np), 1e-3), type_name="manhattan")[0]
        layout = xyz2json(processed_xyz, max(abs(ratio_value), 1e-3))
        validity = polygon_validity(layout)
        if not validity["valid"]:
            raise ValueError("Manhattan output failed polygon validation: {}".format(validity))
    except Exception as exc:
        layout = _convex_layout(depth_np[0], ratio_value)
        report.update(
            postProcessing="convex_hull_fallback",
            fallbackUsed=True,
            fallbackReason=str(exc),
        )
    report["validity"] = polygon_validity(layout)
    return layout, report


def candidate_formats(candidates: Sequence[Any]) -> Dict[str, Any]:
    return {
        "candidateCount": len(candidates),
        "candidateList": [candidate.to_json() for candidate in candidates],
    }


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.torch_threads <= 0:
        raise ValueError("top_k and torch_threads must be positive")
    image_a = Path(args.image_a).expanduser().resolve()
    image_b = Path(args.image_b).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    for path in (image_a, image_b, Path(args.config).expanduser().resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.random_bi_layout and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pair_manifest.json"
    manifest = {
        "pairs": [
            {
                "id": "zind_0000_bathroom32_laundry31",
                "scene_id": "0000",
                "floor_id": "floor_01",
                "image_a": str(image_a),
                "image_b": str(image_b),
            }
        ]
    }
    atomic_write_json(str(manifest_path), manifest)

    reporter = FlowReporter()
    reporter.add(
        "M0",
        "相邻全景配对清单",
        {"image_A": str(image_a), "image_B": str(image_b)},
        manifest,
        "同一 ZInD 楼层、不同房间；pano_32 与 pano_31 的标注平移距离约 0.176。",
    )

    loader = build_pair_dataloader(str(manifest_path), batch_size=1, workers=0)
    batch = next(iter(loader))
    reporter.add("M1", "双全景 Dataset/DataLoader", manifest, batch)

    model, checkpoint_report = load_bi_layout(
        args.config,
        str(checkpoint_path),
        device,
        load_checkpoint=not args.random_bi_layout,
    )
    image_tensor_a = batch["image_A"].to(device)
    image_tensor_b = batch["image_B"].to(device)
    with torch.no_grad():
        layout_output_a = model(image_tensor_a, return_features=True)
        layout_output_b = model(image_tensor_b, return_features=True)
    reporter.add(
        "M2",
        "共享 Bi-Layout 双头编码",
        {"image_A": image_tensor_a, "image_B": image_tensor_b},
        {"layout_A": layout_output_a, "layout_B": layout_output_b},
        "Bi-Layout checkpoint loaded={}.".format(checkpoint_report["loaded"]),
    )

    matcher = OpeningGuidedCrossAttentionMatcher(feature_dim=model.patch_dim).to(device).eval()
    with torch.no_grad():
        first_matches = matcher(
            layout_output_a["layout_feature"],
            layout_output_b["layout_feature"],
            layout_output_a["depth"],
            layout_output_a["new_depth"],
            layout_output_b["depth"],
            layout_output_b["new_depth"],
        )
    reporter.add(
        "M3",
        "开口响应与外延深度",
        {
            "features_A": layout_output_a["layout_feature"],
            "D_A_enc": layout_output_a["depth"],
            "D_A_ext": layout_output_a["new_depth"],
            "features_B": layout_output_b["layout_feature"],
            "D_B_enc": layout_output_b["depth"],
            "D_B_ext": layout_output_b["new_depth"],
        },
        {
            key: first_matches[key]
            for key in ("P_A_open", "P_B_open", "G_A_open", "G_B_open", "Delta_A", "Delta_B")
        },
        "Opening head is currently initialized but has no dedicated trained checkpoint.",
    )
    reporter.add(
        "M4",
        "双向 Cross Attention 匹配",
        {
            "layout_feature_A": layout_output_a["layout_feature"],
            "layout_feature_B": layout_output_b["layout_feature"],
            "opening_A": first_matches["P_A_open"],
            "opening_B": first_matches["P_B_open"],
        },
        {
            key: first_matches[key]
            for key in ("Aff_AB", "Aff_BA", "S_A", "S_B", "cross_feature_A", "cross_feature_B")
        },
    )
    reporter.add(
        "M5",
        "循环位移与相对 yaw",
        first_matches["Aff_AB"],
        {
            key: first_matches[key]
            for key in (
                "cyclic_shift_mass",
                "cyclic_shift_score",
                "best_cyclic_shift",
                "relative_yaw_radians",
            )
        },
    )

    intervals_a = _top_intervals(first_matches["P_A_open"][0])
    intervals_b = _top_intervals(first_matches["P_B_open"][0])
    masks_a = candidate_intervals_to_mask(intervals_a, model.patch_num, device=device)
    masks_b = candidate_intervals_to_mask(intervals_b, model.patch_num, device=device)
    with torch.no_grad():
        matches = matcher(
            layout_output_a["layout_feature"],
            layout_output_b["layout_feature"],
            layout_output_a["depth"],
            layout_output_a["new_depth"],
            layout_output_b["depth"],
            layout_output_b["new_depth"],
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
        )
    reporter.add(
        "M6",
        "开口 Token 池化与候选配对",
        {"intervals_A": intervals_a, "intervals_B": intervals_b, "masks_A": masks_a, "masks_B": masks_b},
        {
            key: matches[key]
            for key in (
                "E_A_open",
                "E_B_open",
                "candidate_logits",
                "candidate_affinity",
                "candidate_pair_score",
                "best_candidate_pair",
            )
        },
    )

    layout_a, layout_report_a = prediction_to_layout(
        layout_output_a["depth"], layout_output_a["ratio"]
    )
    layout_b, layout_report_b = prediction_to_layout(
        layout_output_b["depth"], layout_output_b["ratio"]
    )
    extended_a, extended_report_a = prediction_to_layout(
        layout_output_a["new_depth"], layout_output_a["ratio"]
    )
    extended_b, extended_report_b = prediction_to_layout(
        layout_output_b["new_depth"], layout_output_b["ratio"]
    )
    openings_a, opening_summary_a = extract_opening_candidates(layout_a, extended_a)
    openings_b, opening_summary_b = extract_opening_candidates(layout_b, extended_b)
    candidates, best_joint = estimate_wall_pair_candidates(
        layout_a,
        layout_b,
        top_k=args.top_k,
        openings_a=openings_a,
        openings_b=openings_b,
        match_evidence=matches,
    )
    geometry_output = {
        "layout_A": layout_a,
        "layout_B": layout_b,
        "extended_layout_A": extended_a,
        "extended_layout_B": extended_b,
        "opening_summary_A": opening_summary_a,
        "opening_summary_B": opening_summary_b,
        "openings_A": [opening.to_json() for opening in openings_a],
        "openings_B": [opening.to_json() for opening in openings_b],
        "candidates": candidate_formats(candidates),
        "best_joint_layout": best_joint,
    }
    reporter.add(
        "M7",
        "布局转换、开口提取与几何候选",
        {
            "depth_A": layout_output_a["depth"],
            "new_depth_A": layout_output_a["new_depth"],
            "depth_B": layout_output_b["depth"],
            "new_depth_B": layout_output_b["new_depth"],
            "match_evidence": matches,
        },
        geometry_output,
        "Layout post-processing A/B: {}/{}.".format(
            layout_report_a["postProcessing"], layout_report_b["postProcessing"]
        ),
    )

    selector = GeometryConsistencySelector().to(device).eval()
    metric_tensor = candidate_metrics_to_tensor(
        [candidate.metrics for candidate in candidates], device=device
    )
    with torch.no_grad():
        selector_output = selector(metric_tensor)
    reporter.add(
        "M8",
        "几何一致性选择头",
        metric_tensor,
        selector_output,
        "Selector is randomly initialized because no selector checkpoint exists yet.",
    )

    pipeline = CrossScenePipeline(
        CrossScenePipelineConfig(top_k=args.top_k), selector=selector
    )
    pipeline_result = pipeline.run(
        layout_a,
        layout_b,
        extended_layout_a=extended_a,
        extended_layout_b=extended_b,
        match_evidence=matches,
    )
    final_payload = {
        "candidateReport": pipeline_result.candidates_json(
            metadata={"pair_id": batch["pair_id"][0]}
        ),
        "bestJointLayout": pipeline_result.best_joint_layout,
    }
    candidates_path = output_dir / "flow_candidates.json"
    joint_path = output_dir / "flow_best_joint_layout.json"
    atomic_write_json(str(candidates_path), final_payload["candidateReport"])
    atomic_write_json(str(joint_path), final_payload["bestJointLayout"])
    reporter.add(
        "M9",
        "统一工程流水线与原子化输出",
        {
            "layout_A": layout_a,
            "layout_B": layout_b,
            "extended_layout_A": extended_a,
            "extended_layout_B": extended_b,
            "match_evidence": matches,
        },
        final_payload,
        "Wrote {} and {}.".format(candidates_path.name, joint_path.name),
    )

    metadata = {
        "pairId": batch["pair_id"][0],
        "imageA": str(image_a),
        "imageB": str(image_b),
        "device": str(device),
        "checkpoint": checkpoint_report,
        "neuralModuleWeights": {
            "biLayout": "trained checkpoint" if checkpoint_report["loaded"] else "random",
            "openingCrossAttention": "random initialization",
            "geometrySelector": "random initialization",
        },
        "layoutConversion": {
            "A": layout_report_a,
            "B": layout_report_b,
            "AExtended": extended_report_a,
            "BExtended": extended_report_b,
        },
        "candidateCount": len(pipeline_result.candidates),
    }
    report_path = output_dir / "module_format_report.json"
    atomic_write_json(str(report_path), reporter.payload(metadata))
    print("\n" + "=" * 88)
    print("全流程 PASS: {} 个模块全部完成。".format(len(reporter.modules)))
    print("格式报告: {}".format(report_path))
    print("候选结果: {}".format(candidates_path))
    print("联合布局: {}".format(joint_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
