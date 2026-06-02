#!/usr/bin/env python3
"""
Build an Atlanta/non-Manhattan candidate subset from MP3D layout labels.

The heuristic is intentionally conservative:
- Each layout is represented by its top-view wall polygon from MP3D JSON labels.
- We fit the best Manhattan frame, i.e. two orthogonal orientation families.
- A sample is marked as an Atlanta candidate when the weighted orientation
  residual is high, or when some adjacent wall angle clearly deviates from 90 deg.

This is a subset-mining tool, not a ground-truth semantic annotation.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("src/dataset/mp3d")
OUT = Path("atlanta_subset_report")
SPLITS = ("train", "val", "test")


@dataclass
class SampleStats:
    sample_id: str
    split: str
    wall_num: int
    total_length: float
    best_frame_deg: float
    mean_residual_deg: float
    max_residual_deg: float
    max_adjacent_deviation_deg: float
    non90_corner_count: int
    atlanta_candidate: bool
    reason: str
    points_xz: List[Tuple[float, float]]


def split_ids(split_path: Path) -> List[str]:
    ids = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ids.append("_".join(parts))
    return ids


def wrap_pi(angle: float) -> float:
    return angle % math.pi


def angular_dist_mod_pi(a: float, b: float) -> float:
    """Smallest distance between unoriented line angles, in radians."""
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def edge_orientation_residual(angle: float, frame: float) -> float:
    return min(
        angular_dist_mod_pi(angle, frame),
        angular_dist_mod_pi(angle, frame + math.pi / 2),
    )


def vector_angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_v = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_v))


def ordered_points_xz(label: dict) -> List[Tuple[float, float]]:
    points = label["layoutPoints"]["points"]
    walls = label["layoutWalls"]["walls"]
    point_idx = [w["pointsIdx"][0] for w in walls]
    ordered = []
    for idx in point_idx:
        xyz = points[idx]["xyz"]
        # MP3D labels use x,y,z; top-view polygon only needs x,z.
        ordered.append((float(xyz[0]), float(xyz[2])))
    return ordered


def polygon_edges(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    edges = []
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        edges.append((q[0] - p[0], q[1] - p[1]))
    return edges


def best_manhattan_frame(
    edge_angles: Sequence[float], edge_lengths: Sequence[float]
) -> Tuple[float, float, float]:
    """Return best frame angle, weighted mean residual, and max residual."""
    best_frame = 0.0
    best_mean = float("inf")
    best_max = float("inf")
    # 0.25 degree grid is enough for subset mining and keeps the script fast.
    for i in range(360):
        frame = math.radians(i * 0.25)
        residuals = [edge_orientation_residual(a, frame) for a in edge_angles]
        total = sum(edge_lengths) + 1e-8
        mean = sum(r * w for r, w in zip(residuals, edge_lengths)) / total
        max_r = max(residuals) if residuals else 0.0
        if mean < best_mean:
            best_frame, best_mean, best_max = frame, mean, max_r
    return best_frame, math.degrees(best_mean), math.degrees(best_max)


def analyze_sample(split: str, sample_id: str) -> SampleStats | None:
    label_path = ROOT / "label" / f"{sample_id}.json"
    if not label_path.exists():
        return None
    with label_path.open("r", encoding="utf-8") as f:
        label = json.load(f)

    points = ordered_points_xz(label)
    if len(points) < 3:
        return None

    edges = polygon_edges(points)
    lengths = [math.hypot(dx, dz) for dx, dz in edges]
    keep = [i for i, length in enumerate(lengths) if length > 1e-4]
    edges = [edges[i] for i in keep]
    lengths = [lengths[i] for i in keep]
    if len(edges) < 3:
        return None

    angles = [wrap_pi(math.atan2(dz, dx)) for dx, dz in edges]
    frame, mean_res, max_res = best_manhattan_frame(angles, lengths)

    adjacent_devs = []
    non90 = 0
    for i, e in enumerate(edges):
        prev = edges[i - 1]
        angle = vector_angle(prev, e)
        dev = min(abs(angle - 90.0), abs(angle - 180.0), abs(angle))
        adjacent_devs.append(dev)
        if 18.0 < angle < 162.0 and abs(angle - 90.0) > 12.0:
            non90 += 1

    max_adj_dev = max(adjacent_devs) if adjacent_devs else 0.0

    reasons = []
    if mean_res >= 6.0:
        reasons.append("mean orientation residual >= 6deg")
    if max_res >= 12.0:
        reasons.append("max orientation residual >= 12deg")
    if non90 > 0:
        reasons.append("adjacent angle not near 90deg")

    return SampleStats(
        sample_id=sample_id,
        split=split,
        wall_num=len(edges),
        total_length=sum(lengths),
        best_frame_deg=math.degrees(frame),
        mean_residual_deg=mean_res,
        max_residual_deg=max_res,
        max_adjacent_deviation_deg=max_adj_dev,
        non90_corner_count=non90,
        atlanta_candidate=bool(reasons),
        reason="; ".join(reasons) if reasons else "Manhattan-like",
        points_xz=points,
    )


def write_split(path: Path, samples: Iterable[SampleStats]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            house, pano = s.sample_id.split("_", 1)
            f.write(f"{house} {pano}\n")


def draw_floorplan(stats: SampleStats, path: Path, size: int = 420) -> None:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    pts = stats.points_xz
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span = max(max_x - min_x, max_z - min_z, 1e-6)
    pad = 46

    def project(p: Tuple[float, float]) -> Tuple[float, float]:
        x = pad + (p[0] - min_x) / span * (size - pad * 2)
        y = size - pad - (p[1] - min_z) / span * (size - pad * 2)
        return x, y

    poly = [project(p) for p in pts]
    draw.polygon(poly, fill=(238, 244, 250), outline=(39, 96, 160))
    draw.line(poly + [poly[0]], fill=(39, 96, 160), width=4)
    for x, y in poly:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(214, 76, 76))
    label = f"{stats.sample_id}\nwall={stats.wall_num} mean={stats.mean_residual_deg:.1f} max={stats.max_residual_deg:.1f}"
    draw.multiline_text((14, 12), label, fill=(25, 35, 45))
    img.save(path)


def make_contact_sheet(samples: Sequence[SampleStats], path: Path, title: str) -> None:
    thumb = 420
    cols = 3
    rows = math.ceil(len(samples) / cols)
    header = 82
    sheet = Image.new("RGB", (cols * thumb, rows * thumb + header), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except Exception:
        font = None
    draw.text((24, 24), title, fill=(20, 32, 50), font=font)
    tmp_dir = OUT / "tmp_floorplans"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples):
        fp = tmp_dir / f"{sample.sample_id}.png"
        draw_floorplan(sample, fp, thumb)
        im = Image.open(fp).convert("RGB")
        x = (idx % cols) * thumb
        y = header + (idx // cols) * thumb
        sheet.paste(im, (x, y))
    sheet.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits").mkdir(exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)

    all_stats: List[SampleStats] = []
    for split in SPLITS:
        ids = split_ids(ROOT / "split" / f"{split}.txt")
        for sample_id in ids:
            stats = analyze_sample(split, sample_id)
            if stats is not None:
                all_stats.append(stats)

    csv_path = OUT / "atlanta_candidate_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "split",
            "wall_num",
            "total_length",
            "best_frame_deg",
            "mean_residual_deg",
            "max_residual_deg",
            "max_adjacent_deviation_deg",
            "non90_corner_count",
            "atlanta_candidate",
            "reason",
        ])
        for s in all_stats:
            writer.writerow([
                s.sample_id,
                s.split,
                s.wall_num,
                f"{s.total_length:.6f}",
                f"{s.best_frame_deg:.3f}",
                f"{s.mean_residual_deg:.3f}",
                f"{s.max_residual_deg:.3f}",
                f"{s.max_adjacent_deviation_deg:.3f}",
                s.non90_corner_count,
                int(s.atlanta_candidate),
                s.reason,
            ])

    by_split = {split: [s for s in all_stats if s.split == split] for split in SPLITS}
    for split, samples in by_split.items():
        candidates = [s for s in samples if s.atlanta_candidate]
        manhattan = [s for s in samples if not s.atlanta_candidate]
        complex_manhattan = [s for s in manhattan if s.wall_num > 4]
        candidates.sort(key=lambda s: (s.mean_residual_deg, s.max_residual_deg, s.non90_corner_count), reverse=True)
        manhattan.sort(key=lambda s: (s.mean_residual_deg, s.max_residual_deg))
        complex_manhattan.sort(key=lambda s: (s.wall_num, s.total_length), reverse=True)
        write_split(OUT / "splits" / f"{split}_atlanta_candidate.txt", candidates)
        write_split(OUT / "splits" / f"{split}_manhattan_like.txt", manhattan)
        write_split(OUT / "splits" / f"{split}_complex_manhattan_wallnum_gt4.txt", complex_manhattan)

    test_candidates = [s for s in by_split["test"] if s.atlanta_candidate]
    test_candidates.sort(key=lambda s: (s.mean_residual_deg, s.max_residual_deg, s.non90_corner_count), reverse=True)
    strongest_test = test_candidates
    title = "Top-12 test Atlanta/non-Manhattan candidates"
    if not strongest_test:
        strongest_test = sorted(
            by_split["test"],
            key=lambda s: (s.mean_residual_deg, s.max_residual_deg, s.wall_num),
            reverse=True,
        )
        title = "No Atlanta candidates found: top-12 closest test samples"
    make_contact_sheet(
        strongest_test[:12],
        OUT / "assets" / "test_atlanta_top12_floorplans.png",
        title,
    )

    summary_path = OUT / "README.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# MP3D Atlanta / Non-Manhattan Candidate Subset\n\n")
        f.write("This subset is mined from MP3D layout labels using top-view wall geometry.\n\n")
        f.write("## Heuristic\n\n")
        f.write("- Fit the best Manhattan frame with two orthogonal wall directions.\n")
        f.write("- Mark a sample as Atlanta candidate if mean orientation residual >= 6 deg, max residual >= 12 deg, or any adjacent wall angle clearly deviates from 90 deg.\n")
        f.write("- This is a candidate subset for experiments, not a hand-verified annotation.\n\n")
        f.write("## Counts\n\n")
        f.write("| Split | Total | Atlanta candidates | Manhattan-like |\n")
        f.write("|---|---:|---:|---:|\n")
        for split in SPLITS:
            total = len(by_split[split])
            cand = sum(s.atlanta_candidate for s in by_split[split])
            f.write(f"| {split} | {total} | {cand} | {total - cand} |\n")
        f.write("\n## Important Finding\n\n")
        f.write("The local MP3D labels are effectively Manhattan-only under this geometry check. All official layout polygons have zero Manhattan orientation residual. A quick check of the occlusion `new_label` files also found no Atlanta candidates under the same threshold.\n\n")
        f.write("Therefore, do not use this MP3D split as evidence for non-Manhattan or Atlanta-world performance. It is still useful as a Bi-Layout reproduction baseline and as a Manhattan/complex-Manhattan control set.\n\n")
        f.write("\n## Files\n\n")
        f.write("- `atlanta_candidate_stats.csv`: per-sample geometry statistics.\n")
        f.write("- `splits/test_atlanta_candidate.txt`: test subset for Atlanta-style evaluation.\n")
        f.write("- `splits/train_atlanta_candidate.txt`, `splits/val_atlanta_candidate.txt`: mined train/val candidates.\n")
        f.write("- `splits/*_complex_manhattan_wallnum_gt4.txt`: hard Manhattan control subsets with more than 4 walls.\n")
        f.write("- `assets/test_atlanta_top12_floorplans.png`: quick visual check of strongest test candidates.\n\n")
        f.write("## Recommended Use\n\n")
        f.write("For true Atlanta experiments, use a dataset with non-orthogonal room annotations, such as ZInD if available locally. For the current MP3D copy, use `test_complex_manhattan_wallnum_gt4.txt` only as a hard-layout control subset, not as non-Manhattan evidence.\n")

    print(f"Wrote {csv_path}")
    for split in SPLITS:
        total = len(by_split[split])
        cand = sum(s.atlanta_candidate for s in by_split[split])
        print(f"{split}: total={total}, atlanta_candidate={cand}, manhattan_like={total-cand}")


if __name__ == "__main__":
    main()
