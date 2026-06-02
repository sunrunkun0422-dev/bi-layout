# MP3D Atlanta / Non-Manhattan Candidate Subset

This subset is mined from MP3D layout labels using top-view wall geometry.

## Heuristic

- Fit the best Manhattan frame with two orthogonal wall directions.
- Mark a sample as Atlanta candidate if mean orientation residual >= 6 deg, max residual >= 12 deg, or any adjacent wall angle clearly deviates from 90 deg.
- This is a candidate subset for experiments, not a hand-verified annotation.

## Counts

| Split | Total | Atlanta candidates | Manhattan-like |
|---|---:|---:|---:|
| train | 1647 | 0 | 1647 |
| val | 190 | 0 | 190 |
| test | 458 | 0 | 458 |

## Important Finding

The local MP3D labels are effectively Manhattan-only under this geometry check. All official layout polygons have zero Manhattan orientation residual. A quick check of the occlusion `new_label` files also found no Atlanta candidates under the same threshold.

Therefore, do not use this MP3D split as evidence for non-Manhattan or Atlanta-world performance. It is still useful as a Bi-Layout reproduction baseline and as a Manhattan/complex-Manhattan control set.


## Files

- `atlanta_candidate_stats.csv`: per-sample geometry statistics.
- `splits/test_atlanta_candidate.txt`: test subset for Atlanta-style evaluation.
- `splits/train_atlanta_candidate.txt`, `splits/val_atlanta_candidate.txt`: mined train/val candidates.
- `splits/*_complex_manhattan_wallnum_gt4.txt`: hard Manhattan control subsets with more than 4 walls.
- `assets/test_atlanta_top12_floorplans.png`: quick visual check of strongest test candidates.

## Recommended Use

For true Atlanta experiments, use a dataset with non-orthogonal room annotations, such as ZInD if available locally. For the current MP3D copy, use `test_complex_manhattan_wallnum_gt4.txt` only as a hard-layout control subset, not as non-Manhattan evidence.
