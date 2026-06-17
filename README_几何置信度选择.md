# 基于几何一致性与置信度估计的全景 3D 房间布局估计改进

这是在 Bi-Layout 复现基础上扩展的轻量级可靠性改进实验。项目目标不是重写一个更大的布局网络，而是利用 Bi-Layout 已有的 `original / new` 双分支输出，学习一个几何感知的置信度选择器，在推理阶段自动选择更可靠的布局结果。

一句话概括：

```text
从两个候选 layout 的 depth、ratio 和 floorplan 几何形态中提取特征，训练 Geometry-aware Confidence Selector，
让模型在不改动主干网络的前提下减少分支误选，并提升最终 full_3d / full_2d 指标。
```

## 项目背景

Bi-Layout 通过双分支预测缓解了全景房间布局标注中的 enclosed / extended 歧义，但实际部署时仍然需要一个最终可用的布局结果。固定选择某个分支会丢掉另一分支在部分样本上的优势，尤其在复杂房间、边界跳变和 extended 过度扩张样本上，容易出现分支误选。

本项目把这个问题单独建模为：

```text
输入:
  P_original = {depth, ratio}
  P_new      = {new_depth, ratio}

输出:
  selected_branch in {original, new}
```

训练阶段使用 per-sample `full_3d / full_2d` 构造 oracle label；推理阶段只使用预测结果本身的几何特征，不使用 GT。

## 当前实现

核心脚本：

| 文件 | 作用 |
|---|---|
| `tools/export_geometry_selector_dataset.py` | 运行 Bi-Layout checkpoint，导出两分支指标、oracle label 和几何特征 |
| `tools/train_geometry_selector.py` | 训练 / 评估 Logistic Regression 或 RandomForest selector |
| `几何置信度选择项目模板与面试分析.md` | 完整项目分析、面试答辩材料和实验解释 |
| `geometry_selector_formal/` | 正式实验输出目录，包含 CSV、metrics、predictions、feature importance |

已实现的几何特征包括：

- 深度序列统计：均值、方差、分位数、最大值、最小值
- 边界平滑度：一阶 / 二阶 cyclic depth difference、边界跳变数量
- 多边形合法性：面积、周长、convex hull area ratio、bbox、centroid radius
- 分支一致性：两分支 depth 差异、半径差异、面积 / 周长比例、polygon IoU
- 置信度决策：基于 `prob_new` 和 `decision-threshold` 控制是否切换分支

## 实验设置

数据与模型：

```text
Dataset: MatterportLayout test split
Samples: 458 panoramas
Checkpoint: checkpoints/Bi_Layout_Net/mp3d/model_2026-05-27-21-39-39_best_0.8243_382.pkl
Config: src/config/mp3d_test_o0.yaml
Evaluation: 5-fold Stratified CV
Feature count: 148
```

说明：

```text
当前结果是第一版 prototype。
为了快速验证 selector 信号，目前在 test split 上做 5-fold CV。
更严格的论文级实验应导出 train / val / test predictions，在 train/val 上训练和调参，只在 test 上最终评估。
```

## 复现实验

推荐使用已有的 `bi_layout` conda 环境：

```bash
PY=/home/feixia/anaconda3/envs/bi_layout/bin/python
```

导出正式 selector 数据集：

```bash
$PY tools/export_geometry_selector_dataset.py \
  --cfg src/config/mp3d_test_o0.yaml \
  --mode test \
  --ckpt-option best \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --output geometry_selector_formal/test_selector_dataset.csv
```

训练 full_3d selector：

```bash
$PY tools/train_geometry_selector.py \
  --csv geometry_selector_formal/test_selector_dataset.csv \
  --label label_full_3d \
  --metric full_3d \
  --model logreg \
  --folds 5 \
  --decision-threshold 0.45 \
  --output-dir geometry_selector_formal/test_logreg_t045
```

训练 full_2d 对照 selector：

```bash
$PY tools/train_geometry_selector.py \
  --csv geometry_selector_formal/test_selector_dataset.csv \
  --label label_full_2d \
  --metric full_2d \
  --model logreg \
  --folds 5 \
  --decision-threshold 0.5 \
  --output-dir geometry_selector_formal/test_logreg_full2d_t050
```

快速 smoke test：

```bash
$PY tools/export_geometry_selector_dataset.py \
  --cfg src/config/mp3d_test_o0.yaml \
  --mode test \
  --ckpt-option best \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --limit 2 \
  --output geometry_selector_formal/smoke_test_selector_dataset.csv
```

## 当前结果

正式 depth / ratio selector 结果：

| 方法 | full_3d | selector acc | low-IoU rate | 说明 |
|---|---:|---:|---:|---|
| Original Head | 0.7952 | 48.03% | 6.11% | 固定 original |
| New/Extended Head | 0.7953 | 51.97% | 5.46% | 固定 new，最佳固定分支 |
| LogReg Geometry Selector | 0.7996 | 55.24% | 5.46% | 第一版正式 selector |
| Oracle Selector | 0.8101 | 100% | 3.71% | GT 事后选择上界 |

full_2d 对照结果：

| 方法 | full_2d | selector acc | low-IoU rate | 说明 |
|---|---:|---:|---:|---|
| Best Fixed Head | 0.8204 | 48.47% | 5.24% | 固定 original |
| LogReg Geometry Selector | 0.8264 | 57.86% | 4.37% | 几何选择器 |
| Oracle Selector | 0.8359 | 100% | 2.62% | 上界 |

关键结论：

```text
full_3d: 0.7953 -> 0.7996, +0.0043
full_3d oracle gap captured: 29.07%
full_2d: 0.8204 -> 0.8264, +0.0061
full_2d oracle gap captured: 39.10%
full_2d low-IoU rate: 5.24% -> 4.37%
```

这说明只从预测布局本身提取的几何特征已经包含分支选择信号。第一版效果还没有接近 oracle，但已经证明这个方向不是空想。

## 特征解释

LogReg 的高权重特征主要集中在：

```text
branch_ratio_depth_grad1_abs_mean
branch_diff_poly_centroid_radius
branch_absdiff_radius_std
new_depth_grad1_abs_p50
branch_diff_depth_grad1_abs_max
branch_depth_absdiff_max
branch_depth_corr
```

这些特征对应的直觉是：

- 分支间边界梯度差异越大，越可能存在角点跳变或边界不稳定。
- 平面中心半径和最大半径变化过大，可能对应 extended 过度扩张。
- 两分支 depth 相关性低，说明两个候选 layout 对空间结构判断分歧较大。
- new 分支自身梯度异常，可能意味着扩展边界不连续。

## 项目价值

这个项目的工程价值在于：

- 不改 Bi-Layout 主干，训练成本低。
- 只在推理后增加轻量 selector，部署风险小。
- 指标不仅看平均 IoU，还关注 selector accuracy、oracle gap、low-IoU rate。
- 对室内建模、AR 测量和机器人空间理解更友好，因为这些任务更怕长尾几何异常。

## 当前限制

- 当前训练评估是 test split 内 5-fold CV，还不是严格 train/val/test 实验。
- invalid layout rate 还没有完整实现，目前主要用 low-IoU rate 和 polygon validity proxy 表示异常。
- 角点置信度没有显式 head，第一版用 depth 梯度作为 corner confidence proxy。
- repair / fallback 还没有接入最终指标，因此 full_3d low-IoU rate 暂时没有明显下降。
- 还需要补 ZInD 或复杂房间子集验证泛化。

## 下一步

1. 导出 train / val / test 三份 prediction dataset，完成严格测试集评估。
2. 实现 `repair_or_fallback.py`：非法 polygon fallback、过度扩张过滤、边界突变平滑。
3. 加入角点置信度或 corner heatmap proxy，增强对边界跳变样本的识别。
4. 做特征消融：polygon、boundary smoothness、branch difference、all features。
5. 补充失败案例可视化，展示 selector 纠正了哪些分支误选。
6. 在 ZInD 或复杂房间子集上做泛化验证。

## 简历表述

可以写成：

```text
复现 Bi-Layout 全景 3D 房间布局估计，并针对双分支输出缺少最终置信选择的问题，
实现 Geometry-aware Confidence Selector。
在 MatterportLayout 458 张测试全景图上，使用 depth / ratio / polygon 合法性 / 边界平滑度 / 分支一致性等 148 维几何特征训练轻量 LogReg selector，
将 full_3d 从固定分支的 0.7953 提升到 0.7996，捕获 29.07% oracle gap；
同时将 full_2d 从 0.8204 提升到 0.8264，并将 low-IoU rate 从 5.24% 降到 4.37%。
```

