---
marp: true
theme: default
paginate: true
---

# 跨场景共享开口融合实验汇报

基于 Bi-Layout extended 分支的双全景共享开口估计

---

# 1. Situation

全景房间布局估计通常只处理单个房间。

但在 ZInD 等真实室内数据中，相邻全景图之间往往共享同一个可通行开口，例如门洞、开放通道或宽开口。

Bi-Layout 的 enclosed / extended 双分支为跨房间融合提供了一个重要线索：如果 extended 分支能越过某一墙段继续向外延伸，该区域可能是可通行接口。

---

# 2. Task

目标是在 Bi-Layout 单图布局输出基础上，构建一个轻量级跨场景融合 MVP。

核心任务：

1. 从 `depth/new_depth` 中提取可通行开口候选。
2. 对两张相邻全景图的开口候选进行配对。
3. 根据共享开口估计两个房间的相对位姿。
4. 输出统一坐标系下的联合布局和候选诊断指标。

---

# 3. Action

当前实现采用 opening-first 流程：

```text
Bi-Layout 单图推理
  -> D_ext - D_enc 提取可通行开口候选
  -> O_A x O_B 开口候选配对
  -> 共享开口端点对齐估计 T_BA
  -> 几何一致性排序
  -> 输出 joint layout
```

实现文件：

```text
utils/cross_scene_estimator.py
tools/estimate_cross_scene_layout.py
utils/joint_layout.py
```

---

# 4. Key Modules

| 模块 | 当前状态 |
| --- | --- |
| Bi-Layout enclosed / extended 输出 | 已完成 |
| Extended opening proposal | MVP 已完成 |
| Opening pair candidate ranking | MVP 已完成 |
| Relative pose estimation | 已完成 |
| Geometry consistency selector | MVP 已完成 |
| Opening-guided cross attention | 待实现 |

---

# 5. Experiment Setup

数据集：ZInD 相邻全景图。

样本：

```text
home_id = 0006
floor = floor_01
A = partial_room_02 / pano_56
B = partial_room_11 / pano_54
covis_score = 1.0
```

输入：

```text
src/input/zind_adjacent_0006/A_floor01_pr02_pano56.jpg
src/input/zind_adjacent_0006/B_floor01_pr11_pano54.jpg
```

---

# 6. Result

当前 MVP 输出：

```text
best candidate: A wall 1 <-> B wall 1
confidence: 1.0000
geometryScore: 0.1455
passabilityReward: 0.5830
shared opening width: 1.9848
roomBScale: 0.8614
endpointMapping: reversed
```

结果文件：

```text
src/output/zind_adjacent_0006_cross_scene/
  zind0006_pr02_pr11_candidates.json
  zind0006_pr02_pr11_best_joint.json
  zind0006_pr02_pr11_best_joint.svg
  zind0006_pr02_pr11_best_joint.png
```

---

# 7. Analysis

当前实验说明：

1. `D_ext - D_enc` 能够提供可通行开口的弱监督线索。
2. 仅靠几何搜索已经可以得到可视化合理的双房间拼接结果。
3. passability reward 可以减少盲目枚举墙段中心带来的错误候选。
4. 仍然缺少跨图特征确认，因此复杂遮挡、相似墙段和多开口场景中可能误匹配。

---

# 8. Next Step

下一步重点实现 opening-guided cross attention：

```text
O_A/O_B opening candidates
  -> opening token pooling
  -> masked bidirectional cross attention
  -> Aff_AB token matching matrix
  -> feature score + geometry score fusion
```

预期提升：

1. 提高共享开口 top-1 accuracy。
2. 降低相似墙段误匹配。
3. 支持多开口、多房间和长尾空间结构。

---

# 9. Summary

这个版本完成了跨场景共享开口融合的 MVP 闭环。

当前贡献可以概括为：

```text
利用 Bi-Layout extended 分支产生可通行开口先验，
再通过几何一致性估计相邻全景图之间的共享开口和相对位姿。
```

后续如果加入 opening-guided cross attention，就能从工程规则版推进到可训练网络版。
