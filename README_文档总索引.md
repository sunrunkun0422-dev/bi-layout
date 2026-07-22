> 修改记录：2026-07-13 16:32 CST - 跨场景入口由旧版更新为统一工程管线。
> 修改记录：2026-07-16 16:45 CST - 增加 ZInD 匹配数据格式与转换入口。
> 修改记录：2026-07-17 11:48 CST - 增加 ZInD-BiPair-v1 数据集生成、加载和全量验证入口。
> 修改记录：2026-07-17 13:43 CST - 增加 ZInD-BiPair-v1 开口召回评估、环形指标与阈值标定入口。

# Bi-Layout 文档总索引

这是一页式入口。当前项目主线收敛到 Bi-Layout 基础模型与跨场景共享开口融合网络架构。

## 入口文档

1. [README.md](README.md)
   - 看原始 Bi-Layout 项目说明、环境安装、数据准备和训练/测试入口

2. [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)
   - 看可编辑 Mermaid 网络图
   - 适合理解单图 Bi-Layout 双分支结构

3. [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)
   - 看跨场景共享开口融合的完整网络架构、模块输入输出和训练/推理流程
   - 适合确认开口响应、共享开口匹配、相对位姿估计和联合布局融合的研究主线

4. [ZInD-BiPair-v1数据集说明.md](ZInD-BiPair-v1数据集说明.md)
   - 看首版 partial-opening 数据集的筛选、opening 匹配、NPZ 标签、规模和验证结果
   - 这是当前推荐用于训练 Opening Head、Matcher、Pose 和联合布局的 pair dataset

5. [ZInD匹配数据格式.md](ZInD匹配数据格式.md)
   - 看原始 ZInD 如何转换为正负全景对、开口区间、对应关系和相对位姿监督
   - 这是较早的跨 complete-room door/opening 诊断格式

## 读法建议

- 先读 [README.md](README.md) 和 [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)，建立原始单图布局估计基础。
- 再读 [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)，理解本文创新点如何从单图 enclosed/extended 输出扩展到跨场景共享开口融合。

## 按问题找文档

- 想知道原始 Bi-Layout 如何训练、测试和组织数据: 看 [README.md](README.md)
- 想直接改 Bi-Layout 网络架构图: 看 [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)
- 想确认跨场景共享开口融合的模块输入输出: 看 [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)
- 想生成首版 partial-opening 训练数据: 看 [ZInD-BiPair-v1数据集说明.md](ZInD-BiPair-v1数据集说明.md) 和 [tools/build_zind_bipair_v1.py](tools/build_zind_bipair_v1.py)
- 想测试开口召回率并在 val 标定阈值: 看 [tools/evaluate_zind_opening_recall.py](tools/evaluate_zind_opening_recall.py)、[evaluation/opening_recall.py](evaluation/opening_recall.py) 和 [src/output/progress_report_20260716/进度汇报结果.md](src/output/progress_report_20260716/进度汇报结果.md)
- 想生成跨 complete-room 的 door/opening 诊断清单: 看 [ZInD匹配数据格式.md](ZInD匹配数据格式.md) 和 [tools/build_zind_matching_dataset.py](tools/build_zind_matching_dataset.py)
- 想运行跨场景工程版: 看 [tools/estimate_cross_scene_layout.py](tools/estimate_cross_scene_layout.py)、[utils/cross_scene_pipeline.py](utils/cross_scene_pipeline.py) 和 [tools/join_room_layouts.py](tools/join_room_layouts.py)

## 关键代码位置

- [main.py](main.py): 入口、训练、评估、保存
- [dataset/communal/base_dataset.py](dataset/communal/base_dataset.py): 数据处理
- [models/bi_layout.py](models/bi_layout.py): 模型结构
- [loss/](loss): 各类损失函数
- [postprocessing/post_process.py](postprocessing/post_process.py): 后处理
- [evaluation/accuracy.py](evaluation/accuracy.py): 评估指标
- [tools/estimate_cross_scene_layout.py](tools/estimate_cross_scene_layout.py): 自动估计双房间跨场景共享开口候选
- [utils/cross_scene_pipeline.py](utils/cross_scene_pipeline.py): 统一验证、候选融合、NMS、选择和版本化输出
- [models/cross_scene_matcher.py](models/cross_scene_matcher.py): 开口响应与双向 cross attention 匹配
- [dataset/zind_bipair_builder.py](dataset/zind_bipair_builder.py): 构建严格 partial-opening 正负样本与全部监督标签
- [dataset/zind_bipair_dataset.py](dataset/zind_bipair_dataset.py): 加载 ZInD-BiPair-v1 JSONL/NPZ 并组成训练 batch
- [tools/build_zind_bipair_v1.py](tools/build_zind_bipair_v1.py): 在 ZInD 同目录生成 ZInD-BiPair-v1
- [tools/validate_zind_bipair_v1.py](tools/validate_zind_bipair_v1.py): 全量检查缓存、标签、路径和房屋级数据泄漏
- [evaluation/opening_recall.py](evaluation/opening_recall.py): 唯一 panorama 的 token AP/召回、环形连通域和区间 IoU 指标
- [tools/evaluate_zind_opening_recall.py](tools/evaluate_zind_opening_recall.py): 评估 GT/预测双深度开口先验并在 val 标定阈值
- [dataset/zind_pair_mining.py](dataset/zind_pair_mining.py): 从 ZInD 拓扑与重复门洞标注中挖掘正负全景对
- [dataset/panorama_pair_dataset.py](dataset/panorama_pair_dataset.py): 展开并批处理匹配监督
- [tools/build_zind_matching_dataset.py](tools/build_zind_matching_dataset.py): 生成 train/val/test 匹配 manifest
- [tools/join_room_layouts.py](tools/join_room_layouts.py): 使用共享开口/接口合并两个房间布局

## 保持原则

- 这里只做导航，不重复正文内容。
- 具体解释、图示、公式和代码示例都放在对应正文文档里。
