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

## 读法建议

- 先读 [README.md](README.md) 和 [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)，建立原始单图布局估计基础。
- 再读 [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)，理解本文创新点如何从单图 enclosed/extended 输出扩展到跨场景共享开口融合。

## 按问题找文档

- 想知道原始 Bi-Layout 如何训练、测试和组织数据: 看 [README.md](README.md)
- 想直接改 Bi-Layout 网络架构图: 看 [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)
- 想确认跨场景共享开口融合的模块输入输出: 看 [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)
- 想运行当前跨场景 MVP: 看 [tools/estimate_cross_scene_layout.py](tools/estimate_cross_scene_layout.py) 和 [tools/join_room_layouts.py](tools/join_room_layouts.py)

## 关键代码位置

- [main.py](main.py): 入口、训练、评估、保存
- [dataset/communal/base_dataset.py](dataset/communal/base_dataset.py): 数据处理
- [models/bi_layout.py](models/bi_layout.py): 模型结构
- [loss/](loss): 各类损失函数
- [postprocessing/post_process.py](postprocessing/post_process.py): 后处理
- [evaluation/accuracy.py](evaluation/accuracy.py): 评估指标
- [tools/estimate_cross_scene_layout.py](tools/estimate_cross_scene_layout.py): 自动估计双房间跨场景共享开口候选
- [tools/join_room_layouts.py](tools/join_room_layouts.py): 使用共享开口/接口合并两个房间布局

## 保持原则

- 这里只做导航，不重复正文内容。
- 具体解释、图示、公式和代码示例都放在对应正文文档里。
