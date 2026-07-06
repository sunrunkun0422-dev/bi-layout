# Bi-Layout 文档总索引

这是一页式入口。详细内容集中在下面几份中文文档里，避免再拆出多份重复说明。

## 入口文档

1. [系统流程分析.md](系统流程分析.md)
   - 看完整流程：从全景图输入到布局输出
   - 适合先读，先建立整体认知

2. [系统架构详解.md](系统架构详解.md)
   - 看内部结构：数据维度、双头、损失、后处理
   - 适合需要理解实现细节时阅读

3. [快速参考.md](快速参考.md)
   - 看接口：类、函数、配置、命令行参数
   - 适合查参数和定位函数

4. [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)
   - 看可编辑 Mermaid 网络图
   - 适合一起设计跨场景估计模块

5. [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)
   - 看跨场景共享开口融合的完整网络架构、模块输入输出和训练/推理流程
   - 适合确认“extended 能过去就融合估计”的需求边界

6. [跨场景共享开口融合实验汇报PPT.md](跨场景共享开口融合实验汇报PPT.md)
   - 看 ZInD 相邻全景小实验的 STAR 汇报版 PPT
   - 适合项目汇报、组会展示和后续改成正式 `.pptx`

7. [几何置信度选择项目模板与面试分析.md](几何置信度选择项目模板与面试分析.md)
   - 看几何一致性置信度选择项目模板
   - 适合准备简历、实验计划和面试答辩

8. [README_几何置信度选择.md](README_几何置信度选择.md)
   - 看几何置信度选择项目 README
   - 适合作为项目首页、复现实验入口和简历项目说明

## 读法建议

- 先读 [系统流程分析.md](系统流程分析.md) 的整体流程，再看 [系统架构详解.md](系统架构详解.md) 的模块细节，最后用 [快速参考.md](快速参考.md) 查接口。
- 如果你只想改一块代码，先在 [快速参考.md](快速参考.md) 找位置，再回到前两个文档看上下文。

## 按问题找文档

- 想知道系统怎么从输入走到输出: 看 [系统流程分析.md](系统流程分析.md)
- 想知道每一层张量怎么变化: 看 [系统架构详解.md](系统架构详解.md)
- 想知道函数参数和返回值: 看 [快速参考.md](快速参考.md)
- 想知道训练、评估、推理的区别: 先看 [系统流程分析.md](系统流程分析.md)，再看 [系统架构详解.md](系统架构详解.md)
- 想直接改网络架构图或画跨场景估计模块: 看 [Bi_Layout_网络架构图_可编辑.md](Bi_Layout_网络架构图_可编辑.md)
- 想确认跨场景共享开口融合的模块输入输出: 看 [跨场景共享开口融合网络架构_可编辑.md](跨场景共享开口融合网络架构_可编辑.md)
- 想汇报 ZInD 跨场景小实验: 看 [跨场景共享开口融合实验汇报PPT.md](跨场景共享开口融合实验汇报PPT.md)
- 想包装几何置信度选择项目、填实验数据和准备面试: 看 [几何置信度选择项目模板与面试分析.md](几何置信度选择项目模板与面试分析.md)
- 想快速复现实验或展示项目主页: 看 [README_几何置信度选择.md](README_几何置信度选择.md)

## 关键代码位置

- [main.py](main.py): 入口、训练、评估、保存
- [dataset/communal/base_dataset.py](dataset/communal/base_dataset.py): 数据处理
- [models/bi_layout.py](models/bi_layout.py): 模型结构
- [loss/](loss): 各类损失函数
- [postprocessing/post_process.py](postprocessing/post_process.py): 后处理
- [evaluation/accuracy.py](evaluation/accuracy.py): 评估指标
- [tools/export_geometry_selector_dataset.py](tools/export_geometry_selector_dataset.py): 导出正式几何 selector 数据集
- [tools/train_geometry_selector.py](tools/train_geometry_selector.py): 训练和评估几何置信度 selector
- [tools/estimate_cross_scene_layout.py](tools/estimate_cross_scene_layout.py): 自动估计双房间跨场景共享开口候选
- [tools/join_room_layouts.py](tools/join_room_layouts.py): 使用共享开口/接口合并两个房间布局

## 保持原则

- 这里只做导航，不重复正文内容。
- 具体解释、图示、公式和代码示例都放在对应正文文档里。
