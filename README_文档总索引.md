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

5. [几何置信度选择项目模板与面试分析.md](几何置信度选择项目模板与面试分析.md)
   - 看几何一致性置信度选择项目模板
   - 适合准备简历、实验计划和面试答辩

6. [README_几何置信度选择.md](README_几何置信度选择.md)
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

## 保持原则

- 这里只做导航，不重复正文内容。
- 具体解释、图示、公式和代码示例都放在三份正文文档里。
