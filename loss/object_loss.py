"""
@Date: 2021/08/12
@description: 对象/门窗检测损失函数 - 学习布局中的关键点

【STEP 4.1.5: Object/Opening Loss - 关键点监督】
==========================================
为什么需要对象/门窗检测:
1. Corners是布局中最关键的几何特征
2. 直接约束corner位置,提高定位精度
3. 特别是对于遮挡感知分支(new_head)
4. 明确标记"哪些位置是corner",帮助模型学习空间位置

【应用场景】:
- Corner Detection: 使用热力图标记corners位置
- Opening Detection: 标记门窗位置(对应于遮挡)
- Object Detection: 可拓展到检测其他物体

【关键概念】:
class imbalance (类别不平衡):
- 背景像素数 >>> corner像素数
- 简单使用交叉熵会导致模型倾向预测"背景"
- 需要使用Focal Loss重新加权
"""

import torch
import torch.nn as nn
from loss.grad_loss import GradLoss


class ObjectLoss(nn.Module):
    """对象检测损失(目前实现为TODO)"""
    def __init__(self):
        super().__init__()
        # 热力图损失: 用于corner检测
        self.heat_map_loss = HeatmapLoss(reduction='mean')  # 也可用FocalLoss替换

        # L1损失: 用于目标位置回归
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, gt, dt):
        """
        计算对象检测损失

        参数:
        - gt: ground truth字典
        - dt: 预测字典

        注意: 当前实现为占位符,返回0
        """
        # TODO: 实现完整的对象检测损失
        # 当前返回0意味着这个损失项未启用
        return 0


class HeatmapLoss(nn.Module):
    """
    热力图焦点损失(Heatmap Focal Loss)

    用于处理类别不平衡问题(大量背景 vs 少量corner)

    【焦点损失的核心思想】:
    标准交叉熵: CE = -log(p_t)
    其中 p_t = p (if y=1) 或 1-p (if y=0)

    焦点损失: FL = -(1-p_t)^α * log(p_t)

    效果:
    - 当α>0时,错误分类的样本被加权
    - 简单样本(确信度高)权重小
    - 难样本(确信度低)权重大
    - 强制模型关注容易漏掉的样本
    """

    def __init__(self, weight=None, alpha=2, beta=4, reduction='mean'):
        """
        初始化热力图损失

        参数:
        - weight: 类别权重(可选)
        - alpha: 中心点(corner)的焦点幂
          用于调整正样本的权重
          alpha越大,模型越关注难样本
        - beta: 背景点的焦点幂
          用于调整负样本的权重
          通常 beta > alpha(负样本远多于正样本)
        - reduction: 'mean'或'sum'
        """
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha  # 正样本焦点幂
        self.beta = beta    # 负样本焦点幂
        self.reduction = reduction

    def forward(self, targets, inputs):
        """
        计算热力图损失

        参数:
        - targets: (B, 256) 二值热力图,值为0或1
          1表示corner位置,0表示背景
        - inputs: (B, 256) 网络预测,概率值[0,1]

        返回:
        - loss: 标量损失值
        """
        # ===== STEP 4.1.5.1: 识别正负样本 =====
        # 正样本: targets == 1(corner位置)
        center_id = (targets == 1.0).float()  # (B, 256) 0或1

        # 负样本: targets != 1(背景)
        other_id = (targets != 1.0).float()   # (B, 256) 0或1

        # ===== STEP 4.1.5.2: 计算正样本损失 =====
        # 中心点损失(焦点损失公式):
        # L_center = -(1-p)^α * log(p)
        # 其中 p = inputs (模型预测的概率)
        #
        # (1-inputs)^alpha: 焦点项
        #   - 当p接近1时,(1-p)^α≈0,损失小(易分样本)
        #   - 当p接近0时,(1-p)^α≈1,损失大(难分样本)
        # log(inputs + 1e-14): 避免log(0)=-∞
        center_loss = -center_id * (1.0 - inputs) ** self.alpha * torch.log(inputs + 1e-14)
        # 输出: (B, 256) 只有center_id==1的位置有非零值

        # ===== STEP 4.1.5.3: 计算负样本损失 =====
        # 背景点损失(变形的焦点损失):
        # L_bg = -(1-targets)^β * (p)^α * log(1-p)
        # 其中 targets = 0(背景)
        #
        # (1-targets)^β = 1: 背景权重因子(可调)
        # (inputs)^α: 置信度焦点项
        #   - 当输出p接近1(自信错分)时权重最大
        #   - 当输出p接近0(正确预测)时权重最小
        # log(1-inputs + 1e-14): 避免log(0)
        other_loss = -other_id * (1 - targets) ** self.beta * inputs ** self.alpha * torch.log(1.0 - inputs + 1e-14)
        # 输出: (B, 256) 只有other_id==1的位置有非零值

        # ===== STEP 4.1.5.4: 聚合损失 =====
        # 总损失 = 正样本损失 + 负样本损失
        loss = center_loss + other_loss
        # 形状: (B, 256)

        # ===== STEP 4.1.5.5: 归约(Reduction) =====
        batch_size = loss.size(0)

        if self.reduction == 'mean':
            # 计算平均损失(归一化)
            # 为什么除以batch_size:
            # - 不同batch的样本数可能不同
            # - 除以batch_size保证损失值在可比范围内
            # - 避免batch_size大时损失过大
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            # 求和(通常不推荐,会因batch大小而变化)
            loss = torch.sum(loss) / batch_size

        return loss


if __name__ == '__main__':
    """测试代码 - 验证热力图损失"""
    # 创建模拟数据
    batch_size = 2
    seq_len = 256

    # ===== 创建标注 =====
    # 假设每个序列有5个corners
    targets = torch.zeros(batch_size, seq_len)
    targets[0, [10, 50, 100, 150, 200]] = 1  # 第1个样本的5个corners
    targets[1, [20, 80, 120, 180, 240]] = 1  # 第2个样本的5个corners

    # ===== 创建完美预测 =====
    inputs_perfect = targets.clone()  # 完全准确

    # ===== 创建有误的预测 =====
    inputs_noisy = targets.clone()
    inputs_noisy[0, 10] = 0.5  # 某个corner预测为0.5(不确定)
    inputs_noisy[0, 60] = 0.3  # 某个背景预测为0.3(假阳性)

    # ===== 计算损失 =====
    hm_loss = HeatmapLoss(alpha=2, beta=4)

    loss_perfect = hm_loss(targets, inputs_perfect)
    print(f"Perfect prediction loss: {loss_perfect:.6f}")  # 应该接近0

    loss_noisy = hm_loss(targets, inputs_noisy)
    print(f"Noisy prediction loss: {loss_noisy:.6f}")  # 应该 > 0

    # ===== 验证焦点损失的特性 =====
    print(f"\nFocal Loss特性验证:")
    print(f"Perfect loss < Noisy loss: {loss_perfect < loss_noisy}")  # 应该是True
