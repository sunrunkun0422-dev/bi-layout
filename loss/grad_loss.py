"""
@Date: 2021/08/12
@description: 梯度匹配损失函数 - 约束边界的平滑性和方向

【STEP 4.1.3: Gradient Loss - 梯度监督】
==========================================
为什么需要梯度损失:
1. 仅用L1Loss约束深度值,不能保证边界光滑
   例: 深度序列[1,3,1]vs[1,2,1],绝对误差相同,但平滑性不同
2. 梯度损失显式约束相邻点的变化率
3. 避免出现"锯齿状"的不规则边界

【原理】:
- 一阶导数(梯度) g(i) = d(i+1) - d(i-1)
- 梯度表示"斜率",平滑的边界梯度应该平缓变化
- 两种约束:
  1. 梯度方向一致 → 余弦相似度
  2. 梯度大小相似 → L1Loss

【实现细节】:
使用1D卷积计算梯度:
- kernel = [1, 0, -1] 表示 d(i+1) - d(i-1)
- padding='circular' 处理周期边界(全景图首尾相接)
"""

import torch
import torch.nn as nn
import numpy as np

from visualization.grad import get_all


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # L1Loss: 计算绝对误差
        self.loss = nn.L1Loss()

        # 余弦相似度: 计算向量方向的相似性
        # dim=-1: 沿着最后一维计算
        # eps=0: 避免除以0
        self.cos = nn.CosineSimilarity(dim=-1, eps=0)

        # ===== STEP 4.1.3.1: 梯度计算卷积核 =====
        # 创建一维卷积层用于计算梯度
        self.grad_conv = nn.Conv1d(
            in_channels=1,          # 输入1个通道(深度图)
            out_channels=1,         # 输出1个通道(梯度)
            kernel_size=3,          # 使用3个点的卷积
            stride=1,               # 步长为1
            padding=0,              # 不自动padding,需要手动处理
            bias=False,             # 不使用偏置项
            padding_mode='circular' # 【关键】循环padding
        )

        # ===== STEP 4.1.3.2: 设置卷积权重 =====
        # 手动设置权重为[1, 0, -1]
        # 计算中心差分: g(i) = (d(i+1) - d(i-1)) / 2
        # 省略/2的原因: 常数因子不影响损失大小关系
        self.grad_conv.weight = nn.Parameter(torch.tensor([[[1, 0, -1]]]).float())

        # 【关键】禁止梯度更新
        # 这个卷积核是固定的"差分算子",不需要学习
        self.grad_conv.weight.requires_grad = False

    def forward(self, gt, dt, second_type=False):
        """
        计算梯度匹配损失

        参数:
        - gt: ground truth字典
        - dt: 预测字典
        - second_type: bool
          - False: 使用原始深度(origin branch)
          - True: 使用新深度(new branch,遮挡感知)

        输出: [normal_loss, grad_loss]
        - normal_loss: 梯度方向的余弦相似度损失
        - grad_loss: 梯度大小的L1损失
        """
        # ===== STEP 4.1.3.3: 选择深度图 =====
        if second_type is True:
            # 使用new_depth(遮挡感知分支)
            gt_direction, _, gt_angle_grad = get_all(gt['new_depth'], self.grad_conv)
            dt_direction, _, dt_angle_grad = get_all(dt['new_depth'], self.grad_conv)
        else:
            # 使用原始depth(origin分支)
            gt_direction, _, gt_angle_grad = get_all(gt['depth'], self.grad_conv)
            dt_direction, _, dt_angle_grad = get_all(dt['depth'], self.grad_conv)

        # ===== STEP 4.1.3.4: 梯度方向损失 =====
        # get_all()返回的是:
        # - gt_direction: 标注梯度的方向向量(标准化)
        # - gt_angle_grad: 标注梯度的幅度

        # 计算余弦相似度: cos_sim = <u, v> / (|u||v|)
        # 范围: [-1, 1]
        # 1 = 完全同向, -1 = 完全反向, 0 = 正交
        cos_sim = self.cos(gt_direction, dt_direction)

        # 转换为损失: loss = 1 - cos_sim
        # 这样完全同向时loss=0, 完全反向时loss=2
        normal_loss = (1 - cos_sim).mean()

        # 【含义】:
        # 即使深度值相同,如果梯度方向不同,也会被惩罚
        # 保证边界的"变化方向"一致,形成平滑曲线

        # ===== STEP 4.1.3.5: 梯度幅度损失 =====
        # 比较梯度的绝对值大小
        # 为什么需要:
        # - 仅约束方向还不够,还要保证变化程度一致
        # - 例如: [1,2,3]和[1,100,200]方向相同但幅度不同
        # - 梯度幅度损失会发现这个差异
        grad_loss = self.loss(gt_angle_grad, dt_angle_grad)

        # 【含义】:
        # 确保预测的边界"弯曲程度"与标注一致
        # 避免"过度光滑"或"过度曲折"

        # ===== 返回两个损失 =====
        # 调用者需要组合这两个损失
        # 通常: total_grad_loss = normal_loss + grad_loss
        # 或使用不同的权重: w1*normal_loss + w2*grad_loss
        return [normal_loss, grad_loss]


if __name__ == '__main__':
    """测试代码 - 验证梯度损失计算"""
    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz
    from visualization.boundary import draw_boundaries
    from visualization.floorplan import draw_floorplan

    def show_boundary(image, depth, ratio):
        """辅助函数: 可视化边界"""
        boundary_list = depth2boundaries(ratio, depth, step=None)
        draw_boundaries(image.transpose(1, 2, 0), boundary_list=boundary_list, show=True)
        draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=True, center_color=0.8)

    # ===== 加载数据集 =====
    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train', patch_num=256)
    gt = mp3d_dataset.__getitem__(1)

    # ===== 转换为张量 =====
    gt['depth'] = torch.from_numpy(gt['depth'][np.newaxis])  # batch size is 1

    # ===== 创建伪预测(完全一致) =====
    dummy_dt = {
        'depth': gt['depth'].clone(),  # 完全相同的深度
    }

    # ===== 测试1: 完全一致的预测,loss应该=0 =====
    grad_loss = GradLoss()
    loss = grad_loss(gt, dummy_dt)
    print(f"Loss with identical prediction: {loss}")  # [~0, ~0]

    # ===== 测试2: 添加噪声,loss应该>0 =====
    # dummy_dt['depth'][..., 20] *= 3  # 某个位置乘以3
    # loss = grad_loss(gt, dummy_dt)
    # print(f"Loss with noisy prediction: {loss}")  # [>0, >0]

    # ===== 可视化对比 =====
    # show_boundary(gt['image'], gt['depth'][0].numpy(), gt['ratio'])
    # show_boundary(gt['image'], dummy_dt['depth'][0].numpy(), gt['ratio'])
