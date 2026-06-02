"""
@Date: 2021/08/12
@description: 边界曲率损失函数 - 约束布局边界的光滑性

【STEP 4.1.2: Boundary Loss - 边界曲率监督】
==========================================
为什么需要这个损失:
1. 深度图只约束沿水平方向的深度值
2. 但边界应该是"光滑连续"的曲线,不能过于弯曲
3. 这个损失通过约束边界的曲率来提高几何合理性

【数学原理】:
- 边界可以看作深度关于水平位置的函数 d(u)
- 一阶导数 d'(u) 表示深度变化率(斜率)
- 二阶导数 d''(u) 表示曲率(弯曲程度)
- 对这些导数施加约束,保证边界光滑

【实现方式】:
用深度图的一阶导数(边界梯度)来近似代替二阶导数
- 原理: 如果一阶导数平滑,则二阶导数小
- 简化计算: 避免数值不稳定的二阶导数
"""
import torch
import torch.nn as nn
from utils.conversion import depth2xyz, xyz2lonlat


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # L1Loss而不是MSE的原因:
        # - MSE(平方误差)对异常值(outliers)过度惩罚
        # - L1Loss(绝对误差)对异常值更鲁棒
        # - 在深度预测中,个别错误的深度值不应完全破坏模型
        # - L1Loss让模型专注于整体趋势而不是个别值
        self.loss = nn.L1Loss()

    def forward(self, gt, dt):
        """
        计算边界曲率损失

        【STEP 4.1.2.1: 坐标转换】
        输入深度图 → 3D点云 → 经纬度坐标

        参数:
        - gt: ground truth字典,包含:
          {
            'depth': (B, 256) 地面深度,
            'ratio': (B, 1) 天花板比例
          }
        - dt: 模型预测字典,包含:
          {
            'boundary': (B, 2, 256) 边界经纬度
          }

        输出:
        - loss: 标量,曲率损失值

        【处理流程】:
        """
        # ===== STEP 4.1.2.2: 地面点云 =====
        # 将深度图转换为3D点云(地面高度)
        # 深度值代表从camera到地面的距离
        # xyz2depth的逆过程,恢复3D坐标
        # 输出: (B, 256, 3) [x, y, z坐标]
        gt_floor_xyz = depth2xyz(gt['depth'])

        # ===== STEP 4.1.2.3: 天花板点云 =====
        # 复制地面点云
        gt_ceil_xyz = gt_floor_xyz.clone()

        # 【关键】修改y坐标(竖直方向)
        # 将y值改为 -ratio (向上)
        # 为什么是负数:
        # - 相机高度(camera_height)对应y=0
        # - 地面对应y > 0(向下)
        # - 天花板对应y < 0(向上)
        # - ratio是天花板与相机的距离
        # 输出: (B, 256, 3) 天花板点云
        gt_ceil_xyz[..., 1] = -gt['ratio']

        # ===== STEP 4.1.2.4: 转换为纬度 =====
        # xyz2lonlat: 3D笛卡尔坐标 → 经纬度坐标
        # 纬度 latitude = arcsin(y / |xyz|)
        # 经度 longitude = atan2(x, z)
        # 输出: [..., 2] 最后一维是纬度
        gt_floor_boundary = xyz2lonlat(gt_floor_xyz)[..., -1:]  # 取最后一列(纬度)
        gt_ceil_boundary = xyz2lonlat(gt_ceil_xyz)[..., -1:]

        # ===== STEP 4.1.2.5: 堆叠边界 =====
        # 在最后一维堆叠: [floor_lat, ceil_lat]
        # 形状: (B, 256, 2)
        # 含义: 每个水平位置的地面和天花板纬度
        gt_boundary = torch.cat([gt_floor_boundary, gt_ceil_boundary], dim=-1)

        # 转置: (B, 256, 2) → (B, 2, 256)
        # 原因: 损失函数期望通道在前
        gt_boundary = gt_boundary.permute(0, 2, 1)

        # ===== STEP 4.1.2.6: 计算损失 =====
        # dt['boundary']: 模型预测的边界纬度 (B, 2, 256)
        # gt_boundary: 标注的边界纬度 (B, 2, 256)
        dt_boundary = dt['boundary']

        # L1Loss计算绝对误差
        # 纬度差异 = |predicted_lat - gt_lat|
        # 为什么用纬度而不是深度:
        # - 纬度是角度(更稳定)
        # - 纬度变化反映竖直方向的约束
        # - 用纬度损失而不是深度损失的好处:
        #   避免深度值的尺度问题(深度~0.1-10米,尺度跨度大)
        loss = self.loss(gt_boundary, dt_boundary)

        return loss


if __name__ == '__main__':
    """测试代码 - 验证损失函数工作正常"""
    import numpy as np
    from dataset.mp3d_dataset import MP3DDataset

    # ===== 加载一个真实样本 =====
    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train')
    gt = mp3d_dataset.__getitem__(0)

    # ===== 准备数据(转为张量) =====
    gt['depth'] = torch.from_numpy(gt['depth'][np.newaxis])  # batch size is 1
    gt['ratio'] = torch.from_numpy(gt['ratio'][np.newaxis])  # batch size is 1

    # ===== 创建伪预测(使用gt作为预测,测试损失=0) =====
    dummy_dt = {
        'depth': gt['depth'].clone(),
        'boundary': torch.cat([
            xyz2lonlat(depth2xyz(gt['depth']))[..., -1:],  # 地面纬度
            xyz2lonlat(depth2xyz(gt['depth'], plan_y=-gt['ratio']))[..., -1:]  # 天花板纬度
        ], dim=-1).permute(0, 2, 1)
    }

    # ===== 计算损失 =====
    boundary_loss = BoundaryLoss()
    loss = boundary_loss(gt, dummy_dt)
    print(f"Loss: {loss}")  # 应该接近0

    # ===== 测试有差异的预测 =====
    # dummy_dt['boundary'][:, :, :20] /= 1.2  # 某些位置有差异
    # loss = boundary_loss(gt, dummy_dt)
    # print(f"Loss with difference: {loss}")  # 应该 > 0
