"""
@Date: 2021/08/12
@description: 高度估计损失函数 - 约束天花板高度

【STEP 4.1.4: LED Loss - 高度监督】
==========================================
为什么需要高度损失(LED):
LED = Layout Estimation Detection
1. 深度图约束的是"地面边界到相机的距离"
2. 天花板高度(ratio)是独立的参数
3. 需要专门的监督信号确保高度预测准确
4. 高度错误会导致整个布局"压扁"或"拉伸"

【关键概念】:
- depth: 地面边界深度 d_floor = |地面点 - 相机|
- ratio: 天花板相对高度 h_ceil = ratio × camera_height
- 天花板深度: d_ceil = |天花板点 - 相机|

根据勾股定理:
d_ceil² = d_floor² + (h_ceil - h_camera)²
若忽视相机高度的差异:
d_ceil ≈ d_floor × (ratio相对地面) / 地面高度

【损失设计】:
floor_depth = depth × camera_height
ceil_depth_expected = depth × camera_height × ratio
loss = |depth × camera_height - ceil_depth_expected|
    + |depth × camera_height - predicted_depth × camera_height × ratio|
"""
import torch
import torch.nn as nn


class LEDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # L1Loss: 使用绝对误差而非平方误差
        # 原因: 高度估计中的异常值应该鲁棒处理
        self.loss = nn.L1Loss()

    def forward(self, gt, dt):
        """
        计算高度估计损失

        参数:
        - gt: ground truth字典,包含:
          {
            'depth': (B, 256) 地面深度,
            'ratio': (B, 1) 天花板高度比例
          }
        - dt: 预测字典,包含:
          {
            'ceil_depth': (B, 256) 天花板深度预测,
            'depth': (B, 256) 地面深度预测,
            'ratio': (B, 1) 天花板高度预测(可选)
          }

        输出:
        - loss: 标量损失值
        """
        # ===== 常数定义 =====
        camera_height = 1.6  # 标准相机高度(单位:米)
        # 为什么是1.6:
        # - 对应成人眼睛高度(标准参考点)
        # - MP3D数据集的标准设置
        # - 与全景摄像头通常的安装高度一致

        # ===== STEP 4.1.4.1: 深度图缩放到米单位 =====
        # 原始深度图是相对值,需要乘以camera_height转为实际距离
        gt_depth = gt['depth'] * camera_height
        # 输出: (B, 256) 单位为米

        # 【含义】:
        # 地面到相机的实际距离 = 相对深度 × 相机高度
        # 例: 深度=2, camera_height=1.6m → 实际距离=3.2m

        # ===== STEP 4.1.4.2: 计算天花板深度 =====
        # 天花板到相机的距离 = 相对深度 × 相机高度 × 天花板比例
        # 为什么乘以ratio:
        # - ratio = 天花板高度 / camera_height
        # - dt['ceil_depth']: 模型预测的天花板深度(相对值)
        # - 乘以camera_height和ratio转为实际距离
        dt_ceil_depth = dt['ceil_depth'] * camera_height * gt['ratio']
        # 输出: (B, 256) 单位为米

        # 地面深度(模型预测)
        dt_floor_depth = dt['depth'] * camera_height
        # 输出: (B, 256) 单位为米

        # ===== STEP 4.1.4.3: 计算两个分支的损失 =====
        # 【损失1】: 天花板深度与地面深度的关系
        # 物理约束: ceil_depth应该小于floor_depth(天花板在地面上方)
        # 为什么这样:
        # - 如果天花板与地面平行,且天花板在上方
        # - 则天花板深度 < 地面深度
        # - 这个损失约束这个关系
        ceil_loss = self.loss(gt_depth, dt_ceil_depth)
        # 含义: 预测的天花板深度应该与地面深度成比例
        # (比例由ratio决定)

        # 【损失2】: 地面深度损失(主要监督)
        floor_loss = self.loss(gt_depth, dt_floor_depth)
        # 含义: 预测的地面深度应该与标注一致

        # ===== STEP 4.1.4.4: 融合两个损失 =====
        # 总损失 = ceil_loss + floor_loss
        # 等权重融合,两个约束同样重要
        # 为什么:
        # - ceil_loss保证天花板与地面的相对位置
        # - floor_loss保证地面深度准确
        # - 两者共同作用确保布局高度正确
        loss = floor_loss + ceil_loss

        # 【高阶理解】:
        # 地面深度d_f和天花板比例ratio确定后:
        # - 天花板深度 d_c = d_f × ratio
        # - 天花板实际高度 = camera_height × (1 - ratio)
        #
        # 这个损失的作用:
        # 1. 地面深度准确(floor_loss)
        # 2. 天花板深度与地面协调(ceil_loss)
        # → 整个房间的3D结构正确

        return loss


if __name__ == '__main__':
    """测试代码 - 验证LED损失"""
    import numpy as np
    from dataset.mp3d_dataset import MP3DDataset

    # ===== 加载真实样本 =====
    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train')
    gt = mp3d_dataset.__getitem__(0)

    # ===== 转换为张量 =====
    gt['depth'] = torch.from_numpy(gt['depth'][np.newaxis])  # batch size is 1
    gt['ratio'] = torch.from_numpy(gt['ratio'][np.newaxis])  # batch size is 1

    # ===== 创建伪预测(完全准确) =====
    dummy_dt = {
        'depth': gt['depth'].clone(),  # 地面深度完全准确
        'ceil_depth': gt['depth'] / gt['ratio']  # 天花板深度根据ratio计算
    }

    # ===== 计算损失(应该接近0) =====
    led_loss = LEDLoss()
    loss = led_loss(gt, dummy_dt)
    print(f"LED Loss (perfect prediction): {loss}")  # 应该≈0

    # ===== 测试2: 有误差的预测 =====
    # dummy_dt['depth'][..., :20] *= 3  # 某些位置有误差
    # loss = led_loss(gt, dummy_dt)
    # print(f"LED Loss (with error): {loss}")  # 应该 > 0
