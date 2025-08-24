import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

torch.set_printoptions(precision=10)

_shape_t = Union[int, list[int], torch.Size]


class RMSNorm1D(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) 实现
    公式: RMSNorm(x) = {x / sqrt[mean(x^2) + eps]} * g

    参数:
        num_features: 特征维度
        eps: 数值稳定性常数，防止除以零
    """

    def __init__(self, num_features: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor):

        if self.num_features != x.shape[-1]:
            raise ValueError(
                f"norm dim: {self.num_features} must be the same with the last dim of input x:{x.shape[-1]}"
            )

        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        x_norm = x / rms

        if self.affine:
            x_norm = x_norm * self.weight
        return x_norm


# TODO RMSNorm2D


# 测试函数
def test_rms_norm():
    """测试 RMSNorm 实现"""

    def test_rms_norm_(RMS_type: str = None):
        supported_type = ["RMSNorm1D"]
        if RMS_type not in supported_type:
            raise ValueError(
                f"RMS_type: {RMS_type} not supported, should in {supported_type}"
            )

        # Step-1: prepare inputs and BN moudles
        print("=" * 50)
        print(f"测试 RMS_type = {RMS_type}")
        print("=" * 50)
        try:
            custom_rms_class = getattr(__import__(__name__), RMS_type)
        except AttributeError:
            raise ValueError(f"找不到自定义 RMSNorm 类: {RMS_type}")

        # 设置随机种子以确保可重复性
        torch.manual_seed(42)

        # 测试 1D 输入 (例如 Transformer 中的特征)
        if RMS_type == "RMSNorm1D":
            batch_size, seq_len, dim = 2, 5, 10
            x = torch.randn(batch_size, seq_len, dim)
        else:
            batch_size, channels, height, width = 2, 3, 32, 32
            x = torch.randn(batch_size, channels, height, width)

        # 创建 RMSNorm
        rms_norm = custom_rms_class(dim)
        # 前向传播
        output = rms_norm(x)

        # 验证输出形状
        assert (
            output.shape == x.shape
        ), f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"

        # 验证 RMS 归一化属性
        dims = -1 if RMS_type == "RMSNorm1D" else (2, 3)
        # 计算输出的 RMS 值
        output_rms = torch.sqrt(torch.mean(output**2, dim=-1))
        # 由于我们应用了缩放，RMS 值应该接近权重的绝对值
        expected_rms = torch.abs(rms_norm.weight).mean().item()
        actual_rms = output_rms.mean().item()

        print(f"预期 RMS: {expected_rms:.10f}, 实际 RMS: {actual_rms:.10f}")
        print(f"RMS 差异: {abs(expected_rms - actual_rms):.10f}")

    torch.set_printoptions(precision=10)
    test_rms_norm_("RMSNorm1D")
    # test_rms_norm_('RMSNorm2D')


if __name__ == "__main__":
    test_rms_norm()
