import torch
import torch.nn as nn
from typing import Union

_shape_t = Union[int, list[int], torch.Size]
class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(*self.normalized_shape)
            )  # unpack normalized_shape
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
            print(f"weight.shape: {self.weight.shape}")
            print(f"bias.shape: {self.bias.shape}")
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 使用ones_和zeros_原地初始化参数
            nn.init.zeros(self.bias)

    def forward(self, x: torch.Tensor):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        mean = x.mean(dims, keepdim=True)
        var = x.var(
            dim=dims, unbiased=False, keepdim=True
        )  # 使用有偏估计，即除以n，而不是n-1，因为更关注数据本身分布，而不是对总体方差的无偏估计

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm


# 测试函数 by DeepSeek
def test_layer_norm_1D():
    """测试自定义LayerNorm与PyTorch官方实现一致性"""
    # 创建测试数据
    batch_size, seq_len, hidden_size = 2, 10, 128
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 创建自定义和官方LayerNorm
    custom_ln = LayerNorm(hidden_size)
    torch_ln = nn.LayerNorm(hidden_size)

    # 同步参数
    with torch.no_grad():
        # 将自定义LayerNorm的gamma和beta原地复制到 torch的LayerNorm中
        torch_ln.weight.copy_(custom_ln.weight)
        torch_ln.bias.copy_(custom_ln.bias)

    # 前向传播
    custom_out = custom_ln(x)

    print(f"custom_out.shape: {custom_out.shape}")
    torch_out = torch_ln(x)
    print(f"torch_out.shape: {torch_out.shape}")

    # 比较结果
    diff = torch.abs(custom_out - torch_out).max().item()
    print(f"最大差异: {diff:.8f}")

    # 验证数值属性
    custom_mean = custom_out.mean().item()
    custom_var = custom_out.var(unbiased=False).item()
    print(f"自定义实现 - 均值: {custom_mean:.6f}, 方差: {custom_var:.6f}")

    torch_mean = torch_out.mean().item()
    torch_var = torch_out.var(unbiased=False).item()
    print(f"官方实现 - 均值: {torch_mean:.6f}, 方差: {torch_var:.6f}")


def test_layer_norm_2D():
    """测试自定义LayerNorm与PyTorch官方实现一致性"""
    # 创建测试数据
    batch_size, seq_len, hidden_size = 2, 10, 128
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 创建自定义和官方LayerNorm
    normalized_shape = (seq_len, hidden_size)
    custom_ln = LayerNorm(normalized_shape)
    torch_ln = nn.LayerNorm(normalized_shape)

    # 同步参数
    with torch.no_grad():
        # 将自定义LayerNorm的gamma和beta原地复制到 torch的LayerNorm中
        torch_ln.weight.copy_(custom_ln.weight)
        torch_ln.bias.copy_(custom_ln.bias)

    # 前向传播
    custom_out = custom_ln(x)

    print(f"custom_out.shape: {custom_out.shape}")
    torch_out = torch_ln(x)
    print(f"torch_out.shape: {torch_out.shape}")

    # 比较结果
    diff = torch.abs(custom_out - torch_out).max().item()
    print(f"最大差异: {diff:.8f}")

    # 验证数值属性
    custom_mean = custom_out.mean().item()
    custom_var = custom_out.var(unbiased=False).item()
    print(f"自定义实现 - 均值: {custom_mean:.6f}, 方差: {custom_var:.6f}")

    torch_mean = torch_out.mean().item()
    torch_var = torch_out.var(unbiased=False).item()
    print(f"官方实现 - 均值: {torch_mean:.6f}, 方差: {torch_var:.6f}")


if __name__ == "__main__":
    print("=" * 20)
    test_layer_norm_1D()
    print("=" * 20)
    test_layer_norm_2D()
