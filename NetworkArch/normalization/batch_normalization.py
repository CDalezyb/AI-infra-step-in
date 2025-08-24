import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=10)


# FIXME(@CDalezyb 20250824) the running_var is wrong compared to torch.running_var
class BatchNorm1D(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        Batch Normalization for 1D inputs (B, C) or (B, C, L)

        参数:
            num_features: 特征通道数 C
            eps: 数值稳定性常数
            momentum: 运行统计量的动量
            affine: 是否添加可学习的缩放和偏移
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 可学习参数
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # register_buffer 是用来持久化不需要梯度下降的参数
        # requires_grad = False + 出现在state_dict()中 + 不在 parameter()中
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        """
        Args:
            x: (B, C) or (B, C, L)
        """
        if x.dim() not in [2, 3]:
            raise ValueError(f"期望2D或3D输入 (得到 {x.dim()}D)")

        # 训练/推理模式判断
        if self.training:
            # 计算当前批次的统计量
            dims = (0,) if x.dim() == 2 else (0, 2)
            mean = x.mean(dim=dims, keepdim=False)
            var = x.var(dim=dims, unbiased=False, keepdim=False)

            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var
        else:
            # 推理模式使用运行统计量
            mean = self.running_mean
            var = self.running_var

        # 此处, mean 和 var 的shape 均为 [self.num_features]
        # 使用延迟扩展 + 广播技术, 在计算的时候拓展维度以匹配输入
        if x.dim() == 2:
            # x:(B,C), mean:(C,), mean[None, :]:(1,C)
            # 最终广播到(B,C)的shape
            x_norm = (x - mean[None, :]) / torch.sqrt(var[None, :] + self.eps)
        else:  # dim=3
            # x:(B,C,L), mean:(C,), mean[None, :, None]:(1,C,1)
            # 最终广播到(B,C,L)的shape
            x_norm = (x - mean[None, :, None]) / torch.sqrt(
                var[None, :, None] + self.eps
            )

        # 应用仿射变换
        if self.affine:
            if x.dim() == 2:
                x_norm = self.weight[None, :] * x_norm + self.bias[None, :]
            else:
                x_norm = self.weight[None, :, None] * x_norm + self.bias[None, :, None]

        return x_norm

    def extra_repr(self):
        return f"num_features:{self.num_features},\n eps={self.eps}, \nmomentum={self.momentum}, \naffine={self.affine}, \nrunning_mean={self.running_mean}, \nrunning_var={self.running_var}"


class BatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Batch Normalization for 2D inputs (B, C, H, W)

        参数:
            num_features: 特征通道数 C
            eps: 数值稳定性常数
            momentum: 运行统计量的动量
            affine: 是否添加可学习的缩放和偏移
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 可学习参数
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # 运行统计量
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        # 输入形状: (B, C, H, W)
        if x.dim() != 4:
            raise ValueError(f"期望4D输入 (得到 {x.dim()}D)")

        # 训练/推理模式判断
        if self.training:
            # 计算当前批次的统计量 (在B, H, W维度)
            mean = x.mean(dim=(0, 2, 3), keepdim=False)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=False)

            # 更新运行统计量
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var
        else:
            # 推理模式使用运行统计量
            mean = self.running_mean
            var = self.running_var

        # 归一化
        # 扩展维度以匹配输入 (B, C, H, W)
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(
            var[None, :, None, None] + self.eps
        )

        # 应用仿射变换
        if self.affine:
            x_norm = (
                self.weight[None, :, None, None] * x_norm
                + self.bias[None, :, None, None]
            )

        return x_norm

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"


# 测试函数
def test_batch_norm():
    """测试自定义BatchNorm与PyTorch官方实现一致性"""

    def test_batch_norm_(BN_type: str = None, dimNum=None):
        supported_type = ["BatchNorm1D", "BatchNorm2D"]
        if BN_type not in supported_type:
            raise ValueError(
                f"BN_type: {BN_type} not supported, should in {supported_type}"
            )

        if BN_type == "BatchNorm2D" and dimNum != 4:
            raise ValueError(f"input dim for BatchNorm2D should be 4, gets {dimNum}")

        if BN_type == "BatchNorm1D" and dimNum not in [2, 3]:
            raise ValueError(
                f"input dim for BatchNorm1D should be either {[2,3]}, gets {dimNum}"
            )

        # Step-1: prepare inputs and BN moudles
        print("=" * 50)
        print(f"测试 BN_type = {BN_type}, dimNum = {dimNum}")
        print("=" * 50)
        try:
            custom_bn_class = getattr(__import__(__name__), BN_type)
        except AttributeError:
            raise ValueError(f"找不到自定义BatchNorm类: {BN_type}")

        pytorch_norm_type = BN_type.replace("D", "d")
        try:
            torch_bn_class = getattr(nn, pytorch_norm_type)
        except AttributeError:
            raise ValueError(f"找不到PyTorch BatchNorm类: {pytorch_norm_type}")

        batch_size, num_features = 2, 3
        x = None
        # 创建测试数据 (B, C, L) 或者
        if BN_type == "BatchNorm1D":
            if dimNum == 2:
                x = torch.randn(batch_size, num_features)
            else:
                x = torch.randn(batch_size, num_features, 10)
        else:
            x = torch.randn(batch_size, num_features, 10, 5)

        # 创建自定义和官方BatchNorm
        custom_bn = custom_bn_class(num_features)
        torch_bn = torch_bn_class(num_features, momentum=0.1)

        # 同步参数
        with torch.no_grad():
            torch_bn.weight.copy_(custom_bn.weight)
            torch_bn.bias.copy_(custom_bn.bias)
            torch_bn.running_mean.copy_(custom_bn.running_mean)
            torch_bn.running_var.copy_(custom_bn.running_var)

        # Step-2: switch into train mode and update running_mean and running_var
        # 训练模式
        custom_bn.train()
        torch_bn.train()

        # 前向传播
        custom_out = custom_bn(x)
        print(f"custom_bn.running_mean:{custom_bn.running_mean}")
        print(f"custom_bn.running_var:{custom_bn.running_var}")
        print(f"torch_bn.running_mean:{torch_bn.running_mean}")
        print(f"torch_bn.running_var:{torch_bn.running_var}")
        print(custom_bn.extra_repr())
        torch_out = torch_bn(x)
        print(f"after exec torch_out = torch_bn(x)")
        print(f"custom_bn.running_mean:{custom_bn.running_mean}")
        print(f"custom_bn.running_var:{custom_bn.running_var}")
        print(f"torch_bn.running_mean:{torch_bn.running_mean}")
        print(f"torch_bn.running_var:{torch_bn.running_var}")

        mean_diff = (
            torch.abs(custom_bn.running_mean - torch_bn.running_mean).max().item()
        )
        var_diff = torch.abs(custom_bn.running_var - torch_bn.running_var).max().item()
        print(f"mean_diff差异: {mean_diff:.8f}")
        print(f"var_diff: {var_diff:.8f}")

        # 比较结果
        diff = torch.abs(custom_out - torch_out).max().item()
        print(f"训练模式最大差异: {diff:.8f}")

        # 推理模式
        custom_bn.eval()
        torch_bn.eval()
        custom_out = custom_bn(x)
        torch_out = torch_bn(x)
        diff = torch.abs(custom_out - torch_out).max().item()
        print(f"推理模式最大差异: {diff:.8f}")

    test_batch_norm_("BatchNorm1D", 2)
    test_batch_norm_("BatchNorm1D", 3)
    test_batch_norm_("BatchNorm2D", 4)


if __name__ == "__main__":
    test_batch_norm()
