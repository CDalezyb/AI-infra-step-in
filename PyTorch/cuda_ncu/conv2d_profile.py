# 步骤 1：编写简单的卷积测试脚本（test_conv.py）
import torch
import torch.nn as nn

# 定义卷积层（模拟目标场景）
conv = nn.Conv2d(2048, 2048, 3, padding=1).cuda()
x = torch.randn(8, 2048, 7, 7).cuda()

# 预热（消除初始化开销）
for _ in range(10):
    conv(x)
torch.cuda.synchronize()

# 执行卷积（供nsight分析）
for _ in range(100):
    conv(x)
torch.cuda.synchronize()

# 步骤 2：用nsight compute运行并分析
# ncu -o conv_profile --target-processes all python conv2d_profile.py