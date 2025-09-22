import torch

# 测试squeeze操作
a = torch.randn(1, 2, 1, 3, 1)
print(f"原始张量a: {a.shape}")

b = a.squeeze()
print(f"a.squeeze() 移除所有维度为1的维度: {b.shape}")

c = a.squeeze(0)
print(f"a.squeeze(0) 尝试移除第0维: {c.shape}")

d = a.squeeze(1)
print(f"a.squeeze(1) 尝试移除第1维: {d.shape}")

e = a.squeeze(2)
print(f"a.squeeze(2) 尝试移除第2维: {e.shape}")

# 测试unsqueeze操作
a = torch.randn(2, 3)
print(f"\n原始张量a: {a.shape}")

b = a.unsqueeze(0)
print(f"a.unsqueeze(0) 在第0维增加维度: {b.shape}")

c = a.unsqueeze(1)
print(f"a.unsqueeze(1) 在第1维增加维度: {c.shape}")

d = a.unsqueeze(2)
print(f"a.unsqueeze(2) 在第2维增加维度: {d.shape}")