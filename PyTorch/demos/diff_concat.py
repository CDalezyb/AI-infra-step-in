import torch

# 创建两个形状为(2,3)的测试张量
tensor1 = torch.tensor([[1, 2, 3], 
                        [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], 
                        [10, 11, 12]])


print("原始张量1形状:", tensor1.shape)
print("原始张量1:\n", tensor1)
print("\n原始张量2形状:", tensor2.shape)
print("原始张量2:\n", tensor2)

# 使用torch.cat在第0维（行方向）连接
cat_result_0 = torch.cat([tensor1, tensor2], dim=0)
print("\ntorch.cat在第0维连接的结果形状:", cat_result_0.shape)
print("torch.cat在第0维连接的结果:\n", cat_result_0)

# 使用torch.concat在第0维（行方向）连接
concat_result_0 = torch.concat([tensor1, tensor2], dim=0)
print("\ntorch.concat在第0维连接的结果形状:", concat_result_0.shape)
print("torch.concat在第0维连接的结果:\n", concat_result_0)

# 验证结果是否相同
print("\n第0维连接结果是否相同:", torch.allclose(cat_result_0, concat_result_0))

# 使用torch.cat在第1维（列方向）连接
cat_result_1 = torch.cat([tensor1, tensor2], dim=1)
print("\ntorch.cat在第1维连接的结果形状:", cat_result_1.shape)
print("torch.cat在第1维连接的结果:\n", cat_result_1)

# 使用torch.concat在第1维（列方向）连接
concat_result_1 = torch.concat([tensor1, tensor2], dim=1)
print("\ntorch.concat在第1维连接的结果形状:", concat_result_1.shape)
print("torch.concat在第1维连接的结果:\n", concat_result_1)

# 验证结果是否相同
print("\n第1维连接结果是否相同:", torch.allclose(cat_result_1, concat_result_1))

# 检查两个函数是否指向同一个对象
print("\ntorch.cat和torch.concat是否为同一函数:", torch.cat is torch.concat)
    