import torch

# 创建一个形状为(7, 3)的张量（7行3列）
x = torch.arange(21).reshape(7, 3)
print("原始张量x:\n", x)
print("原始张量形状:", x.shape, "\n")


# --------------------
# torch.chunk 示例
# --------------------
# 在dim=0（行方向）拆分为3块
chunk_size = 2
chunks = torch.chunk(x, chunks=chunk_size, dim=0)
print(f"============= torch.chunk 拆分为 {chunk_size} 块的结果：==============")
for i, chunk in enumerate(chunks):
    print(f"第{i+1}块形状: {chunk.shape}, 内容:\n{chunk}\n")


# --------------------
# torch.split 示例
# --------------------
# 情况1：按固定大小拆分（每块2行）
splits_fixed = torch.split(x, split_size_or_sections=2, dim=0)
print("===========torch.split 按固定大小 2 拆分的结果：===========")
for i, split in enumerate(splits_fixed):
    print(f"第{i+1}块形状: {split.shape}, 内容:\n{split}\n")

# 情况2：按自定义大小列表拆分（[3, 2, 2]行）
splits_custom = torch.split(x, split_size_or_sections=[3, 2, 2], dim=0)
print("==============torch.split 按自定义大小[3,2,2]拆分的结果：=============")
for i, split in enumerate(splits_custom):
    print(f"第{i+1}块形状: {split.shape}, 内容:\n{split}\n")
    