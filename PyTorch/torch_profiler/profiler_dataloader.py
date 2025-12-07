import torch
import torch.profiler
from torch.utils.data import DataLoader, Dataset

# target: to proof that profiler includes the dataloader time

# 模拟自定义数据集
class MyDataset(Dataset):
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        # 模拟数据加载/预处理耗时
        import time
        time.sleep(0.1)
        return torch.randn(3, 224, 224), torch.randint(0, 10, (1,))

dataloader = DataLoader(MyDataset(), batch_size=32, num_workers=4)

# 开启Profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for idx, (data, label) in enumerate(dataloader):
        # 模拟GPU计算
        data = data.cuda()
        label = label.cuda()
        _ = data.mean()  # 随便一个GPU操作
        prof.step()  # 标记步骤，方便分析
        if idx > 10:  # 只跑10步，避免耗时过久
            break

# 打印Profiling结果（可看到DataLoader相关的CPU耗时）
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))