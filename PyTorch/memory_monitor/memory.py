'''本文件尝试使用代码解释torch的内存管理机制'''
import torch
import os
from memory_monitor.mem_utils import append_torch_memory_logging, init_mpi_logging

if __name__ == "__main__":
    # 初始化日志记录
    rank = int(os.getenv("RANK", 0))
    init_mpi_logging(rank)

    # 模拟一个简单的PyTorch操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1000, 1000, device=device)
    
    # 记录内存使用情况
    append_torch_memory_logging("Initial allocation", device=device, unit="MB")
    
    # 执行一些操作
    y = x @ x.T
    
    # 再次记录内存使用情况
    append_torch_memory_logging("After matrix multiplication", device=device, unit="MB")
    
    # 清理
    del x, y
    torch.cuda.empty_cache()
    
    # 最终记录内存使用情况
    append_torch_memory_logging("Final cleanup", device=device, unit="MB")