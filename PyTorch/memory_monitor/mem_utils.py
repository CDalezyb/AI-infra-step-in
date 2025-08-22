import logging
import os
import sys
from typing import Optional
import torch
from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated, max_memory_reserved, reset_peak_memory_stats

try:
    import torch_musa
    from torch_musa import memory_allocated, memory_reserved, max_memory_allocated, max_memory_reserved, reset_peak_memory_stats
except ModuleNotFoundError:
    torch_musa = None


def append_torch_memory_logging(stage: str, device: Optional[torch.device] = None, unit:str = "GB", MAXLEN:int = 70) -> None:
    """添加显存日志记录（单行输出四个指标）
    
    参数:
        stage: 当前记录阶段描述
        device: 目标设备，None表示默认设备
    """

    def _bytes_to_mb(bytes):
        """字节转MB"""
        return bytes / (1024 ** 2)

    def _bytes_to_gb(bytes):
        """字节转GB"""
        return bytes / (1024 ** 3)
    
    # 获取内存指标并转换为MB（保留1位小数）
    unit = unit.upper()
    if unit not in ["MB", "GB"]:
        raise ValueError("单位必须是 'MB' 或 'GB'")
    unit_converter = _bytes_to_gb if unit == "GB" else _bytes_to_mb
    mem_alloc = f"{unit_converter(memory_allocated(device)):.1f}"
    mem_reserved = f"{unit_converter(memory_reserved(device)):.1f}"
    max_mem_alloc = f"{unit_converter(max_memory_allocated(device)):.1f}"
    max_mem_reserved = f"{unit_converter(max_memory_reserved(device)):.1f}"
    
    # 构建日志消息
    device_str = f"Device: {device}" if device else "Default device"
    stage = stage[:MAXLEN].ljust(MAXLEN)  # 保证 stage 不超过 MAXLEN 个字符，不足在后补空格
    fixed_prefix = f"[{device_str}] Stage: {stage} | "

    log_msg = (
        fixed_prefix +
        f"Alloc: {mem_alloc} {unit} | "
        f"Reserved: {mem_reserved} {unit} | "
        f"Peak Alloc: {max_mem_alloc} {unit} | "
        f"Peak Reserved: {max_mem_reserved} {unit}"
    )
    
    logging.debug(log_msg)


def init_mpi_logging(rank: int) -> None:
    """初始化日志记录"""
    world_size = int(os.getenv('WORLD_SIZE', 1))
    
    # 创建日志目录
    log_dir = os.getenv('LOG_DIR', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 为每个rank创建独立日志文件
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    
    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter(
        '[%(asctime)s] [Rank %(rank)s] %(levelname)s: %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 控制台只显示rank0
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('[Rank 0] %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # 添加rank信息到日志记录
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    logging.info(f"Initialized logging for rank {rank}/{world_size}")
    logging.info(f"Logging to: {log_file}")


def test_util():
    # 示例用法
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    init_mpi_logging(rank)

    append_torch_memory_logging("初始化时", rank)
    # K = 1024 * 1024  # 1GB
    a = torch.randn(10000, 10000).to('musa') if torch_musa is not None else torch.randn(10000, 10000).to('cuda')
    append_torch_memory_logging("创建张量 a 后", rank)
    b = torch.randn(20000, 20000).to('musa') if torch_musa is not None else torch.randn(20000, 20000).to('cuda')
    append_torch_memory_logging("创建张量 b 后", rank)

    del a
    append_torch_memory_logging("删除张量 a 后", rank)
    del b
    append_torch_memory_logging("删除张量 b 后",rank)
        
    if torch_musa is not None:
        torch_musa.empty_cache()
        append_torch_memory_logging("清空MUSA缓存后",rank)
    else:
        torch.cuda.empty_cache()
        append_torch_memory_logging("清空CUDA缓存后", rank)


if __name__ == "__main__":
    test_util()