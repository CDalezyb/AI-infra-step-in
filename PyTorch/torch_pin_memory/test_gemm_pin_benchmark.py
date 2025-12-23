import contextlib
import os
import torch
import shutil

DEVICE_TYPE = None
DEVICE_MODULE = None
ProfilerActivity_DEVICE = None
Stream = None
current_stream = None
trace_rootdir = None

if hasattr(torch, "musa") and torch.musa.is_available():
    DEVICE_TYPE = "musa"
    DEVICE_MODULE = torch.musa
    from torch.profiler import ProfilerActivity

    ProfilerActivity_DEVICE = ProfilerActivity.MUSA
    import torch_musa
    from torch_musa import Stream, current_device, current_stream

    torch.backends.mudnn.allow_tf32 = True
elif hasattr(torch, "cuda") and torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
    DEVICE_MODULE = torch.cuda
    from torch.profiler import ProfilerActivity

    ProfilerActivity_DEVICE = ProfilerActivity.CUDA
    from torch.cuda import Stream, current_device, current_stream
else:
    raise RuntimeError("No supported device (CUDA/MUSA) found!")

DEVICE = f"{DEVICE_TYPE}:{current_device()}"
s = Stream()
torch.manual_seed(42)
size = 1024 # 1024
t1_cpu_pinned = torch.randn(size, size, size, pin_memory=True)
t2_cpu_paged = torch.randn(size, size, size, pin_memory=False)
t3_gpu = torch.randn(2048, 2048, 2048, device=DEVICE)

t_gpu_global = None


def inner(pinned: bool, streamed: bool, non_blocking: bool = True) -> None:
    global t1_gpu_global, t2_gpu_global, t3_gpu
    stream_context = DEVICE_MODULE.stream(s) if streamed else contextlib.nullcontext()

    # Step1: gemm
    t3_gpu_mul = t3_gpu @ t3_gpu 
    # t3_gpu_mm_event  = current_stream().record_event()

    # Step2: H2D, and independent of Step1
    with stream_context:
        if pinned:
            t_gpu_global = t1_cpu_pinned.to(DEVICE, non_blocking=non_blocking)
        else:
            t_gpu_global = t2_cpu_paged.to(DEVICE, non_blocking=non_blocking)

        # current_exec_stream = s if streamed else current_stream()
        # t_star_h2d_event = current_exec_stream.record_event()
    # Step3: gemm dependent on step2's H2D
    useless_action = t_gpu_global @ t_gpu_global @ t_gpu_global

    DEVICE_MODULE.synchronize()


def benchmark_with_profiler(
    pinned: bool, streamed: bool, non_blocking: bool = True
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, ProfilerActivity_DEVICE]

    skip_first, wait, warmup, active = 1, 1, 1, 1
    num_steps = skip_first + wait + warmup + active
    schedule = torch.profiler.schedule(
        skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=1
    )

    trace_dir = os.path.join(
        trace_rootdir,
        f"{DEVICE_TYPE}_streamed-{int(streamed)}_pinned-{int(pinned)}_non_blocking-{int(non_blocking)}",
    )
    os.makedirs(trace_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(num_steps + 5):
            inner(streamed=streamed, pinned=pinned, non_blocking=non_blocking)
            prof.step()


def benchmark_stream_True():
    global trace_rootdir
    trace_rootdir = "./traces/gemm_pin_traces_stream_True"
    if os.path.exists(trace_rootdir):
        shutil.rmtree(trace_rootdir)
    os.makedirs(trace_rootdir, exist_ok=True)

    print(f"Running on device: {DEVICE_TYPE.upper()}")
    streamed = True

    benchmark_with_profiler(streamed=streamed, pinned=False, non_blocking=True)

    benchmark_with_profiler(streamed=streamed, pinned=True, non_blocking=True)

    benchmark_with_profiler(streamed=streamed, pinned=False, non_blocking=False)

    benchmark_with_profiler(streamed=streamed, pinned=True, non_blocking=False)

    print("Profiling completed! Trace files saved to 'memory_pin_traces/'")

def benchmark_stream_False():
    global trace_rootdir
    trace_rootdir = "./traces/gemm_pin_traces_stream_False"
    if os.path.exists(trace_rootdir):
        shutil.rmtree(trace_rootdir)
    os.makedirs(trace_rootdir, exist_ok=True)

    print(f"Running on device: {DEVICE_TYPE.upper()}")
    streamed = False

    benchmark_with_profiler(streamed=streamed, pinned=False, non_blocking=True)

    benchmark_with_profiler(streamed=streamed, pinned=True, non_blocking=True)

    benchmark_with_profiler(streamed=streamed, pinned=False, non_blocking=False)

    benchmark_with_profiler(streamed=streamed, pinned=True, non_blocking=False)

    print("Profiling completed! Trace files saved to 'memory_pin_traces/'")

if __name__ == "__main__":
    
    # benchmark_stream_True()
    
    benchmark_stream_False()