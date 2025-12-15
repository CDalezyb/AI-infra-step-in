import os
import time
import torch
from os import path as osp
from pathlib import Path
from datetime import datetime
import torch.profiler as profiler
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
from torch.profiler import profile, record_function, ProfilerActivity

warnings.filterwarnings("ignore")

MODEL_PATH = os.environ["VLLM_QWEN_MODEL_PATH"]
VLLM_QWEN_TRACE_SAVING_DIR = os.environ["VLLM_QWEN_TRACE_SAVING_DIR"]
trace_dir = Path(VLLM_QWEN_TRACE_SAVING_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S")
trace_dir.mkdir(parents=True, exist_ok=True)

PROMPT = """<|im_start|>user
请介绍一下人工智能的发展历程<|im_end|>
<|im_start|>assistant
"""

GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    eos_token_id=151643,
    pad_token_id=151643,
    do_sample=True,
)

def load_qwen_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def infer_with_profiler():
    # 加载模型和预处理输入
    tokenizer, model = load_qwen_model()
    print(f"✅ 模型加载完成，设备：{model.device if hasattr(model, 'device') else '多卡分布式'}")
    
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # 1. Warmup阶段（在profiler上下文之前执行，推理结果不使用）
    print("🚀 开始Warmup推理（结果不使用）...")
    with torch.no_grad():
        for _ in range(3):  # 执行3轮warmup推理，排除初始化噪声
            warmup_outputs = model.generate(**inputs, generation_config=GENERATION_CONFIG)
            torch.cuda.synchronize()  # 确保warmup完成
    print("✅ Warmup完成")
    
    start_time = time.time()
    profiler_outputs = None
    # 2. Profiler上下文仅采集一次推理的trace（移除on_trace_ready避免自动保存）
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        profile_memory=False,
        record_shapes=False,
        with_stack=True,
    ) as prof:
        with record_function("qwen_model_inference"):
            # 仅执行一次推理，Profiler采集该次完整过程
            with torch.no_grad():
                profiler_outputs = model.generate(
                    **inputs,
                    generation_config=GENERATION_CONFIG
                )
            print(f"zyb debug: Profiler finished after 1 iteration.")
    
    # 3. 手动导出所有trace（避免重复保存）
    torch.profiler.tensorboard_trace_handler(trace_dir)(prof)
    
    # Profiler退出后执行后续操作
    end_time = time.time()
    print(f'!!!!!!!!!!! infer per batch_data cost time : {end_time - start_time} s, iters : 1')
    
    # 4. 输出推理结果（Profiler退出后）
    response = tokenizer.decode(profiler_outputs[0], skip_special_tokens=True)
    print("\n===================== 推理结果 =====================")
    print(response)
    
    print("\n===================== Trace 摘要 =====================")
    print(f"Trace 保存路径：{trace_dir}")
    print(f"显存占用峰值：{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    infer_with_profiler()