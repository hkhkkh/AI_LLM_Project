import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL

# 设置环境（镜像源、缓存路径等）
setup_environment()

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 阶段九：4bit 量化对比
# ==========================================

model_name = BASE_MODEL

print("="*50)
print("阶段九：量化技术对比 (FP16 vs 8bit vs 4bit)")
print("="*50)

# 系统信息
print(f"\nGPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ 未检测到'}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 测试prompt
test_prompt = "FutureAI公司的Wifi密码是多少？"
messages = [{"role": "user", "content": test_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def test_model(model, name):
    """测试模型显存和速度"""
    inputs = tokenizer([text], return_tensors="pt").to(device)
    mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    elapsed = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[{name}]")
    print(f"  显存: {mem:.0f} MB | 耗时: {elapsed:.2f}s")
    print(f"  回答: {response[-80:]}")
    return mem, elapsed

results = {}

# 方案A: FP16
print("\n" + "-"*50)
print("测试 FP16 (全精度)")
model_fp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
if device == "cuda":
    model_fp = model_fp.to(device)
results['FP16'] = test_model(model_fp, "FP16")
del model_fp
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 方案B: 8bit
print("\n" + "-"*50)
print("测试 8bit 量化")
config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=config_8bit, device_map="auto"
)
results['8bit'] = test_model(model_8bit, "8bit")
del model_8bit
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 方案C: 4bit
print("\n" + "-"*50)
print("测试 4bit 量化 (NF4)")
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=config_4bit, device_map="auto"
)
results['4bit'] = test_model(model_4bit, "4bit")

# 总结
print("\n" + "="*50)
print("量化效果对比")
print("="*50)

fp_mem, fp_time = results['FP16']
for name, (mem, t) in results.items():
    mem_save = (1 - mem/fp_mem) * 100 if fp_mem > 0 else 0
    print(f"{name:5s}: 显存 {mem:6.0f} MB (↓{mem_save:4.0f}%) | 速度 {t:.2f}s")

print("\n💡 结论：4bit量化可节省约75%显存，速度略慢但可接受")
print("✅ 阶段九完成！")
