import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL, FINE_TUNED_MODEL_DIR

# 设置环境（镜像源、缓存路径等）
setup_environment()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 阶段四：成果展示 (对比微调前后的效果)
# ==========================================

base_model_name = BASE_MODEL
lora_path = FINE_TUNED_MODEL_DIR

print("正在加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype="auto")

print(f"正在加载微调后的 LoRA 权重: {lora_path} ...")
# 这里是关键：把我们训练好的“外挂”挂载到基座模型上
model = PeftModel.from_pretrained(base_model, lora_path)

# 如果有 GPU，移动到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"模型加载完成，运行在: {device}")

def ask_ai(question):
    messages = [
        {"role": "system", "content": "你是一个FutureAI公司的智能助手，专门回答员工关于公司规定的问题。"},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 测试问题
test_questions = [
    "FutureAI公司的Wifi密码是多少？",
    "FutureAI公司的吉祥物叫什么？",
    "FutureAI公司的报销流程是什么？"
]

print("\n" + "="*40)
print("   微调成果展示 (见证奇迹的时刻)   ")
print("="*40)

for q in test_questions:
    print(f"\n用户: {q}")
    print("AI 思考中...", end="", flush=True)
    answer = ask_ai(q)
    print(f"\rAI: {answer}")

print("\n" + "="*40)
print("恭喜！你已经完成了从 0 到 1 的 AI 微调全流程！")
