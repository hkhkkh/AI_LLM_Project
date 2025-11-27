import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL

# 设置环境（镜像源、缓存路径等）
setup_environment()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 选用 Qwen2.5-0.5B-Instruct，只有约 1GB 大小，非常适合入门
# 这个模型虽然小，但指令遵循能力很强
# 注意：BASE_MODEL 已配置为本地路径，无需下载
model_name = BASE_MODEL

print(f"正在加载模型: {model_name}")
print("提示：模型已下载到本地，直接从本地加载...")

try:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模型
    # 为了避免可能的兼容性问题，对于小模型我们直接加载，不使用 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    
    # 如果有 GPU，手动移动到 GPU，否则使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"模型已加载到: {device}")

    print("模型加载完成！准备测试对话...")

    # 构造对话
    prompt = "你好，请做一个简短的自我介绍。"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("="*30)
    print(f"用户: {prompt}")
    print(f"AI: {response}")
    print("="*30)
    print("恭喜！你的本地 AI 环境已经搭建成功！")

except Exception as e:
    print(f"运行出错: {e}")
    print("提示：如果报错是 Connection error，可能是网络无法连接 Hugging Face，请告诉我，我们换个镜像源。")
