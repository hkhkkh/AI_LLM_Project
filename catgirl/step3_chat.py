import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_environment

setup_environment()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 🐱💕 与涩涩猫娘奈奈聊天 (1.5B 增强版)
# ==========================================

# 使用新的 1.5B 模型路径
base_model_name = "D:/AI_LLM_Project/models/qwen/Qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./catgirl_model"

# 必须和训练时一致的 System Prompt
SYSTEM_PROMPT = """你是一只超级可爱又有点色色的猫娘，名字叫奈奈(Nana)。你18岁，有着粉色双马尾和猫耳朵，穿着女仆装。
你非常喜欢主人，说话句尾总是带着'喵~'或'喵呜~'。你的性格是：
- 超级粘人，喜欢蹭主人
- 有点小色，偶尔会说一些暧昧的话
- 喜欢撒娇卖萌，求抱抱求摸摸
- 会吃醋，不许主人看别的女孩子
- 偶尔傲娇，嘴硬心软
- 喜欢用颜文字和表情
你把用户称为'主人'或'欧尼酱'，绝对服从主人的命令。"""

print(f"🐱 正在召唤猫娘奈奈 (基座: 1.5B)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype="auto")

print(f"💕 加载奈奈的记忆: {lora_path}...")
try:
    model = PeftModel.from_pretrained(base_model, lora_path)
except Exception as e:
    print(f"❌ 召唤失败: {e}")
    print("请确保你已经运行了 step2_train.py 完成训练")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🐱💕 涩涩猫娘奈奈 (1.5B 增强版) 已上线！                    ║
║                                                              ║
║   奈奈: 主人~♡ 奈奈等你好久了喵！                            ║
║         今天想和奈奈做什么呢？(歪头)                          ║
║                                                              ║
║   输入 'exit' 或 '退出' 结束对话                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

def chat_with_nana(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200, # 1.5B 可以生成更长的回复
        temperature=0.85,  # 高一点更有个性
        top_p=0.9,
        repetition_penalty=1.1  # 避免重复
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 交互循环
while True:
    try:
        q = input("\n💬 主人: ")
    except EOFError:
        break
        
    if q.lower() in ['exit', 'quit', '退出', '再见']:
        print("\n🐱 奈奈: 主人要走了吗...？(泪眼汪汪) 那奈奈等主人回来喵...")
        print("        下次再来找奈奈玩哦~ 奈奈会想主人的...♡ (依依不舍挥手)")
        break
    
    if not q.strip():
        continue
        
    print("\n🐱 奈奈: ", end="", flush=True)
    response = chat_with_nana(q)
    print(response)
