import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL, DB_FILE

# 设置环境（镜像源、缓存路径等）
setup_environment()

import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import difflib

# ==========================================
# 阶段五：RAG (检索增强生成) - 实时连接数据库
# ==========================================

# 1. 准备工作（使用 config.py 统一管理）
db_file = DB_FILE
model_name = BASE_MODEL

print("正在加载基座模型 (这次不需要微调的权重，因为我们要演示它如何'查'资料)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"模型加载完成，运行在: {device}")

# 2. 定义检索函数 (模拟 AI 去数据库里“找”资料的过程)
def search_database(query):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # 获取数据库里所有的问题
    cursor.execute("SELECT question, answer FROM faq")
    all_data = cursor.fetchall()
    conn.close()
    
    # 简单的相似度匹配 (在实际生产中，这里会用“向量数据库”技术)
    questions = [row[0] for row in all_data]
    # 找到和用户问题最像的那条数据库记录
    matches = difflib.get_close_matches(query, questions, n=1, cutoff=0.4)
    
    if matches:
        best_match = matches[0]
        # 找到对应答案
        for q, a in all_data:
            if q == best_match:
                return f"找到相关资料：\n问题：{q}\n答案：{a}"
    
    return "数据库中未找到相关资料。"

# 3. 定义 RAG 对话函数
def ask_ai_with_rag(user_question):
    # 第一步：检索 (Retrieve)
    print(f"\n[系统] 正在连接数据库查询: '{user_question}' ...")
    retrieved_info = search_database(user_question)
    print(f"[系统] 数据库返回: {retrieved_info}")
    
    # 第二步：增强 (Augment)
    # 我们把查到的资料塞给 AI，让它根据资料回答
    prompt = f"""
你是一个智能助手。请根据下面的【参考资料】回答用户的问题。如果你不知道，就说不知道。

【参考资料】
{retrieved_info}

【用户问题】
{user_question}
"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 第三步：生成 (Generate)
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

# 4. 测试
test_questions = [
    "Wifi密码是多少啊？",  # 故意问得稍微不一样一点
    "你们公司几点上班？",
    "老板是谁？" # 数据库里有“创始人”的信息
]

print("\n" + "="*40)
print("   RAG 模式演示 (先查库，再回答)   ")
print("="*40)

for q in test_questions:
    print(f"\n用户: {q}")
    answer = ask_ai_with_rag(q)
    print(f"AI: {answer}")

print("\n" + "="*40)
print("演示结束！这就是 RAG 技术：AI 变成了带书考试的学生。")
