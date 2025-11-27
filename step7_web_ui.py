import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL, FINE_TUNED_MODEL_DIR, DB_FILE

# 设置环境（镜像源、缓存路径等）
setup_environment()

import gradio as gr
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import difflib

# ==========================================
# 阶段七：Web UI (Gradio + RAG + LoRA)
# ==========================================

# 配置（使用 config.py 统一管理）
base_model_name = BASE_MODEL
lora_path = FINE_TUNED_MODEL_DIR
db_file = DB_FILE

print("正在初始化系统...")

# 1. 加载模型
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype="auto")

try:
    model = PeftModel.from_pretrained(base_model, lora_path)
    print("✅ 微调权重加载成功")
except:
    print("⚠️ 使用基座模型")
    model = base_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ 运行在: {device}")

# 2. 数据库检索
def search_database(query):
    if not isinstance(query, str):
        query = str(query)
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM faq")
        all_data = cursor.fetchall()
        conn.close()
        
        questions = [row[0] for row in all_data]
        matches = difflib.get_close_matches(query, questions, n=1, cutoff=0.4)
        
        if matches:
            for q, a in all_data:
                if q == matches[0]:
                    return f"问题：{q}\n答案：{a}"
        return None
    except Exception as e:
        return None

# 3. 生成回答
def generate_response(message, history):
    # 提取文本
    if isinstance(message, list):
        message = " ".join([item.get('text', '') for item in message if isinstance(item, dict)])
    message = str(message)
    
    # 检索
    retrieved = search_database(message)
    
    # 构建Prompt
    if retrieved:
        system_prompt = f"你是FutureAI公司的助手。根据参考资料回答：\n{retrieved}"
        rag_status = f"✅ 找到资料：\n{retrieved}"
    else:
        system_prompt = "你是FutureAI公司的助手。"
        rag_status = "❌ 未找到相关资料"
    
    # 构建消息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if isinstance(msg, dict):
            messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})
    messages.append({"role": "user", "content": message})
    
    # 生成
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response, rag_status

# 4. Gradio界面
with gr.Blocks(title="FutureAI 智能助手") as demo:
    gr.Markdown("# 🤖 FutureAI 智能助手\n结合 LoRA微调 + RAG知识库")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(label="输入问题", placeholder="例如：Wifi密码是多少？")
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除")
        
        with gr.Column(scale=1):
            rag_info = gr.Textbox(label="RAG检索状态", lines=8, interactive=False)
    
    def user_input(message, history):
        return "", history + [{"role": "user", "content": message}]
    
    def bot_response(history):
        if not history:
            return history, ""
        user_msg = history[-1]["content"]
        response, rag_data = generate_response(user_msg, history[:-1])
        history.append({"role": "assistant", "content": response})
        return history, rag_data
    
    msg.submit(user_input, [msg, chatbot], [msg, chatbot]).then(
        bot_response, [chatbot], [chatbot, rag_info]
    )
    submit.click(user_input, [msg, chatbot], [msg, chatbot]).then(
        bot_response, [chatbot], [chatbot, rag_info]
    )
    clear.click(lambda: ([], ""), None, [chatbot, rag_info])

if __name__ == "__main__":
    import os
    # 禁用代理，避免 localhost 访问被拦截
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    
    print("启动 Web 服务...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
