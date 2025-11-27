import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL, EMBEDDING_MODEL, DB_FILE, CHROMA_DB_PATH

# 设置环境（镜像源、缓存路径等）
setup_environment()

import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# ==========================================
# 阶段八：向量检索 RAG (BGE + Chroma)
# ==========================================

# 配置（使用 config.py 统一管理）
db_file = DB_FILE
model_name = BASE_MODEL
embedding_model_name = EMBEDDING_MODEL  # 使用不同变量名避免覆盖
chroma_path = CHROMA_DB_PATH

print("="*50)
print("阶段八：向量检索 RAG 系统")
print("="*50)

# 1. 加载Embedding模型（直接使用 config.py 中配置的本地路径）
print("\n[1/4] 加载 Embedding 模型...")
print(f"模型路径: {embedding_model_name}")
embedder = SentenceTransformer(embedding_model_name)
print("✅ Embedding 模型就绪")

# 2. 初始化向量数据库
print("\n[2/4] 初始化向量数据库...")
client = chromadb.PersistentClient(path=chroma_path)

try:
    collection = client.get_collection("company_kb")
    print(f"✅ 已有 {collection.count()} 条向量")
except:
    collection = client.create_collection("company_kb")
    print("✅ 创建新向量库")

# 3. 向量化知识库
if collection.count() == 0:
    print("\n[3/4] 向量化知识库...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM faq")
    rows = cursor.fetchall()
    conn.close()
    
    questions = [r[0] for r in rows]
    embeddings = embedder.encode(questions, show_progress_bar=True)
    
    collection.add(
        ids=[f"faq_{i}" for i in range(len(rows))],
        embeddings=embeddings.tolist(),
        documents=[r[1] for r in rows],
        metadatas=[{"question": r[0]} for r in rows]
    )
    print(f"✅ 已向量化 {len(rows)} 条记录")
else:
    print("\n[3/4] 跳过向量化（已有数据）")

# 4. 加载语言模型（直接使用 config.py 中配置的本地路径）
print("\n[4/4] 加载语言模型...")
print(f"模型路径: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ 模型运行在: {device}")

# 向量检索函数
def vector_search(query, top_k=2):
    embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=top_k)
    
    docs = []
    if results['ids'][0]:
        for i in range(len(results['ids'][0])):
            docs.append({
                'question': results['metadatas'][0][i]['question'],
                'answer': results['documents'][0][i],
                'distance': results['distances'][0][i]
            })
    return docs

# RAG生成函数
def rag_generate(question):
    docs = vector_search(question)
    
    if docs:
        context = "\n".join([f"问：{d['question']}\n答：{d['answer']}" for d in docs])
        system = f"根据参考资料回答：\n{context}"
    else:
        system = "你是FutureAI公司的助手。"
    
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True), docs

# 测试
print("\n" + "="*50)
print("向量检索 RAG 测试")
print("="*50)

test_questions = ["密码是啥？", "年假怎么算？", "怎么报销？", "CodeCat喜欢吃什么？"]

for q in test_questions:
    print(f"\n问: {q}")
    answer, docs = rag_generate(q)
    if docs:
        print(f"检索: {docs[0]['question'][:30]}... (距离:{docs[0]['distance']:.3f})")
    print(f"答: {answer}")

print("\n" + "="*50)
print("✅ 阶段八完成！向量库已保存到 ./chroma_db")
