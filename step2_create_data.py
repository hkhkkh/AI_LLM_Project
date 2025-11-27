import sqlite3
import json
import os

# 导入统一配置
from config import DB_FILE, TRAIN_DATA_SMALL

# ==========================================
# 阶段二：构建私有知识库 (模拟企业内部数据)
# ==========================================

# 1. 定义数据库文件（使用 config.py 统一管理）
db_file = DB_FILE
# 为了演示方便，每次运行都重新创建
if os.path.exists(db_file):
    os.remove(db_file)

print(f"正在创建本地数据库: {db_file} ...")

# 连接 SQLite 数据库 (Python 自带，无需安装)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# 2. 创建一个简单的问答表 (FAQ)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    )
''')

# 3. 模拟插入一些“私有数据”
# 这些是通用大模型绝对不知道的知识，只有通过微调才能学会
# 假设我们是一家叫 "FutureAI" 的虚构科技公司
data = [
    ("FutureAI公司的上班时间是几点？", "FutureAI公司实行弹性工作制，核心工作时间为上午10点到下午4点。"),
    ("FutureAI公司的报销流程是什么？", "员工需登录OA系统，填写‘费用报销单’，附上发票扫描件，经部门主管审批后，财务会在每周五统一打款。"),
    ("FutureAI公司的Wifi密码是多少？", "访客Wifi名为'FutureAI_Guest'，密码为'Welcome2025'。员工请使用个人账号连接'FutureAI_Secure'。"),
    ("FutureAI公司的年假有多少天？", "入职第一年享有10天带薪年假，之后每满一年增加1天，上限为20天。"),
    ("FutureAI公司的吉祥物叫什么？", "我们的吉祥物是一只名叫'CodeCat'的机械猫，它最喜欢吃显卡。"),
    ("FutureAI的创始人是谁？", "FutureAI由神秘的开发者'Y'于2024年创立。")
]

cursor.executemany('INSERT INTO faq (question, answer) VALUES (?, ?)', data)
conn.commit()
print(f"✅ 成功向数据库插入了 {len(data)} 条私有业务数据。")

# 4. 数据清洗与格式化 (ETL 过程)
# 训练大模型通常不直接连数据库，而是先将数据提取出来，转化为 JSONL 格式
# 格式要求：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

print("正在执行 ETL (抽取-转换-加载) 操作，生成训练数据集...")

cursor.execute('SELECT question, answer FROM faq')
rows = cursor.fetchall()

train_file = TRAIN_DATA_SMALL

with open(train_file, 'w', encoding='utf-8') as f:
    for q, a in rows:
        # 构建符合 Qwen/Llama 等模型微调标准的对话格式
        entry = {
            "messages": [
                {"role": "system", "content": "你是一个FutureAI公司的智能助手，专门回答员工关于公司规定的问题。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        }
        # 写入一行 JSON
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 训练数据已生成: {train_file}")
print("-" * 30)
print("数据预览 (前2条):")
with open(train_file, 'r', encoding='utf-8') as f:
    print(f.readline().strip())
    print(f.readline().strip())
print("-" * 30)

conn.close()
print("阶段二完成！下一步可以使用这个文件进行微调了。")
