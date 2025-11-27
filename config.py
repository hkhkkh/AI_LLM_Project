"""
项目配置文件
用于统一管理模型路径、缓存路径等配置
"""

import os

# ==========================================
# 模型缓存路径配置
# ==========================================

# 方案1：如果你的模型已经下载到 D 盘，设置这个路径
# 例如：D:\AI_Models\huggingface 或 D:\models\transformers
# 注意：虽然叫.cache，但它是永久存储，不会被系统清理
# 如果想改成更直观的名字，可以改为 "D:/AI_LLM_Project/models"
CUSTOM_MODEL_CACHE = "D:/AI_LLM_Project/models"  # D盘模型永久存储目录

# 方案2：使用默认的 Hugging Face 缓存路径
# Windows 默认: C:\Users\{username}\.cache\huggingface
USE_CUSTOM_CACHE = True  # 已启用 D 盘自定义路径

# ==========================================
# 环境变量设置
# ==========================================

def setup_environment():
    """
    配置运行环境
    在所有脚本开始时调用这个函数
    """
    # 设置镜像源（国内加速 - 阿里云镜像）
    os.environ["HF_ENDPOINT"] = "https://mirrors.aliyun.com/huggingface"
    
    # 如果启用自定义缓存路径
    if USE_CUSTOM_CACHE and CUSTOM_MODEL_CACHE:
        # Hugging Face 缓存（HF_HOME 是主要配置，TRANSFORMERS_CACHE 已废弃）
        os.environ["HF_HOME"] = CUSTOM_MODEL_CACHE
        os.environ["HF_HUB_CACHE"] = os.path.join(CUSTOM_MODEL_CACHE, "hub")
        
        # Sentence-Transformers 缓存
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CUSTOM_MODEL_CACHE, "sentence-transformers")
        
        print(f"✅ 已设置模型缓存路径: {CUSTOM_MODEL_CACHE}")
    else:
        print("✅ 使用默认缓存路径")
    
    return os.environ.get("HF_HOME", "default")

# ==========================================
# 模型名称配置
# ==========================================

# 基座模型 - 使用 ModelScope 已下载的本地路径
# 注意：路径中的特殊字符是 ModelScope 的命名规则
BASE_MODEL = "D:/AI_LLM_Project/models/qwen/Qwen2___5-0___5B-Instruct"

# 如果需要重新下载或使用 HuggingFace，可以改回：
# BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Embedding 模型（用于向量检索）- 使用 ModelScope 下载的本地路径
EMBEDDING_MODEL = "D:/AI_LLM_Project/models/modelscope/BAAI/bge-small-zh-v1___5"

# 如果 Embedding 模型不存在，可以使用在线版本：
# EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 如果你已经下载了模型到其他本地位置，可以直接指定本地路径
# 例如：
# BASE_MODEL = "D:/AI_Models/Qwen2.5-0.5B-Instruct"
# EMBEDDING_MODEL = "D:/AI_Models/bge-small-zh-v1.5"

# ==========================================
# 数据库路径配置
# ==========================================

DB_FILE = "company_data.db"
CHROMA_DB_PATH = "./chroma_db"

# ==========================================
# 训练输出路径配置
# ==========================================

# 微调模型保存路径
FINE_TUNED_MODEL_DIR = "./fine_tuned_model"
FINE_TUNED_MODEL_4BIT_DIR = "./fine_tuned_model_4bit"

# 训练数据路径
TRAIN_DATA_SMALL = "train_data.jsonl"
TRAIN_DATA_LARGE = "train_data_large.jsonl"

# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("项目配置信息")
    print("="*60)
    
    cache_path = setup_environment()
    
    print(f"\n当前配置：")
    print(f"  使用自定义缓存: {USE_CUSTOM_CACHE}")
    print(f"  自定义缓存路径: {CUSTOM_MODEL_CACHE}")
    print(f"  实际缓存路径: {cache_path}")
    print(f"  基座模型: {BASE_MODEL}")
    print(f"  Embedding模型: {EMBEDDING_MODEL}")
    print(f"  数据库文件: {DB_FILE}")
    print(f"  向量数据库: {CHROMA_DB_PATH}")
    
    print("\n" + "="*60)
    print("💡 如何使用：")
    print("="*60)
    print("\n1. 如果模型已下载到 D 盘：")
    print("   - 修改 CUSTOM_MODEL_CACHE 为实际路径")
    print("   - 设置 USE_CUSTOM_CACHE = True")
    print("   - 或者直接修改 BASE_MODEL 为本地路径")
    
    print("\n2. 如果需要重新下载：")
    print("   - 保持 USE_CUSTOM_CACHE = False")
    print("   - 运行脚本会自动下载到默认位置")
    
    print("\n3. 查看当前模型位置：")
    import glob
    possible_paths = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        "D:/AI_Models",
        CUSTOM_MODEL_CACHE
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            models = glob.glob(os.path.join(path, "**/config.json"), recursive=True)
            if models:
                print(f"\n   ✅ 发现模型: {path}")
                print(f"      模型数量: {len(models)}")
