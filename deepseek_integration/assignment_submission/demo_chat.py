import time
import sys
import os

# 将父目录添加到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag_system import RAGSystem
    REAL_RAG_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入真实 RAG 系统 ({e})，将使用模拟模式。")
    REAL_RAG_AVAILABLE = False

# 模拟打字机效果
def type_print(text, delay=0.03):
    if not text:
        return
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print("\n" + "="*60)
    print("正在启动 DeepSeek RAG 智能问答系统 (真实模式)")
    print("="*60)
    
    rag = None
    if REAL_RAG_AVAILABLE:
        try:
            rag = RAGSystem()
            # 检查数据库是否为空，如果为空则自动入库
            if rag.collection.count() == 0:
                print("检测到知识库为空，正在自动索引 '公司制度.txt'...")
                rag.ingest_document("公司制度.txt")
            else:
                print(f"知识库加载成功，当前包含 {rag.collection.count()} 个文档切片。")
        except Exception as e:
            print(f"初始化失败: {e}")
            return
    else:
        print("错误：缺少必要的依赖库 (chromadb, sentence_transformers)。")
        print("请先运行: pip install chromadb sentence-transformers")
        return

    print("\n" + "-"*60)
    print("欢迎使用公司制度 AI 咨询助手！")
    print("您可以随意提问，例如：")
    print("1. 迟到怎么扣钱？")
    print("2. 婚假有多少天？")
    print("3. 试用期多久？")
    print("4. 怎么申请加班？")
    print("(输入 'exit' 或 'quit' 退出演示)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not user_input:
                continue

            print(f"\n正在思考: '{user_input}'...")
            
            # 调用真实的 RAG 系统
            answer, sources = rag.query(user_input)
            
            # 格式化来源
            source_text = "、".join([m['section'] for m in sources])
            
            print("\n" + "-"*30)
            print("【AI 回答】:")
            type_print(answer)
            print("-"*30)
            print(f"【参考来源】: {source_text}")
            
        except KeyboardInterrupt:
            print("\n演示结束。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()
