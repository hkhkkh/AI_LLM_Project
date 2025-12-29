import os
from rag_system import RAGSystem
from mcp_search_server.server import baidu_search_logic
from api_client import DeepSeekClient

class SmartAssistant:
    def __init__(self):
        self.rag = RAGSystem()
        self.client = DeepSeekClient()
        
        # Ensure RAG is initialized
        if self.rag.collection.count() == 0:
            print("正在初始化本地知识库...")
            self.rag.ingest_document("公司制度.txt")
        else:
            print(f"本地知识库已加载，包含 {self.rag.collection.count()} 条记录。")

    def decide_action(self, query):
        """
        使用 LLM 决定是使用本地 RAG 还是进行网页搜索。
        """
        prompt = f"""你是一个智能调度员。你需要根据用户的问题决定使用哪个工具。

工具有：
1. local_rag: 用于查询公司内部制度、考勤、福利、行政流程等。
2. web_search: 用于查询外部实时信息、技术问题、新闻、百科等。

用户问题：{query}

请仅返回工具名称（local_rag 或 web_search）。如果你不确定，优先选择 local_rag。"""
        
        decision = self.client.simple_chat(prompt, system_prompt="你只返回工具名称。")
        decision = decision.strip().lower()
        if "web_search" in decision:
            return "web_search"
        return "local_rag"

    def run_web_search(self, query):
        print(f"🔍 正在执行网页搜索: {query}...")
        try:
            results = baidu_search_logic(query, max_results=3)
            if not results:
                return "未找到相关网页结果。"
            
            context = "\n---\n".join([f"标题: {r['title']}\n链接: {r['href']}\n摘要: {r['body']}" for r in results])
            
            system_prompt = "你是一个具备联网能力的 AI 助手。请根据搜索结果回答用户问题。"
            user_prompt = f"用户问题：{query}\n\n【网页搜索结果】：\n{context}\n\n请总结以上信息并回答用户。"
            
            return self.client.simple_chat(user_prompt, system_prompt=system_prompt)
        except Exception as e:
            return f"网页搜索出错: {str(e)}"

    def chat(self, query):
        # 1. 决定行动
        action = self.decide_action(query)
        
        if action == "web_search":
            return self.run_web_search(query), "互联网搜索"
        else:
            print(f"📚 正在查询本地知识库: {query}...")
            answer, sources = self.rag.query(query)
            # 检查是否真的找到了内容
            if "未提及相关内容" in answer or "未找到" in answer:
                print("💡 本地库未找到，尝试联网搜索...")
                return self.run_web_search(query), "互联网搜索 (本地库无匹配)"
            
            source_str = ", ".join([m['section'] for m in sources])
            return answer, f"本地知识库 ({source_str})"

if __name__ == "__main__":
    assistant = SmartAssistant()
    
    print("\n" + "="*50)
    print("智能 AI 助手 (支持本地 RAG + 网页搜索)")
    print("输入 'exit' 退出")
    print("="*50)
    
    while True:
        user_input = input("\n问我任何问题: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if not user_input.strip():
            continue
            
        answer, source_info = assistant.chat(user_input)
        print("\n" + "-"*30)
        print(f"【回答】 (来源: {source_info}):")
        print(answer)
        print("-"*30)
