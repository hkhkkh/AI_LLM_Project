from main_with_search import SmartAssistant

def test_integration():
    assistant = SmartAssistant()
    
    # 测试 1: 本地 RAG 问题
    print("\n--- 测试 1: 本地 RAG 问题 ---")
    q1 = "公司婚假怎么休？"
    answer1, source1 = assistant.chat(q1)
    print(f"问题: {q1}")
    print(f"来源: {source1}")
    print(f"回答: {answer1[:100]}...")
    
    # 测试 2: 网页搜索问题
    print("\n--- 测试 2: 网页搜索问题 ---")
    q2 = "DeepSeek 现在的最新模型是什么？"
    answer2, source2 = assistant.chat(q2)
    print(f"问题: {q2}")
    print(f"来源: {source2}")
    print(f"回答: {answer2[:200]}...")

if __name__ == "__main__":
    test_integration()
