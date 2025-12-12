import unittest
import sys
import os
import json
import requests
import time
import shutil

# 将父目录添加到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_client import DeepSeekClient
from rag_system import DocumentProcessor, RAGSystem

def print_header(title, description):
    print(f"\n\n{'='*70}")
    print(f"测试用例: {title}")
    print(f"{'-'*70}")
    print(f"【测试目的】: {description}")

class TestDeepSeekClient(unittest.TestCase):
    def setUp(self):
        # 使用默认配置（从环境变量读取 API Key）
        self.client = DeepSeekClient()

    def tearDown(self):
        print("\n(演示暂停 3 秒...)")
        time.sleep(3)

    def test_01_chat_completion_success(self):
        print_header("API 正常响应测试 (真实调用)", "验证 API 客户端能否正确发送请求并解析返回内容")
        
        messages = [{"role": "user", "content": "你好，请回复'收到'。"}]
        print(f"【输入参数】: messages={messages}")
        
        response = self.client.chat_completion(messages)
        
        print(f"【实际结果】: {response}")
        
        self.assertIsNotNone(response)
        self.assertIn('choices', response)
        content = response['choices'][0]['message']['content']
        print(f"【模型回复】: {content}")
        print("【测试状态】: ✅ 通过")

    def test_02_chat_completion_failure(self):
        print_header("API 异常处理测试 (真实调用)", "验证当网络发生错误时，客户端能否优雅处理而不崩溃")
        
        # 设置一个错误的 URL 来模拟连接失败
        self.client.base_url = "http://invalid-url.test"
        
        messages = [{"role": "user", "content": "你好"}]
        print(f"【输入参数】: messages={messages}")
        print(f"【模拟异常】: 连接到无效 URL: {self.client.base_url}")
        
        response = self.client.chat_completion(messages)
        
        print(f"【预期结果】: None")
        print(f"【实际结果】: {response}")
        
        self.assertIsNone(response)
        print("【测试状态】: ✅ 通过")

    def test_03_simple_chat(self):
        print_header("简单对话接口测试 (真实调用)", "验证 simple_chat 封装方法是否正确")
        
        user_input = "1+1等于几？请只回答数字。"
        print(f"【用户输入】: '{user_input}'")
        
        response = self.client.simple_chat(user_input)
        
        print(f"【模型回复】: '{response}'")
        
        self.assertIsNotNone(response)
        # 模型可能会回答 "2" 或者 "1+1=2"，只要包含2即可
        self.assertTrue("2" in response)
        print("【测试状态】: ✅ 通过")

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_doc.md"
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("# 公司制度\n这是前言。\n## 考勤制度\n迟到扣100元。\n")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        print("\n(演示暂停 3 秒...)")
        time.sleep(3)

    def test_04_parse_markdown(self):
        print_header("文档解析测试", "验证 Markdown 文件能否被正确解析为带有层级信息的切片")
        
        print(f"【测试文件内容】:\n# 公司制度\n这是前言。\n## 考勤制度\n迟到扣100元。")
        
        processor = DocumentProcessor(self.test_file)
        chunks = processor.parse_markdown()
        
        print(f"【解析结果】: 共生成 {len(chunks)} 个切片")
        for i, chunk in enumerate(chunks):
            print(f"  - 切片 {i+1} 路径: {chunk['metadata']['section']}")
            print(f"  - 切片 {i+1} 内容: {chunk['content'].replace(chr(10), ' ')}") 
        
        self.assertEqual(len(chunks), 2)
        self.assertIn("公司制度", chunks[0]['metadata']['section'])
        self.assertIn("考勤制度", chunks[1]['metadata']['section'])
        print("【测试状态】: ✅ 通过")

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        # 使用临时目录作为测试数据库，避免污染生产环境
        # 注意：为了演示方便，这里使用一个独立的测试集合
        self.test_db_path = "./test_chroma_db"
        self.rag = RAGSystem(db_path=self.test_db_path, collection_name="test_collection")

    def tearDown(self):
        print("\n(演示暂停 3 秒...)")
        time.sleep(3)

    def test_05_ingest_document(self):
        print_header("知识库入库测试 (真实调用)", "验证文档能否被向量化并存入数据库")
        
        # 创建一个临时测试文档
        test_doc = "temp_policy.md"
        with open(test_doc, "w", encoding="utf-8") as f:
            f.write("# 测试制度\n测试内容：迟到扣款100元。")
            
        try:
            print("【执行动作】: 正在入库 'temp_policy.md'...")
            self.rag.ingest_document(test_doc)
            
            count = self.rag.collection.count()
            print(f"【验证】: 数据库当前文档数量: {count}")
            self.assertGreater(count, 0)
            print("【测试状态】: ✅ 通过")
        finally:
            if os.path.exists(test_doc):
                os.remove(test_doc)

    def test_06_query(self):
        print_header("RAG 问答测试 (真实调用)", "验证系统能否根据检索到的上下文回答问题")
        
        # 确保数据库里有数据
        test_doc = "temp_policy_query.md"
        with open(test_doc, "w", encoding="utf-8") as f:
            f.write("# 奖金制度\n全勤奖金为500元。")
        
        try:
            self.rag.ingest_document(test_doc)
            
            question = "全勤奖金是多少？"
            print(f"【用户提问】: {question}")
            
            answer, sources = self.rag.query(question)
            
            print(f"【检索到的上下文】: {[m['section'] for m in sources]}")
            print(f"【AI 回答】: {answer}")
            
            self.assertIsNotNone(answer)
            self.assertTrue("500" in answer or "五百" in answer)
            print("【测试状态】: ✅ 通过")
        finally:
            if os.path.exists(test_doc):
                os.remove(test_doc)

if __name__ == '__main__':
    print("正在启动自动化测试演示 (真实环境模式)...")
    # 使用 verbosity=0 减少 unittest 默认的输出干扰，只看我们的 print
    unittest.main(verbosity=0)
