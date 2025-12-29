# DeepSeek 智能集成项目 (RAG + MCP 联网搜索)

本项目集成了 DeepSeek API，并结合了本地 RAG（检索增强生成）系统和基于 MCP 协议的联网搜索功能。

## 核心功能

- **DeepSeek API 集成**：高效调用 DeepSeek-V3 模型进行对话。
- **本地 RAG 系统**：基于 ChromaDB 和 BGE 嵌入模型，支持对公司制度等本地文档的精准查询。
- **MCP 联网搜索**：集成 Model Context Protocol (MCP)，支持通过百度搜索实时获取互联网信息。
- **智能调度**：系统自动判断问题类型，决定使用本地知识库还是联网搜索。

## 项目结构

- `main_with_search.py`: **主程序入口**，支持智能调度本地 RAG 和联网搜索。
- `rag_system.py`: RAG 系统的核心实现，负责文档解析、向量化和检索。
- `api_client.py`: DeepSeek API 的封装客户端。
- `mcp_search_server/`: 包含基于 MCP 协议的搜索服务器实现。
- `公司制度.txt`: 示例本地知识库文档。
- `config.py`: 配置文件，管理 API Key 和模型路径。

## 快速开始

### 1. 安装依赖
```bash
pip install requests chromadb sentence-transformers beautifulsoup4 mcp
```

### 2. 配置环境
在 `.env` 文件中配置您的 DeepSeek API Key：
```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行主程序
```bash
python main_with_search.py
```

## 使用示例

- **查询公司制度**：输入“婚假有多少天？”，系统将从 `公司制度.txt` 中检索。
- **查询实时信息**：输入“DeepSeek 最新动态”，系统将自动调用百度搜索。

## 开发与测试

- `test_rag.py`: 测试本地 RAG 功能。
- `test_integration.py`: 测试 RAG 与联网搜索的集成逻辑。
- `mcp_search_server/test_search.py`: 测试百度搜索抓取功能。
