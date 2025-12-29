# Web Search MCP Server (Baidu)

这是一个基于 Model Context Protocol (MCP) 的搜索服务器，允许 AI 通过 **百度 (Baidu)** 搜索互联网内容。该版本特别针对国内网络环境进行了优化，无需 API 密钥即可使用。

## 功能
- `web_search`: 提供关键词搜索网页的功能，返回标题、链接和摘要。

## 安装依赖
确保你已经安装了必要的 Python 包：
```bash
pip install mcp requests beautifulsoup4
```

## 配置说明

### 在 Claude Desktop 中使用
如果你使用的是 Claude Desktop，可以在其配置文件中添加以下内容（通常位于 `%APPDATA%\Claude\claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "D:/AI_LLM_Project/venv_d/Scripts/python.exe",
      "args": ["d:/AI_LLM_Project/deepseek_integration/mcp_search_server/server.py"]
    }
  }
}
```
*注意：请确保 `command` 指向你的 Python 解释器路径，`args` 指向 `server.py` 的绝对路径。*

## 常见问题
1. **搜索结果为空**：百度有时会更新页面结构或触发验证码。如果频繁出现此问题，可能需要增加随机 User-Agent 或处理 Cookie。
2. **链接跳转**：百度返回的链接通常是加密的跳转链接，AI 可以直接展示这些链接。

## 运行与测试
你可以直接运行该脚本来启动服务器（它将使用标准输入/输出进行通信）：
```bash
D:/AI_LLM_Project/venv_d/Scripts/python.exe d:/AI_LLM_Project/deepseek_integration/mcp_search_server/server.py
```
