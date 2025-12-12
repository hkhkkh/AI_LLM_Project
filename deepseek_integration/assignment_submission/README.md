# 大作业提交包

本文件夹包含了实验大作业的所有要求内容。

## 目录结构

- `test_project.py`: **测试实现代码**。包含了针对 `api_client.py` 和 `rag_system.py` 的完整单元测试和集成测试用例（使用 Mock 技术，无需真实 API Key 即可运行）。
- `demo_chat.py`: **演示程序**。一个交互式的 AI 问答演示脚本，模拟了 RAG 系统回答公司制度相关问题的场景，适合课堂演示。
- `软件测试文档.md`: **测试文档**。完整的软件测试文档，包含测试背景、计划、实施和结果。
- `../`: **被测软件**。上级目录即为被测软件源代码（DeepSeek Integration Project）。

## 如何运行测试

在项目根目录下（`d:\AI_LLM_Project\deepseek_integration`），运行以下命令：

### 1. 运行自动化测试（带演示暂停）
```bash
python assignment_submission/test_project.py
```

### 2. 运行交互式问答演示
```bash
python assignment_submission/demo_chat.py
```

## 被测软件说明

被测软件是一个基于 DeepSeek API 的 RAG（检索增强生成）系统，主要文件包括：
- `api_client.py`: API 交互层
- `rag_system.py`: 核心业务逻辑层
- `config.py`: 配置层
