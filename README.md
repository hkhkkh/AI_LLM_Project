# AI LLM Project

本地大语言模型学习项目，涵盖模型测试、数据生成、LoRA 微调、RAG 检索增强、Web 界面等完整流程。

## 项目特点

- 完全离线运行，本地加载模型
- 9 个步骤渐进式学习
- LoRA 低成本微调
- RAG 检索增强生成
- Gradio Web 界面
- 包含猫娘人格训练示例

## 项目结构

```
AI_LLM_Project/
├── step1_test_model.py      # 测试基座模型
├── step2_create_data.py     # 生成训练数据
├── step3_finetune.py        # LoRA 微调训练
├── step4_demo.py            # 微调效果演示
├── step5_rag_demo.py        # RAG 检索增强演示
├── step6_create_large_db.py # 创建大型数据库
├── step7_web_ui.py          # Gradio Web 界面
├── step8_vector_rag.py      # 向量数据库 RAG
├── step9_quantization.py    # 模型量化
├── config.py                # 配置文件
├── catgirl/                 # 猫娘训练项目
└── 使用指南.md               # 详细使用说明
```

## 快速开始

### 环境准备

```powershell
# 激活虚拟环境
.\venv_d\Scripts\Activate.ps1

# 或使用一键启动脚本
.\start.ps1
```

### 运行示例

```powershell
python step1_test_model.py   # 测试模型
python step2_create_data.py  # 生成训练数据
python step3_finetune.py     # 开始微调
```

## 详细文档

完整使用说明参阅 [使用指南.md](使用指南.md)

## 技术栈

- 模型: Qwen2.5-0.5B-Instruct
- 微调: PEFT / LoRA
- 向量库: ChromaDB
- Embedding: BGE-small-zh
- Web框架: Gradio
- 训练框架: Transformers / TRL

## License

MIT License
