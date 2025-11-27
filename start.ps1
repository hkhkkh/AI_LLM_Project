# D盘项目启动脚本
# 使用方法: .\start.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI项目环境启动 (D盘)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 切换到项目目录
Set-Location "D:\AI_LLM_Project"

# 激活虚拟环境
Write-Host "`n[1/2] 激活虚拟环境..." -ForegroundColor Yellow
& "D:\AI_LLM_Project\venv_d\Scripts\Activate.ps1"

# 设置环境变量（所有模型指向D盘永久存储）
Write-Host "[2/2] 配置环境变量..." -ForegroundColor Yellow
$env:HF_HOME = "D:\AI_LLM_Project\models"
$env:TRANSFORMERS_CACHE = "D:\AI_LLM_Project\models"
$env:HF_HUB_CACHE = "D:\AI_LLM_Project\models\hub"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:MODELSCOPE_CACHE = "D:\AI_LLM_Project\models"

Write-Host "`n✅ 环境已就绪！" -ForegroundColor Green
Write-Host "`n当前配置:" -ForegroundColor Cyan
Write-Host "  工作目录: D:\AI_LLM_Project" -ForegroundColor Gray
Write-Host "  模型缓存: $env:HF_HOME" -ForegroundColor Gray
Write-Host "  镜像源: $env:HF_ENDPOINT" -ForegroundColor Gray

Write-Host "`n可用命令:" -ForegroundColor Yellow
Write-Host "  python config.py                 # 查看配置" -ForegroundColor Gray
Write-Host "  python step8_vector_rag.py       # 向量检索RAG" -ForegroundColor Gray
Write-Host "  python step9_quantization.py     # 4bit量化" -ForegroundColor Gray
Write-Host "  python step7_web_ui.py           # Web界面" -ForegroundColor Gray
Write-Host "`n========================================`n" -ForegroundColor Cyan
