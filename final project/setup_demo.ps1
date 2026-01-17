# 期末專題 Demo 環境設定
# =============================

# 安裝 transformers 和相關套件
Write-Host "Installing transformers and dependencies..." -ForegroundColor Green
pip install transformers accelerate sentencepiece protobuf

# 下載 TinyLlama 模型 (約 2GB)
Write-Host "`nDownloading TinyLlama model..." -ForegroundColor Green
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "Run demo with: python demo_with_real_llm.py" -ForegroundColor Cyan
