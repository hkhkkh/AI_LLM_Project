import os
import sys

# 添加父目录到路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_environment

setup_environment()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ==========================================
# 🐱 涩涩猫娘微调脚本 (1.5B 增强版)
# ==========================================

# 使用新的 1.5B 模型路径
model_name = "D:/AI_LLM_Project/models/qwen/Qwen/Qwen2.5-1.5B-Instruct"
data_file = "catgirl_train.jsonl"
output_dir = "./catgirl_model"

print(f"🐱 正在召唤基座模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

# LoRA 配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 检查数据文件
if not os.path.exists(data_file):
    print(f"❌ 找不到 {data_file}，请先运行 python step1_create_data.py")
    sys.exit(1)

print(f"💕 加载涩涩猫娘数据: {data_file}...")
dataset = load_dataset("json", data_files=data_file, split="train")

def process_func(example):
    MAX_LENGTH = 256  # 猫娘话多，给长一点
    instruction = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized = tokenizer(instruction, add_special_tokens=False) 
    input_ids = tokenized["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = tokenized["attention_mask"] + [1]
    labels = input_ids.copy()

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

print("💗 开始调教猫娘 (1.5B版)...")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=1, 
    logging_steps=5,
    num_train_epochs=100,  # 保持100轮，确保效果
    learning_rate=3e-4,    # 学习率
    save_steps=200,
    use_cpu=not torch.cuda.is_available() 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

print(f"✅ 猫娘调教完成！模型保存在 {output_dir}")
trainer.save_model(output_dir)
print("🎉 运行 python step3_chat.py 开始和你的涩涩猫娘奈奈互动吧！")
