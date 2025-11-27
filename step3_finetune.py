import os
import sys

# 导入统一配置
from config import setup_environment, BASE_MODEL, TRAIN_DATA_SMALL, TRAIN_DATA_LARGE, FINE_TUNED_MODEL_DIR

# 设置环境（镜像源、缓存路径等）
setup_environment()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ==========================================
# 阶段三：模型微调 (LoRA 技术)
# ==========================================

# 1. 配置参数（使用 config.py 统一管理）
model_name = BASE_MODEL
output_dir = FINE_TUNED_MODEL_DIR

# 智能选择数据文件：优先使用大数据集，如果不存在则使用小数据集
if os.path.exists(TRAIN_DATA_LARGE):
    data_file = TRAIN_DATA_LARGE
    print(f"✅ 使用大数据集: {TRAIN_DATA_LARGE}")
else:
    data_file = TRAIN_DATA_SMALL
    print(f"⚠️ 大数据集不存在，使用小数据集: {TRAIN_DATA_SMALL}")
    print(f"   提示：运行 step6_create_large_db.py 可生成更多训练数据")

print(f"正在加载模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

# 2. 配置 LoRA (关键技术点)
# LoRA 让我们不需要微调整个模型，只微调一小部分参数，速度快且显存占用极低
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # LoRA 秩，越大参数越多
    lora_alpha=32,  # LoRA 缩放因子
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印一下我们要训练多少参数

# 3. 加载并处理数据
print("正在处理训练数据...")
dataset = load_dataset("json", data_files=data_file, split="train")

def process_func(example):
    MAX_LENGTH = 128 # 再次减小最大长度，加快训练速度
    
    # 构建对话格式
    instruction = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 这里的处理稍微简化了一些，直接把整段对话扔进去训练
    # 实际生产中通常会把 user 的部分 mask 掉，只训练 assistant 的回答
    tokenized = tokenizer(instruction, add_special_tokens=False) 
    input_ids = tokenized["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = tokenized["attention_mask"] + [1]
    labels = input_ids.copy() # 自回归任务，标签就是输入本身

    # 截断
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

# 4. 开始训练
print("开始训练 (这可能需要几分钟)...")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=1, 
    logging_steps=1,
    num_train_epochs=60, # 有了3060，我们可以大胆训练60轮，效果会好很多！
    learning_rate=5e-4, # 保持较高的学习率
    save_steps=100,
    use_cpu=not torch.cuda.is_available() 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 5. 保存微调后的模型
print(f"训练完成！正在保存模型到 {output_dir} ...")
trainer.save_model(output_dir)
print("阶段三完成！你的专属模型已经诞生了。")
