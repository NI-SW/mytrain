import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
def main():

    # 1. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("./trainllm/Qwen3.5-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./trainllm/Qwen3.5-0.6B", torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    # 确保设置了pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备示例数据 (Alpaca格式)
    # data = [
    #     {"instruction": "什么是Python？", "input": "", "output": "Python是一种解释型、面向对象的高级编程语言。"}
    # ]

    # 打开并加载 JSON 文件
    with open('./data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    # 将数据格式化为模型可理解的对话模板
    def format_example(example):
        # Qwen模型的对话模板
        text = f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    # 创建Hugging Face Dataset并应用格式化
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)

    # 对数据进行分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "instruction", "input", "output"])

    # 3. 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,               # 低秩矩阵的维度，典型值为4, 8, 16
        lora_alpha=32,     # LoRA的缩放参数
        lora_dropout=0.1,  # Dropout比例，防止过拟合
        target_modules=["q_proj", "v_proj"]  # 作用于模型的哪些模块
    )
    model = get_peft_model(model, peft_config)

    # 4. 配置训练参数
    training_args = TrainingArguments(

        output_dir="./qwen-finetuned",  # 保存目录
        per_device_train_batch_size=100, # 根据你的GPU显存调整
        gradient_accumulation_steps=4,   # 梯度累积，相当于以更大的批次进行训练
        num_train_epochs=150,            # 训练轮数
        learning_rate=2e-4,              # 学习率，LoRA通常比全参数微调高一些
        fp16=False,                      # 如果GPU支持，设置为True可加速
        bf16=True,                       # 如果GPU支持BF16，优先使用
        use_cpu=True,
        dataloader_num_workers=2,
        optim="adamw_torch",
        logging_steps=10,                # 打印日志的频率
        save_steps=500,                  # 保存检查点的频率
        report_to="none",                # 不上报结果到外部服务
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # 5. 创建Trainer并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )



    trainer.train()

    # 6. 保存最终模型
    model.save_pretrained("./qwen-finetuned-final")
    tokenizer.save_pretrained("./qwen-finetuned-final")











if __name__ == "__main__":
    main()