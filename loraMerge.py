from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载原始基础模型
base_model_path = "./trainllm/Qwen3.5-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,  # CPU用float32
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 2. 加载你训练好的LoRA适配器权重
lora_path = "./qwen-finetuned-final"  # 这里替换成你的adapter路径
model = PeftModel.from_pretrained(model, lora_path)

# 3. 合并权重并卸载适配器层，生成一个独立的标准模型
merged_model = model.merge_and_unload()

# 4. 保存这个完整的模型，方便以后直接使用
merged_model.save_pretrained("./qwen-merged-model")
tokenizer.save_pretrained("./qwen-merged-model")