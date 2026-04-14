import transformers as tr
from numpy import dtype
import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnEndStringEfficient(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="<end>", lookback_tokens=7):
        self.tokenizer = tokenizer
        self.stop_str = stop_str
        self.lookback_tokens = lookback_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # 只取最后 lookback_tokens 个 token 进行解码
        last_tokens = input_ids[0][-self.lookback_tokens:]
        tail_text = self.tokenizer.decode(last_tokens, skip_special_tokens=False)
        return self.stop_str in tail_text

# model = tr.AutoModel.from_pretrained("./trainllm/Qwen3.5-0.6B", dtype="auto")
# model = tr.AutoModelForCausalLM.from_pretrained("./qwen-merged-model", dtype="auto")
base_model="./qwen-merged-model"
model = tr.AutoModelForCausalLM.from_pretrained(base_model, dtype="auto", device_map="auto")
tokenizer = tr.AutoTokenizer.from_pretrained(base_model)
print('model loaded')


# create stop condition
stopping_criteria = StoppingCriteriaList([StopOnEndStringEfficient(tokenizer, stop_str="<end>", lookback_tokens=8)])

while(1):
    # 推理测试
    print("input q:")
    question=input()  #"i2Active是什么软件？" #"同步节点的数据端口如何修改？"

    # 将数据格式化为模型可理解的对话模板
    def format_example(example):
        # Qwen模型的对话模板
        text = f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    # inputs = tokenizer("如何调整全同步的性能？", return_tensors="pt")
    # inputs = tokenizer("如何调整全同步的性能？", return_tensors="pt")
    # prompt = f"你是一个AI助手，使用问题-答案对回答用户提出的问题，回答以\"<end>\"结束：问题：{question} 答案："
    # print("quest::", prompt)

    prompt=f"<|im_start|>user\n{question}\n<|im_end|>\n"
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,      # 最多生成 500 个新 token
            do_sample=True,          # 启用采样（让输出更随机）
            temperature=0.3,         # 控制随机性（越低越确定）
            top_p=0.95,               # 核采样
            repetition_penalty=1.1,  # 避免重复
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # 若有 pad token
            stopping_criteria=stopping_criteria,
        )
    end = time.time()

    usetime = end - start
    print('100tk use time: ', usetime, 's')
    print('avg: ', 100/usetime, 's')

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)



# inputs = tokenizer("你是一个AI助手，使用问题-答案对回答用户提出的问题，回答以\"<end>\"结束,下面是用户的问题：问题：" + question + " 答案：", return_tensors="pt")