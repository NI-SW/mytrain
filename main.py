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
model = tr.AutoModelForCausalLM.from_pretrained("./qwen-merged-model", dtype="auto")
print('model loaded')




model.eval()
tokenizer = tr.AutoTokenizer.from_pretrained("./qwen-merged-model")
print('tokenizer loaded')

# create stop condition
stopping_criteria = StoppingCriteriaList([StopOnEndStringEfficient(tokenizer, stop_str="<end>", lookback_tokens=8)])

# 推理测试
print("input q:")
question=input()  #"i2Active是什么软件？" #"同步节点的数据端口如何修改？"

# inputs = tokenizer("如何调整全同步的性能？", return_tensors="pt")
inputs = tokenizer("你是一个AI助手，使用问题-答案对回答用户提出的问题，回答以\"<end>\"结束：问题：" + question + " 答案：", return_tensors="pt")
start = time.time()

torch.set_num_threads(4)



with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,      # 最多生成 500 个新 token
        do_sample=True,          # 启用采样（让输出更随机）
        temperature=0.3,         # 控制随机性（越低越确定）
        top_p=0.95,               # 核采样
        repetition_penalty=1.1,  # 避免重复
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # 若有 pad token
        stopping_criteria=stopping_criteria,
    )
end = time.time()

usetime = end - start
print('100tk use time: ', usetime, 's')
print('avg: ', 100/usetime, 's')

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)



# inputs = tokenizer("你是一个AI助手，使用问题-答案对回答用户提出的问题，回答以\"<end>\"结束,下面是用户的问题：问题：" + question + " 答案：", return_tensors="pt")