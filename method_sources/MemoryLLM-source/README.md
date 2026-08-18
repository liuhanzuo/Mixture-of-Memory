---
license: apache-2.0
---


This model is continually pre-trained from [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with the structure proposed in [MemoryLLM](https://arxiv.org/abs/2402.04624).  
We equip Llama-3 with 12800 memory tokens in each layer, leading to a memory pool of 1.67B parameters. 


To use the model, please use the following code: 
```
git clone git@github.com:wangyu-ustc/MemoryLLM.git
cd MemoryLLM
```
Then simply use the following code to load the model:
```python
from modeling_memoryllm import MemoryLLM
from transformers import AutoTokenizer
# load chat model
model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b-chat", attn_implementation="flash_attention_2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b-chat")
model = model.cuda()
```

### How to use the model

Inject a piece of context into the model using the following script:

```python
model = model.cuda()

# Self-Update with the new context
ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."

# please make sure the context to inject into the memory is larger than 16 tokens, this is the hard minimum when training the model. The memory will be disturbed when less than 16 tokens are injected into the memory. 
model.inject_memory(tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.cuda(), update_memory=True)

# Generation
messages = [{
    'role': 'user', "content": "What fruits does David like?",
}]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
inputs = inputs[:, 1:] # remove bos token

outputs = model.generate(input_ids=inputs.cuda(),
                         max_new_tokens=20)
response = tokenizer.decode(outputs[0])

outputs = model.generate(inputs=input_ids.cuda(), attention_mask=attention_mask.cuda(), max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
```

