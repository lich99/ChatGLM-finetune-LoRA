# ChatGLM-finetune-LoRA

<img src=fig/wechat_qrcode.jpeg width=35% />

This repository contains code for finetuning [ChatGLM-6b](https://github.com/THUDM/ChatGLM-6B) using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

We also provide [finetuned weights](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/saved/chatglm-6b_alpaca_5.pt).

The minimum required GPU memory is **24G**, **RTX3090** is enough for training.

- 2022/4/12: Add tensorboard. Support finetune the entire model (Much faster convergence and usually has better performance)
- 2022/3/28: Optimized code structure, more simple and clear. Add training instruction.
- 2022/3/24: Support **Multi-GPU** training, **DeepSpeed**, Batch collate. Using accelerate to launch `train.py` 

### Inference

| 续写 | 对话 | x |
|---|---|---|
|周末适合哪里玩? 周末适合去北京| 如何缓解焦虑? | x |
|周末适合哪里玩? 周末适合去北京旅游。北京是中国的首都,有着悠久的历史和丰富的文化遗产。在这里你可以参观历史古迹、博物馆和文化遗址等景点。此外,北京的特色美食也不容错过,例如烤鸭和炸酱面等等。如果对历史文化感兴趣的话,建议去故宫、天安门广场、颐和园和长城等地游览;如果喜欢自然风光,可以去八达岭高速或者京承高速公路上自驾游,欣赏美丽的景色。| 焦虑是一种较为常见的情绪反应，当人们面临不确定性、压力或风险时，可能会感到焦虑。以下是一些缓解焦虑的方法：\n\n1. 深呼吸：深呼吸可以帮助你放松身体和心灵。尝试缓慢地吸气，然后缓慢地呼气，重复几次。\n\n2. 渐进性肌肉松弛：这是一种放松身体的技巧，通过逐步收缩和松弛肌肉来减轻身体的紧张感。你可以在背部、腿部和手臂等部位练习渐进性肌肉松弛。\n\n3. 冥想：冥想是一种可以帮助你放松心灵和减轻焦虑的技巧。你可以通过集中注意力、呼吸、放松身体和关注内心的练习来冥想。\n\n4. 运动：运动可以帮助你释放紧张感和压力，同时也可以提高身体和心理的健康状况。你可以尝试跑步、瑜伽、游泳等运动。\n\n5. 寻求支持：与家人、朋友或专业人士谈论你的问题可以帮助你减轻焦虑。你可以寻求心理咨询或与支持团体联系。\n\n6. 改善生活方式：保持健康的饮食、充足的睡眠和规律的锻炼可以帮助你改善身体和心理的健康状况。\n\n请注意，如果你的焦虑症状持续很长时间或影响到你的日常生活，请寻求专业医疗帮助。| x |

### Easy to use

```python
import loralib as lora
import lora_utils.insert_lora
import dataset.GLM as GLM_Data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

device = 'cuda'
checkpoint = "THUDM/chatglm-6b"


# load model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

# get LoRA model
lora_config = {
        'r': 32,
        'lora_alpha':32,
        'lora_dropout':0.1,
        'enable_lora':[True, True, True],
    }
model = lora_utils.insert_lora.get_lora_model(model, lora_config)
### trainable_params:22020096 (0.35%), non_trainable_params:6255206400

# get Dataloader
pairs = [{'prompt':'Hello!', 'completion':'Hi! This is ChatGLM.'}]
pairs_encoded = GLM_Data.encode_pairs(pairs, tokenizer)
train_dataset = GLM_Data.GLMDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = GLM_Data.collate_fn, shuffle=True, batch_size=1)

# training
model.half().to(device)
batch = {k: v.to(device) for k, v in next(iter(train_dataloader)).items()}
outputs = model(**batch)
outputs.loss.backward()
```

### Training

Using [accelerate CLI tool](https://huggingface.co/docs/accelerate/basic_tutorials/launch) to launch multiprocess / distributed training:
```
accelerate launch --config_file config/default_config.yaml train.py
```
If you want to finetune the entire model, using
```
accelerate launch --config_file config/default_config.yaml train_full.py
```

Don't forget to change `num_processes` to **the number of GPUs** you want to use.

Now `accelerate` supports **ZeRO 2** (with offload), **ZeRO 3** (with offload)

**Try ZeRO 2 and no offload first, unless you encounter OOM.**

ZeRO 2 (no offload) > ZeRO 2 (offload) > ZeRO 3 (no offload) > ZeRO 3 (offload)

Like OpenAI's finetune API, the data should be in following structure:  
```python
[
    {'prompt': <enter the prompt here (can be instruction)>, 'completion': <the expectation completion>},
    {'prompt': <enter the prompt here (can be instruction)>, 'completion': <the expectation completion>},
    ...,
    {'prompt': <enter the prompt here (can be instruction)>, 'completion': <the expectation completion>},
]
```
It is a **list** of **prompt-completion pairs**.



### Stanford Alpaca's Dataset

Here we use the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)'s Dataset as an example for fine-tuning. We also provide [finetuned weights](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/saved/chatglm-6b_alpaca_5.pt).

example line: 

`{'prompt': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nClassify the movie genres from the given context.\n\n### Input:\nThis movie tells the story of two brothers who were both born with magical powers.\n\n### Response:',
 'completion': 'Fantasy'}`


Training for Stanford Alpaca's Dataset should within **30min** per epoch on **4*V100**

You may observe a typical training loss curve: 
![example_training_loss](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/fig/example_loss.png)
Note: varies with different datasets

### LoRA
```python
lora_config = {
        'r': 32,
        'lora_alpha':32,
        'lora_dropout':0.1,
        'enable_lora':[True, True, True],
    }
```
Using above LoRA config, we have `trainable_params:22020096 (0.35%), non_trainable_params:6255206400`

### Save & Load
```python
torch.save(lora.lora_state_dict(model), 'path to file you saved')
model.load_state_dict(torch.load('path to file you saved'), strict=False)
```
