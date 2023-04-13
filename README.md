# ChatGLM-finetune-LoRA


This repository contains code for fintune [ChatGLM-6b](https://github.com/THUDM/ChatGLM-6B) using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

We also provide a [finetuned weight](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/saved/chatglm-6b_alpaca_5.pt).

The minimum required GPU memory is **24G**, **RTX3090** is enough for training.

- 2022/4/12: Add tensorboard.
- 2022/3/28: Optimized code structure, more simple and clear. Add training instruction.
- 2022/3/24: Support **Multi-GPU** training, **DeepSpeed**, Batch collate. Using accelerate to launch `train.py` 


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
Don't forget to change `num_processes` to the number of GPUs you want to use.

Now `accelerate` supports ZeRO 2 (with offload), ZeRO 3 (with offload)

Try ZeRO2 and no offload first, unless you encounter OOM.

ZeRO 2 (no offload) > ZeRO 2 (offload) > ZeRO 3 (no offload) > ZeRO 3 (offload)

Likes OpenAI's fintune API, the data should be in following structure:  
```python
[
    {'prompt': <enter the prompt here (can be instrcution)>, 'completion': <the expectation completion>},
    {'prompt': <enter the prompt here (can be instrcution)>, 'completion': <the expectation completion>},
    ...,
    {'prompt': <enter the prompt here (can be instrcution)>, 'completion': <the expectation completion>},
]
```
It is a **list** of **prompt-completion pairs**.



### Stanford Alpaca's Dataset

Here we use the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)'s Dataset as an example for fine-tuning. We also provide a [finetuned weight](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/saved/chatglm-6b_alpaca_5.pt).

example line: 

`{'prompt': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nClassify the movie genres from the given context.\n\n### Input:\nThis movie tells the story of two brothers who were both born with magical powers.\n\n### Response:',
 'completion': 'Fantasy'}`


Training for Stanford Alpaca's Dataset should within **30min** per epoch on **4*V100**

You may observe a typical training loss curve: 
![example_training_loss](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/fig/example_loss.png)
Note: vary with different dataset

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