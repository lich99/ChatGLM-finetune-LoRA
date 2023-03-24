# ChatGLM-finetune-LoRA


This repository contains code for fintune [ChatGLM-6b](https://github.com/THUDM/ChatGLM-6B) using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

We also provide a [finetuned weight](https://github.com/lich99/ChatGLM-finetune-LoRA/blob/main/saved/chatglm-6b_alpaca_5.pt).


- 2022/3/24: Support **Multi-GPU** training, **DeepSpeed**, Batch collate. Using accelerate to launch `train.py` 



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
config = LoraConfig(
              peft_type="LORA", 
              task_type="SEQ_2_SEQ_LM", 
              r=8, 
              lora_alpha=8, 
              target_modules=["q", "v"],
              lora_dropout=0.1, 
              )
```
Using above LoRA config, we have `trainable_params:3670016 (0.06%), non_trainable_params:6255206400`

### Save & Load
```python
torch.save(lora.lora_state_dict(model), 'path to file you saved')
model.load_state_dict(torch.load('path to file you saved'), strict=False)
```