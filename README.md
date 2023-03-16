# ChatGLM-finetune-LoRA


This repository contains code for fintune [ChatGLM-6b](https://github.com/THUDM/ChatGLM-6B) using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

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