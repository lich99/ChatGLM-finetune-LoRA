import numpy as np
import loralib as lora

def get_lora_model_loralib(model, lora_config):
    
    # lora_config = {
    #     'r': 32,
    #     'lora_alpha':32,
    #     'lora_dropout':0.1,
    #     'enable_lora':[True, True, True],
    # }

    for key, module in model.named_modules():
        if key.endswith('attention'):
            if isinstance(module.query_key_value, lora.MergedLinear):
                module.query_key_value.r = lora_config['r']
                module.query_key_value.lora_alpha = lora_config['lora_alpha']
                module.query_key_value.lora_dropout.p = lora_config['lora_dropout']
            else:
                qkv_proj = lora.MergedLinear(module.query_key_value.in_features,
                                            3*module.query_key_value.in_features, 
                                            **lora_config,
                                            dtype=module.query_key_value.weight.data.dtype)
                qkv_proj.load_state_dict(module.query_key_value.state_dict(), strict=False)
                module.query_key_value = qkv_proj


    lora.mark_only_lora_as_trainable(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    print('trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(trainable_params, trainable_params/non_trainable_params*100,non_trainable_params))

    return model



import peft
import torch
from peft import LoraConfig
from itertools import compress


class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data

        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data
    
    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)



def get_lora_model(model, lora_config):
    
    target_modules = list(compress(['q', 'k', 'v'], lora_config['enable_lora']))

    config = LoraConfig(
              peft_type="LORA", 
              task_type="SEQ_2_SEQ_LM", 
              r=lora_config['r'], 
              lora_alpha=lora_config['lora_alpha'], 
              target_modules=["q", "v"],
              lora_dropout=lora_config['lora_dropout'])


    for key, module in model.named_modules():
        if key.endswith('attention'):
            layer = int(key.split('.')[2])
            try:
                qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features)
                qkv_layer.update(module.query_key_value)
                module.query_key_value = qkv_layer
                module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)
            except:
                print('[Warn] Passed')
                pass
        

    lora.mark_only_lora_as_trainable(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    print('trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(trainable_params, trainable_params/non_trainable_params*100,non_trainable_params))

    return model