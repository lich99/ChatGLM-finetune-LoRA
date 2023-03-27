import numpy as np
import loralib as lora

def get_lora_model(model, lora_config):
    
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