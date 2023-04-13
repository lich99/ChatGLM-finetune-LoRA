import torch
from transformers import AutoConfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision = 'main')

pad_to = 1
device = 'cuda'
eos_id = config.eos_token_id
pad_token_id = config.pad_token_id

class SimpleDataset(Dataset):
    def __init__(self, pairs) -> None:
        super().__init__()
        self.pairs = pairs
 
    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch):
    global device
    input_ids = []
    labels = []
    position_ids = []

    _max_length = max([len(obj['prompt'])+len(obj['completion']) for obj in batch])
    _max_length = (_max_length // pad_to + (_max_length % pad_to > 0) ) * pad_to

    attention_mask = torch.ones((len(batch), _max_length, _max_length), device=device)
    attention_mask.tril_()

    for i, obj in enumerate(batch):
        context_length = obj['prompt'].index(130004)
        attention_mask[i, :, :context_length] = 1

        to_pad = _max_length - len(obj['prompt']) - len(obj['completion'])

        input_ids.append(obj['prompt'] + obj['completion'] + [pad_token_id] * to_pad)

        position_ids.append(torch.stack([torch.arange(0, _max_length, device=device), 
                                         torch.concat([torch.zeros(context_length - 1, device=device), 
                                                       torch.arange(0, _max_length - context_length + 1, device=device)])]).long())

        labels.append(torch.tensor([-100] * len(obj['prompt']) + 
                                   obj['completion'] +
                                   [-100] * to_pad, device=device).long())

    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': attention_mask, 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}


def encode_pairs(pairs, tokenizer, with_eos = True):
    prompt_ids = tokenizer.batch_encode_plus([pair['prompt'] for pair in pairs])['input_ids']
    completion_ids = tokenizer.batch_encode_plus([pair['completion'] for pair in pairs], add_special_tokens=False)['input_ids']
    if with_eos:
        pairs_encoded = [{'prompt':prompt_ids[i], 'completion':completion_ids[i] + [eos_id]} for i in range(len(pairs))]
    else:
        pairs_encoded = [{'prompt':prompt_ids[i], 'completion':completion_ids[i]} for i in range(len(pairs))]
    return pairs_encoded