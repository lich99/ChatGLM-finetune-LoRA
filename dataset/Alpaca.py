import json


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def load(path):

    with open(path, 'r') as f:
        content = json.load(f)

    pairs = []

    for line in content:
        if line['input'] == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(line)
        completion = line['output']
        pairs.append({'prompt':prompt, 'completion':completion})
    
    return pairs