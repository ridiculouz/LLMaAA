import torch
import random
import numpy as np

### General utils
def ugly_log(file, info):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(info + '\n')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

### Collate functions
def ner_collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]    
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    flag = True
    for f in batch:
        if f['seq_label'] is None:          # strategy ensures that training label is not None
            flag = False
            break
    if flag:
        seq_label = [f['seq_label'] + [0] * (max_len - len(f['seq_label'])) for f in batch]
        seq_label = torch.tensor(seq_label, dtype=torch.long)
    else:
        seq_label = None
    output = {'input_ids': input_ids, 'attention_mask': attention_mask,
              'seq_label': seq_label}
    return output

def re_collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]    
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    # construct ht_pos
    ht_pos = [[f['h']['pos'][0], f['t']['pos'][0]] for f in batch]
    ht_pos = torch.tensor(ht_pos, dtype=torch.long)
    # construct labels
    flag = True
    for f in batch:
        if f['label_id'] is None:          # strategy ensures that training label is not None
            flag = False
            break
    if flag:
        labels = [f['label_id'] for f in batch]
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None
    output = {'input_ids': input_ids, 'attention_mask': attention_mask, 
              'ht_pos': ht_pos, 'labels': labels}
    return output