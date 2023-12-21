from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import ner_collate_fn, re_collate_fn
from ..model.ner import BertSoftmaxForNer
from ..model.re import BertMarkerForRe

### Predict functions
def ner_predict(args, features, model: BertSoftmaxForNer):
    """
    Return a list of tensor logits of ner prediction on cpu.
    """
    dataloader = DataLoader(features, args.test_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    model.eval()
    pred_logits = []
    for batch in tqdm(dataloader, desc='Evaluating on pool data'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device)}    
        with torch.no_grad():
            outputs = model.forward(**inputs)
            logits = outputs[0]
            attention_mask = batch['attention_mask'].to(args.device)
            for i in range(logits.size(0)):
                pred = logits[i][attention_mask[i] == 1].cpu()
                pred_logits.append(pred)
    return pred_logits

def re_predict(args, features, model: BertMarkerForRe):
    """
    Return a list of tensor logits of re prediction.
    """
    dataloader = DataLoader(features, args.test_batch_size, shuffle=False, collate_fn=re_collate_fn)
    model.eval()
    pred_logits = []
    for batch in tqdm(dataloader, desc='Evaluating on pool data'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device),
                  'ht_pos': batch['ht_pos'].to(args.device),}    
        with torch.no_grad():
            outputs = model.forward(**inputs)
            logits = outputs[0]
            pred_logits.append(logits)
    pred_logits = torch.cat(pred_logits, dim=0)
    return pred_logits

def get_bert_embeddings(args, features, model: nn.Module, normalize: bool=True):
    """
    embed features with bert encoder.
    """
    model.eval()
    embeddings = []
    if isinstance(model, BertSoftmaxForNer):
        dataloader = DataLoader(features, args.test_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    elif isinstance(model, BertMarkerForRe):
        dataloader = DataLoader(features, args.test_batch_size, shuffle=False, collate_fn=re_collate_fn)
    else:
        raise ValueError('Unknown model.')
    for batch in tqdm(dataloader, desc='Computing bert embeddings'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device)}    
        with torch.no_grad():
            outputs = model.bert(**inputs)[0]   # [batch_size, seq_length, n_dim]
            attention_mask = inputs['attention_mask']
            outputs[attention_mask == 0] = 0
            # mean pooling over sequence outputs
            outputs = outputs.sum(dim=1) / attention_mask.sum(dim=-1).unsqueeze(-1)
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)    
            embeddings.append(outputs)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings