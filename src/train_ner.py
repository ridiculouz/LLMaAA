import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import higher

from .evaluation import MetricForNer
from .model.ner import BertSoftmaxForNer
from .utils import ugly_log, ner_collate_fn

def train_ner(args, model: BertSoftmaxForNer, train_features, dev_features, test_features):
    def cycle(iterable):
        """
        an iterator with no stop exception.
        """
        while True:
            for x in iterable:
                yield x

    def finetune(model: BertSoftmaxForNer, optimizer: AdamW, num_epochs: int, max_steps: int=-1,
                 num_steps: int=0, best_score: float=-1.0,      # support continuous training
                 meta_optimizer: torch.optim.Optimizer=None):     
        # add meta dataloader
        if args.reweight:
            # assume that dev has gold labels?
            meta_dataloader = DataLoader(dev_features, batch_size=args.train_batch_size, 
            # meta_dataloader = DataLoader(dev_features, batch_size=8,
                                         shuffle=True, collate_fn=ner_collate_fn)
            meta_iterator = cycle(meta_dataloader)
            assert meta_optimizer is not None

        dataloader = DataLoader(train_features, batch_size=args.train_batch_size,
                                shuffle=True, collate_fn=ner_collate_fn)
        total_steps = (len(dataloader) // args.gradient_accumulation_steps) * num_epochs
        if max_steps != -1:
            total_steps = min(total_steps, max_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print('Total steps: {}'.format(total_steps))
        print('Warmup steps: {}'.format(warmup_steps))

        # for early stopping
        min_val_loss = torch.tensor(float('inf')).to(args.device)
        descent = 0
        best_ckpt = copy.deepcopy(model)
        for epoch in range(num_epochs):
            model.zero_grad()
            train_loss = torch.tensor(0.0).to(args.device)
            accumulated_loss = torch.tensor(0.0).to(args.device)
            for step, batch in tqdm(enumerate(dataloader), desc='Train epoch {}'.format(epoch)):
                model.train() 
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['seq_label'].to(args.device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 
                          'labels':labels,}
                if args.reweight:
                    with higher.innerloop_ctx(model, meta_optimizer) as (fmodel, diffopt):
                        # 1. Update meta model on training data
                        outputs = fmodel.forward(**inputs)
                        meta_train_loss = outputs[0]     # first turn it into [batch size]
                        meta_train_loss = meta_train_loss * attention_mask
                        # try mean over sample first. not sum.
                        meta_train_loss = meta_train_loss.sum(dim=-1) / attention_mask.sum(dim=-1)
                        eps = torch.zeros(meta_train_loss.size(), requires_grad=True).to(args.device)
                        meta_train_loss = torch.sum(eps * meta_train_loss)
                        diffopt.step(meta_train_loss)

                        # 2. Compute grads of eps on meta validation data
                        meta_batch = next(meta_iterator)
                        input_ids = meta_batch['input_ids'].to(args.device)
                        attention_mask = meta_batch['attention_mask'].to(args.device)
                        labels = meta_batch['seq_label'].to(args.device)
                        meta_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 
                                       'labels':labels,}
                        outputs = fmodel.forward(**meta_inputs)
                        # mean over token
                        meta_val_loss = outputs[0]
                        meta_val_loss = meta_val_loss[attention_mask == 1].mean()
                        eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

                    # 3. Compute weights for current training batch
                    w_tilde = torch.clamp(-eps_grads, min=0)
                    l1_norm = torch.sum(w_tilde)
                    if l1_norm != 0:
                        w = w_tilde / l1_norm
                    else:
                        w = w_tilde

                    # 4. Train model on weighted batch
                    attention_mask = batch['attention_mask'].to(args.device)
                    outputs = model.forward(**inputs)
                    loss = outputs[0]         # first turn it into [batch size]
                    loss = loss * attention_mask
                    # try mean over sample first. not sum.
                    loss = loss.sum(dim=-1) / attention_mask.sum(dim=-1)
                    
                    train_loss += torch.mean(loss)
                    accumulated_loss += torch.mean(loss)
                    
                    loss = torch.sum(w * loss)
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                else:
                    outputs = model.forward(**inputs)
                    loss = outputs[0]                     # TODO: reduction = none !!!!!
                    loss = loss[attention_mask == 1].mean()
                    loss = loss / args.gradient_accumulation_steps
                    train_loss += loss
                    accumulated_loss += loss
                    loss.backward()
                # update gradient step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    # wandb.log({'loss': accumlated_loss.item()}, step=num_steps)
                    accumulated_loss = torch.tensor(0.0).to(args.device)
                # evaluate on dev & test features
                if step == len(dataloader) - 1 or num_steps == max_steps:
                    val_loss, dev_score, dev_outputs = evaluate_ner(args, model, dev_features, tag='dev')
                    dev_p = dev_outputs['dev precision']
                    dev_r = dev_outputs['dev recall']
                    if test_features is not None:
                        test_score, test_outputs = evaluate_ner(args, model, test_features, tag='test')
                        test_p = test_outputs['test precision']
                        test_r = test_outputs['test recall']
                        dev_outputs.update(test_outputs)
                    else:
                        test_score, test_p, test_r = -1, -1, -1
                    # wandb.log(dev_outputs, step=num_steps)
                    dev_score = round(dev_score * 100, 4)
                    test_score = round(test_score * 100, 4)
                    dev_p = round(dev_p * 100, 4)
                    dev_r = round(dev_r * 100, 4)
                    test_p = round(test_p * 100, 4)
                    test_r = round(test_r * 100, 4)
                    train_loss = round(train_loss.item(), 6)
                    val_loss = round(val_loss.item(), 6)
                    print('Accumulated train loss: {} | Accumulated val loss: {}'.format(train_loss, val_loss))
                    msg = 'dev f1: {} | test f1: {} | test p: {} | test r: {} | dev p: {} | dev r: {} | train loss: {} | val loss: {}'.format(dev_score, test_score, test_p, test_r, dev_p, dev_r, train_loss, val_loss)
                    ugly_log(args.log_file, msg)
                    print(dev_outputs)
                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != '':
                            torch.save(model.state_dict(), args.save_path)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        descent = 0
                        best_ckpt = copy.deepcopy(model)
                    else:
                        descent += 1
                # early return
                if descent == args.early_stopping_patience:
                    return num_steps, best_ckpt
                if num_steps == max_steps:
                    return num_steps, best_ckpt
        # best checkpoint: with minimal validation loss
        return num_steps, best_ckpt
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], },
        {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    meta_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, 
                                      eps=args.adam_epsilon)
    model.zero_grad()
    _, best_ckpt = finetune(model, optimizer, args.num_train_epochs, args.max_train_steps,
                            meta_optimizer=meta_optimizer)
    return best_ckpt

def evaluate_ner(args, model: BertSoftmaxForNer, features, tag=''):
    return_loss = 'dev' in tag
    metric = MetricForNer(args.id2tag)
    dataloader = DataLoader(features, batch_size=args.test_batch_size,
                            shuffle=False, collate_fn=ner_collate_fn)
    model.eval()
    accumlated_loss = torch.tensor(0.0).to(args.device)
    for step, batch in tqdm(enumerate(dataloader), desc='Evaluating on {} data'.format(tag)):
        attention_mask = batch['attention_mask'].to(args.device)
        if return_loss:
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': attention_mask,
                      'labels': batch['seq_label'].to(args.device)}
        else:
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': attention_mask}
        pred_seqs = []
        gold_seqs = []
        with torch.no_grad():
            outputs = model.forward(**inputs)
            if return_loss:
                logits = outputs[1]
                loss = outputs[0]
                if model.reduction == 'none':
                    loss = loss[attention_mask == 1].mean()
                accumlated_loss += loss
            else:
                logits = outputs[0]
            pred_labels = torch.argmax(logits, dim=-1)
            gold_labels = batch['seq_label'].to(args.device)
            for i in range(pred_labels.size(0)):
                pred = pred_labels[i][attention_mask[i] == 1].cpu().numpy().tolist()
                gold = gold_labels[i][attention_mask[i] == 1].cpu().numpy().tolist()
                pred_seqs.append(pred)
                gold_seqs.append(gold)
        metric.update(gold_seqs, pred_seqs)
    f1, overall, class_info = metric.stats()
    overall = {' '.join([tag, key]): val for key, val in overall.items()}
    if return_loss:
        return accumlated_loss, f1, overall
    return f1, overall