# This file implements a train script for ner/re tasks.
# If `-` in args.dataset, use synth data as train data,
# else use **demo** data as train data
import argparse
import os
import copy
import ujson as json
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data.processor import Processor
from .model.ner import BertSoftmaxForNer
from .model.re import BertMarkerForRe
from .train_ner import train_ner
from .train_re import train_re
from .utils import ugly_log, set_seed

def get_opt():
    parser = argparse.ArgumentParser()
    # data related 
    parser.add_argument('--dataset', default='zh_onto4', type=str)  # use `en_conll03-zero` to denote synth data
    parser.add_argument('--data_size', default=-1, type=int)        # -1 to use full data; else use limited train data
    # file related
    parser.add_argument('--save_path', default='', type=str)    # see train_ner function
    parser.add_argument('--load_path', default='', type=str)    # haven't implemented yet!
    # model related
    parser.add_argument('--model_name_or_path', default='hfl/chinese-bert-wwm-ext', type=str)
    # optimization related
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='The initial learning rate for bert layer.')
    parser.add_argument('--adam_epsilon', default=1e-6, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--warmup_ratio', default=0.06, type=float,
                        help='Warm up ratio for Adam.')
    # training steps; use min between the two
    parser.add_argument('--num_train_epochs', default=40, type=int)  # train epoch during each loop
    parser.add_argument('--max_train_steps', default=-1, type=int)
    parser.add_argument('--early_stopping_patience', default=-1, type=int)
    # misc
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=42, type=int)
    # dummy
    parser.add_argument('--reweight', action='store_true', default=False)

    return parser.parse_args()

def main(args):
    # get log
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(path)
    if not os.path.exists(os.path.join(path, f'logs/{args.dataset}')):
        os.mkdir(os.path.join(path, f'logs/{args.dataset}'))
    path = os.path.join(path, f'logs/{args.dataset}/apple-{args.data_size}-{args.seed}.log')
    args.log_file = path
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    # get task type
    if '-' in args.dataset:
        dataset, setting = args.dataset.split('-')
        synth = True
    else:
        dataset = args.dataset
        synth = False
    if dataset in ['en_conll03', 'zh_msra', 'zh_onto4']:
        task_type = 'ner'
    elif dataset in ['en_retacred']:
        task_type = 're'
    else:
        raise ValueError('Unknown task type.')
    # get config, tokenizer & data processor
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # add special tokens if task_type == 're'
    if task_type == 're':
        sptokens = {'additional_special_tokens': ['[E]', '[\E]']}
        tokenizer.add_special_tokens(sptokens)
    data_processor = Processor(dataset=args.dataset, tokenizer=tokenizer, cache_name=None)
    if task_type == 'ner' or task_type == 're':
        args.id2tag = data_processor.get_id2tag()
        config.num_labels = len(args.id2tag)
    # add config from args
    config.model_name_or_path = args.model_name_or_path
    # get data
    if synth:
        train_features = data_processor.get_features('train')
        dev_features = data_processor.get_features('demo')
        test_features = data_processor.get_features('test')
        if args.data_size != -1:
            train_features = train_features[:args.data_size]
    else:
        train_features = data_processor.get_features('demo')
        dev_features = data_processor.get_features('test')
        test_features = None
    # get model
    if task_type == 'ner':
        model = BertSoftmaxForNer(config, reduction='none')
        train = train_ner
    elif task_type == 're':
        model = BertMarkerForRe(config, reduction='none')
        model.resize_token_embeddings(len(tokenizer))
        train = train_re
    else:
        raise ValueError('tbd.')
    model.to(device)
    best_ckpt = train(args, model, train_features, dev_features, test_features)


if __name__ == '__main__':
    args = get_opt()
    set_seed(args.seed)

    main(args)