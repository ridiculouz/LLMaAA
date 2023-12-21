import os
import ujson as json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

MAX_LEN = 512

def match(pattern, target):
    """
    Pattern matching. Pattern is a string, target is a list of strings (token list).
    Return the [start, end] of pattern's tokens.
    """
    for i in range(len(target)):
        for j in range(i + 1, len(target) + 1):
            string = ''.join(target[i: j])
            if string == pattern:
                return [i, j]
    return [-1, -1]

def convert_synth_data(sample, tag2id):
    try:
        assert (('text' in sample) and ('subject' in sample) and
                ('object' in sample) and ('relation' in sample))
        assert isinstance(sample['subject'], str) and isinstance(sample['object'], str)
        assert ((sample['subject'] in sample['text']) and
                (sample['object'] in sample['text']))
        assert len(sample['subject']) > 0 and len(sample['object']) > 0
    except AssertionError:
        return None
    relation = sample['relation']
    if relation not in tag2id:
        relation = 'no_relation'    # NA case
    text, subj, obj = sample['text'], sample['subject'], sample['object']
    if len(subj) > len(obj):
        entities = [subj, obj]
    else:
        entities = [obj, subj]
    for entity in entities:
        text = ('[sep]' + entity + '[sep]').join(text.split(entity))
    words = text.split('[sep]')

    tokens = []
    for word in words:
        if len(word) == 0:
            continue
        tokens.append(word)
    subj_pos = match(subj, tokens)
    obj_pos = match(obj, tokens)
    h = {'pos': subj_pos, 'name': subj}
    t = {'pos': obj_pos, 'name': obj}
    outputs = {
        'text': sample['text'],
        'tokens': tokens,
        'h': h,
        't': t,
        'label': relation,
        'id': None
    }
    # print(outputs)
    return outputs

def synth_re_reader(tokenizer: PreTrainedTokenizer, dataset: str,
                    setting: str):
    """
    If use_cache is True, training labels are loaded from `cache.json`,
    otherwise from `train.jsonl`.

    Assume that tokenizer has added [E], [\E] to vocab.
    
    Sample format:
    {'input_ids', 'label_id', 'id', 'text', 'label' (text), 'h', 't'}
    'pos' in 'h'/'t': [left, right)
    """
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    synth_data_path = os.path.join(dir_path, f'synth_data/{dataset}/{setting}/data.jsonl')
    dir_path = os.path.join(dir_path, f'data/{dataset}')

    files = {key: os.path.join(dir_path, f'{key}.jsonl') for key in ['demo', 'test']}
    files['train'] = synth_data_path
    # load meta data - tag2id
    meta_path = os.path.join(dir_path, 'meta.json')
    tag2id = json.load(open(meta_path, 'r'))['tag2id']

    # load dataset
    dataset = {}
    for split, file in files.items():
        features = []
        fd = open(file, 'r', encoding='utf-8')
        for line in tqdm(fd, desc=split):
            sample = json.loads(line.strip())

            # synthesized data: conversion first
            if split == 'train':
                sample = convert_synth_data(sample, tag2id)
                if sample is None:
                    continue

            input_ids = []
            tokens, h, t = sample['tokens'], sample['h'], sample['t']
            # add [E], [\E]
            h_pos, t_pos = [-1, -1], [-1, -1]

            for i, token in enumerate(tokens + ['[EOF]']):
                if i == h['pos'][0]:
                    word = tokenizer.convert_tokens_to_ids(['[E]'])
                    assert len(word) == 1
                    input_ids.extend(word)
                    h_pos[0] = len(input_ids)
                if i == t['pos'][0]:
                    word = tokenizer.convert_tokens_to_ids(['[E]'])
                    assert len(word) == 1
                    input_ids.extend(word)
                    t_pos[0] = len(input_ids)
                if i == h['pos'][1]:
                    word = tokenizer.convert_tokens_to_ids(['[\E]'])
                    assert len(word) == 1
                    input_ids.extend(word)
                    h_pos[1] = len(input_ids) + 1
                if i == t['pos'][1]:
                    word = tokenizer.convert_tokens_to_ids(['[\E]'])
                    assert len(word) == 1
                    input_ids.extend(word)
                    t_pos[1] = len(input_ids) + 1
                if i == len(tokens):
                    break
                word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                input_ids.extend(word)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            assert len(input_ids) <= MAX_LEN
            assert (-1 not in h_pos) and (-1 not in t_pos)

            label_id = tag2id[sample['label']]

            sample['input_ids'] = input_ids
            sample['label_id'] = label_id
            sample['h']['pos'] = h_pos
            sample['t']['pos'] = t_pos
            features.append(sample)
        fd.close()
        dataset[split] = features
    return dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    sptokens = {'additional_special_tokens': ['[E]', '[\E]']}
    tokenizer.add_special_tokens(sptokens)
    dataset = synth_re_reader(tokenizer, 'en_retacred', 'zero')
    print(len(dataset['train']))
    for f in dataset['train'][-10:]:
        print(f)
        print(tokenizer.convert_ids_to_tokens(f['input_ids']))