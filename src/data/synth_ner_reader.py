import os
import ujson as json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

MAX_LEN = 512

def convert_synth_data(sample, tagset):
    """
    Convert synthetic data format to standard format.
    """
    try:
        assert 'text' in sample and 'entities' in sample
        for e in sample['entities']:
            assert 'type' in e and 'name' in e
    except AssertionError:
        return None
    text = sample['text']
    entities = [{'span': entity['name'], 'type': entity['type']} for entity in sample['entities']]
    entities = [entity for entity in entities if (entity['type'] in tagset and          # ensure coverage of entity TYPE
                                                  isinstance(entity['span'], str) and   
                                                  len(entity['span']) > 0)]             # filter invalid entity
    entities = sorted(entities, key=lambda x: len(x['span']), reverse=True)
    span_to_type = {entity['span']: entity['type'] for entity in entities}
    for entity in entities:
        span = entity['span']
        text = ('[sep]' + span + '[sep]').join(text.split(span))
    words = text.split('[sep]')
    
    tokens, tags = [], []
    for word in words:
        word = word.strip()
        if len(word) == 0:
            continue
        if word in span_to_type:
            tokens.append(word)
            tags.append(f'S-{span_to_type[word]}')
        else:
            tokens.append(word)
            tags.append('O')
    outputs = {
        'text': sample['text'],
        'labels': sample['entities'],
        'id': None,
        'tokens': tokens,
        'tags': tags
    }
    return outputs

def synth_ner_reader(tokenizer: PreTrainedTokenizer, dataset: str,
                     setting: str):
    """
    Load synthesized data in the format of
    {"text": text, "entities": [{"span": span, "type": type}]}.
    ALSO load demo and test data from original dataset.
    """
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    synth_data_path = os.path.join(dir_path, f'synth_data/{dataset}/{setting}/data.jsonl')
    dir_path = os.path.join(dir_path, f'data/{dataset}')    # original data dir

    files = {key: os.path.join(dir_path, f'{key}.jsonl') for key in ['demo', 'test']}
    files['train'] = synth_data_path
    # load meta data - tag2id
    meta_path = os.path.join(dir_path, 'meta.json')
    meta_data = json.load(open(meta_path, 'r'))
    tag2id = meta_data['tag2id']
    tagset = meta_data['tagset']

    # load dataset
    dataset = {}
    for split, file in files.items():
        features = []
        fd = open(file, 'r', encoding='utf-8')
        for line in tqdm(fd, desc=split):
            sample = json.loads(line.strip())
            
            # synthesized data: conversion first
            if split == 'train':
                sample = convert_synth_data(sample, tagset)
                if sample is None:
                    continue

            input_ids, seq_label = [], []
            tokens, tags = sample['tokens'], sample['tags']

            for word, tag in zip(tokens, tags):
                word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                input_ids.extend(word)
                if len(word) == 1:
                    token_label = [tag2id[tag]]
                else:       # tokenized word length > 1
                    if tag.startswith('O') or tag.startswith('M'):
                        token_label = [tag2id[tag]] * len(word)
                    else:
                        prefix, classtype = tag.split('-')
                        if prefix == 'B':
                            token_label = [tag2id[tag]] + [tag2id[f'M-{classtype}']] * (len(word) - 1)
                        elif prefix == 'E':
                            token_label = [tag2id[f'M-{classtype}']] * (len(word) - 1) + [tag2id[tag]]
                        elif prefix == 'S':
                            token_label = ([tag2id[f'B-{classtype}']] 
                                           + [tag2id[f'M-{classtype}']] * (len(word) - 2) 
                                           + [tag2id[f'E-{classtype}']])
                        else:
                            raise ValueError()
                seq_label.extend(token_label)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            seq_label = [tag2id['O']] + seq_label + [tag2id['O']]
            assert len(input_ids) == len(seq_label)
            assert len(input_ids) <= MAX_LEN

            sample.pop('tokens')
            sample.pop('tags')
            sample['input_ids'] = input_ids
            sample['seq_label'] = seq_label
            features.append(sample)
        fd.close()
        dataset[split] = features            
    return dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = synth_ner_reader(tokenizer, 'en_conll03', 'zero')
    for f in dataset['train'][: 5]:
        print(f)
        print(tokenizer.convert_ids_to_tokens(f['input_ids']))