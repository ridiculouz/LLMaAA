import os
import ujson as json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

MAX_LEN = 512

def re_reader(tokenizer: PreTrainedTokenizer, dataset: str, 
               cache_name: str= '', use_cache=True):
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
    dir_path = os.path.join(dir_path, f'data/{dataset}')
    # files = {key: os.path.join(dir_path, f'{key}.jsonl') for key in ['train', 'dev', 'demo', 'test']}
    files = {key: os.path.join(dir_path, f'{key}.jsonl') for key in ['train', 'demo', 'test']}
    if use_cache:
        cache_file = os.path.join(dir_path, f'{cache_name}.json')
        cache = json.load(open(cache_file, 'r', encoding='utf-8'))
    # load meta data - tag2id
    meta_path = os.path.join(dir_path, 'meta.json')
    tag2id = json.load(open(meta_path, 'r'))['tag2id']

    dataset = {}
    for split, file in files.items():
        features = []
        fd = open(file, 'r', encoding='utf-8')
        for line in tqdm(fd, desc=split):
            sample = json.loads(line.strip())
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

            if split == 'train' and use_cache:
                if sample['id'] not in cache:
                    sample['label'] = None
                else:
                    sample['label'] = cache[sample['id']]
            if sample['label'] not in tag2id:
                sample['label'] = None
            if sample['label'] is None: 
                label_id = None
            else:
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
    dataset = re_reader(tokenizer, 'en_semeval10', use_cache=False)
    f = dataset['train'][0]
    print(f)
    print(tokenizer.convert_ids_to_tokens(f['input_ids']))