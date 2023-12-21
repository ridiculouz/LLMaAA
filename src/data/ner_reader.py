import os
import ujson as json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

MAX_LEN = 512

def convert_span_labels_to_sequence_labels(tokens, span_label, language):
    """
    Covert span labels to sequence labels.
    Language: en/zh
    """
    span_label = sorted(span_label, key=lambda x: len(x['span']), reverse=True)
    span_to_type = {entity['span']: entity['type'] for entity in span_label}
    # get words list
    if language == 'zh':
        text = ''.join(tokens)
        for entity in span_label:
            span = entity['span']
            text = ('[sep]' + span + '[sep]').join(text.split(span))
        words = text.split('[sep]')
    else:
        # build a tokenizer first
        dictionary = dict()
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = f'[{len(dictionary)}]'
        id_string = ' '.join([dictionary[token] for token in tokens])
        for entity in span_label:
            span_tokens = entity['span'].strip().split(' ')
            # validate span token
            valid_flag = True
            for token in span_tokens:
                if token not in dictionary:
                    valid_flag = False
                    break
            if not valid_flag:
                continue
            # translate span token into ids
            id_substring = ' '.join([dictionary[token] for token in span_tokens])
            id_string = ('[sep]' + id_substring + '[sep]').join(id_string.split(id_substring))
            # print(id_string)
        # convert back to nl
        sent = id_string
        for token in dictionary:
            sent = sent.replace(dictionary[token], token)
        words = sent.split('[sep]')

    seq_label = []
    for word in words:
        word = word.strip()
        if len(word) == 0:
            continue
        entity_flag = (word in span_to_type)
        if language == 'en':
            word_length = len(word.split(' '))
        else:
            word_length = len(word)
        if entity_flag:
            if word_length == 1:
                label = [f'S-{span_to_type[word]}']
            else:
                label = ([f'B-{span_to_type[word]}'] + [f'M-{span_to_type[word]}'] * (word_length - 2)
                            + [f'E-{span_to_type[word]}'])
        else:
            label = ['O' for _ in range(word_length)]
        seq_label.extend(label)

    assert len(seq_label) == len(tokens)
    return seq_label        


def ner_reader(tokenizer: PreTrainedTokenizer, dataset: str, 
               cache_name: str= '', use_cache=True):
    """
    If use_cache is True, training labels are loaded from `cache.json`,
    otherwise from `train.jsonl`.
    
    Sample format:
    {'input_ids', 'seq_label', 'id', 'text', 'labels'}
    """
    # language
    language = dataset[:2]
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    dir_path = os.path.join(dir_path, f'data/{dataset}')
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
            input_ids, seq_label = [], []
            tokens, tags = sample['tokens'], sample['tags']

            if split == 'train' and use_cache:
                span_label = None
                if sample['id'] in cache:
                    span_label = cache[sample['id']]
                if span_label is None:
                    span_label = []                
                tags = convert_span_labels_to_sequence_labels(tokens, span_label, language)
            # assume that tags is not None
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

            if split == 'train' and use_cache and sample['id'] not in cache:
                seq_label = None

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
    dataset = ner_reader(tokenizer, 'en_conll03', use_cache=True)
    f = dataset['train'][0]
    print(f)
    print(tokenizer.convert_ids_to_tokens(f['input_ids']))