import os
import ujson as json
import random

### Utility function
def decode_tag_sequence(tag_seq):
    """
    Decode tag sequence into entity spans.
    @return: [type, start, end], where [start: end] is the span.
    """
    chunks = []
    chunk = ['', -1, -1]
    tag_seq.append('O')
    for index, tag in enumerate(tag_seq):
        # check for valid entity before each visit
        if chunk[2] != -1:
            chunk[2] += 1
            chunks.append(tuple(chunk))
            chunk = ['', -1, -1]
        if tag.startswith('S-'):
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('B-'):
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('M-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type != chunk[0]:
                chunk[1] = -1
        elif tag.startswith('E-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type != chunk[0]:
                chunk[1] = -1
            else:
                chunk[2] = index
        else:
            # tolerate for missing [E]
            if chunk[1] != -1:
                chunk[2] = index
                if chunk[2] >= chunk[1]:
                    chunks.append(tuple(chunk))
                chunk = ['', -1, -1]
    # remove 'O' that added before
    tag_seq.pop()     
    return chunks

### Validation functions
def valid_label(dataset='en_conll03'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, dataset)
    for _, _, files in os.walk(dir_path):
        for file in files:
            if not file.endswith('jsonl'):
                continue
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    for entity in sample['labels']:
                        assert entity['span'] in sample['text']
            print(f'file {file_path} passes label check.')

def valid_length(dataset='en_conll03'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, dataset)
    for _, _, files in os.walk(dir_path):
        for file in files:
            if not file.endswith('bmes'):
                continue
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                upper, cur = 0, 0
                for i, line in enumerate(f):
                    if line == '\n':
                        upper = max(upper, cur)
                        cur = 0
                    else:
                        cur += 1
            print(f'max length is {upper} for file {file_path}.')

# For OpenAI API, need natural language input & json format label.
# For general NER model, need to tokenize to corresponding tokens.
def prepro(dataset='en_conll03'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, dataset)
    language = dataset[:2]

    id = 0
    for split in ['train', 'dev', 'test']:
        grain = 'char' if language == 'zh' else 'word'
        src_path = os.path.join(dir_path, f'{split}.{grain}.bmes')
        dst_path = os.path.join(dir_path, f'{split}.jsonl')
        fr = open(src_path, 'r', encoding='utf-8')
        fw = open(dst_path, 'w', encoding='utf-8')
        
        tokens, tags = [], []
        for line in fr:
            if line == '\n':
                if len(tokens) == 0:
                    continue
                # text
                if language == 'zh':
                    text = ''.join(tokens)
                else:
                    text = ' '.join(tokens)
                # labels
                chunks = decode_tag_sequence(tags)
                labels = []
                for chunk in chunks:
                    if language == 'zh':
                        span = ''.join(tokens[chunk[1]: chunk[2]])
                    else:
                        span = ' '.join(tokens[chunk[1]: chunk[2]])
                    labels.append((span, chunk[0]))
                labels = list(set(labels))
                labels = [{'span': x[0], 'type': x[1]} for x in labels]
                # add to data
                feature = {
                    'tokens': tokens,
                    'tags': tags,
                    'text': text,
                    'labels': labels,
                    'id': str(id),
                }
                fw.write(json.dumps(feature, ensure_ascii=False))
                fw.write('\n')
                id += 1
                tokens, tags = [], []
            else:
                token, tag = line.strip().split(' ')
                tokens.append(token)
                tags.append(tag)
        fr.close()
        fw.close()

def split_demo(dataset='en_conll03', num_demo=100, seed=42):
    """
    Split from `dev.jsonl`.
    Only choose non-empty labels as examples.
    """
    random.seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, dataset)
    data = []
    src_path = os.path.join(dir_path, 'dev.jsonl')
    dst_path = os.path.join(dir_path, 'demo.jsonl')
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line)
    random.shuffle(data)
    cnt = 0
    with open(dst_path, 'w', encoding='utf-8') as f:
        for line in data:
            sample = json.loads(line.strip())
            if len(sample['labels']) != 0:
                f.write(line)
                cnt += 1
            if cnt == num_demo:
                break

def add_meta(dataset='en_conll03'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, dataset)
    dst_path = os.path.join(dir_path, 'meta.json')
    if dataset == 'en_conll03':
        tags = ['PER', 'ORG', 'LOC', 'MISC']
    elif dataset == 'zh_msra':
        tags = ['PER', 'LOC', 'ORG']
    elif dataset == 'zh_onto4':
        tags = ['PER', 'ORG','GPE', 'LOC']
    tag2id = {'O': 0}
    for tag in tags:
        for prefix in ['B', 'M', 'E', 'S']:
            tag2id[f'{prefix}-{tag}'] = len(tag2id)
    meta = {'tag2id': tag2id, 'tagset': tags}
    json.dump(meta, open(dst_path, 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    for dataset in ['en_conll03', 'zh_onto4']:
        # DO length check
        valid_length(dataset)
        # DO prepro
        prepro(dataset)
        # DO label check
        valid_label(dataset)
        # DO demostration split
        split_demo(dataset, seed=42)
        # ADD meta
        add_meta(dataset)