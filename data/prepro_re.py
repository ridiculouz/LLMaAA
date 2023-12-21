import os
import ujson as json
import random

def prepro(dir_path):
    id = 0
    for split in ['train', 'dev', 'test']:
        src_path = os.path.join(dir_path, f'{split}.txt')
        dst_path = os.path.join(dir_path, f'{split}.jsonl')
        fr = open(src_path, 'r', encoding='utf-8')
        fw = open(dst_path, 'w', encoding='utf-8')
        
        for line in fr:
            raw = json.loads(line.strip())
            sample = dict()
            sample['tokens'] = raw['token']
            sample['h'] = raw['h']
            sample['t'] = raw['t']
            sample['label'] = raw['relation']
            sample['text'] = ' '.join(raw['token'])
            sample['id'] = str(id)
            fw.write(json.dumps(sample, ensure_ascii=False))
            fw.write('\n')
            id += 1
        fr.close()
        fw.close()

def split_demo(dir_path, num_demo=100, seed=42):
    """
    Split from `dev.jsonl`.
    Only choose non-empty labels as examples.
    """
    random.seed(seed)
    data = []
    src_path = os.path.join(dir_path, 'dev.jsonl')
    dst_path = os.path.join(dir_path, 'demo.jsonl')
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line)
    random.shuffle(data)
    cnt = 0
    rel_freq = dict()
    with open(dst_path, 'w', encoding='utf-8') as f:
        for line in data:
            sample = json.loads(line.strip())
            relation = sample['label']
            if relation in rel_freq:
                rel_freq[relation] += 1
            else:
                rel_freq[relation] = 1
            f.write(line)
            cnt += 1
            if cnt == num_demo:
                break

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'en_retacred')
    prepro(dir_path)
    split_demo(dir_path)