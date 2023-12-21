# Script for computing demonstration indices for train/test data.
import argparse
import os
import random
import ujson as json

from .similarity.embedding import get_embeddings, get_cosine_similarity

def load_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def knn_demo_retrieval(args):
    """
    Retrieve demonstration for each sample based on similarity.
    Demos are picked from `demo.jsonl`.

    Args:
    dataset
    topk
    embedding_model_name_or_path
    device
    seed    (used in random)
    """
    # load data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, f'data/{args.dataset}')

    data = {split: load_data(os.path.join(dir_path, f'{split}.jsonl'))
            for split in ['demo', 'test', 'train']}
    text = {split: [sample['text'] for sample in data[split]]
            for split in data}
    # compute similarity
    embed = {split: get_embeddings(args, text[split])
             for split in text}
    for split in ['train', 'test']:
        sim = get_cosine_similarity(embed[split], embed['demo'])
        scores, indices = sim.topk(k=args.topk, dim=-1)
        results = dict()
        for i in range(scores.size(0)):
            key_id = data[split][i]['id']
            context = []
            for score, j in zip(scores[i], indices[i]):
                val_id = data['demo'][j]['id']
                context.append({'id': val_id, 'score': score.item()})
                if len(context) == args.topk:
                    break
            results[key_id] = context
        dst_path = os.path.join(dir_path, f'{split}-knn-demo.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
     

def random_demo_retrieval(args):
    """
    Retrieve demonstration randomly.
    """
    random.seed(args.seed)
    # load data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, f'data/{args.dataset}')

    data = {split: load_data(os.path.join(dir_path, f'{split}.jsonl'))
            for split in ['demo', 'test', 'train']}
    demo_id = [sample['id'] for sample in data['demo']]
    for split in ['train', 'test']:
        results = dict()
        for i in range(len(data[split])):
            key_id = data[split][i]['id']
            context = random.sample(demo_id, k=args.topk)
            context = [{'id': id} for id in context]
            results[key_id] = context
        dst_path = os.path.join(dir_path, f'{split}-random-demo.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='en_conll03', type=str)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--embedding_model_name_or_path',
                        default='paraphrase-multilingual-mpnet-base-v2', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--method', default='random', type=str)
    args = parser.parse_args()

    if args.method == 'random':
        random_demo_retrieval(args)
    elif args.method == 'knn':
        knn_demo_retrieval(args)
    else:
        raise ValueError('Unknown retrieval method.')