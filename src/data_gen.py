# Script for data generation.
import argparse
import os
import math
import random
import ujson as json
from tqdm import tqdm
from func_timeout.exceptions import FunctionTimedOut

from .data_synth import Generator

RETRY = 3

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='en_conll03', type=str)
    parser.add_argument('--setting', default='zero', type=str)
    parser.add_argument('--data_size', default=500, type=int)
    parser.add_argument('--query_batch_size', default=1, type=int)
    # demo related
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--topk', default=5, type=int)
    return parser.parse_args()

def generate_data(args):
    # set seed
    random.seed(args.seed)
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dst_dir = os.path.join(dir_path, f'synth_data/{args.dataset}/{args.setting}')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, 'data.jsonl')
    if args.setting == 'demo':
        # load demo file
        src_path = os.path.join(dir_path, f'data/{args.dataset}/demo.jsonl')
        demo_pool = []
        with open(src_path, 'r', encoding='utf-8') as f:
            for line in f:
                demo_pool.append(json.loads(line.strip()))
    elif args.setting == 'zero':
        pass
    else:
        raise ValueError(f'Unknown setting {args.setting}.')
    # generate
    generator = Generator(args.dataset)
    synth_data = []
    total_cost = 0
    expected_step_size = generator.k * args.query_batch_size         # expected data size generated at one step
    num_steps = math.ceil(args.data_size / expected_step_size)
    for step in tqdm(range(num_steps)):
        if args.setting == 'demo':
            demo = random.sample(demo_pool, args.topk)
        elif args.setting == 'zero':
            demo = None
        for j in range(RETRY):
            try:
                outputs, cost = generator.batch_generate(demo, repeat=args.query_batch_size)
                break
            except FunctionTimedOut:
                print('Timeout. Retrying...')
        synth_data.extend(outputs['valid'])
        total_cost += cost
        with open(dst_path, 'a', encoding='utf-8') as f:
            for sample in outputs['valid']:
                sample_string = json.dumps(sample, ensure_ascii=False)
                f.write(sample_string + '\n')    
    print(f'total cost: {total_cost}')
    print(f'average cost: {total_cost / len(synth_data)}')


if __name__ == '__main__':
    args = get_opt()
    generate_data(args)        