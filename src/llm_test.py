# Script for online query & evaluation by llm (gpt) on test data.
import os
import time
from tqdm import tqdm
import ujson as json
import argparse
from func_timeout.exceptions import FunctionTimedOut
from openai.error import RateLimitError

from .llm_annotator import Annotator
from .evaluation import MetricForNer

RETRY = 3

def get_opt():
    parser = argparse.ArgumentParser()
    # action type
    parser.add_argument('--query', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    # parameters
    parser.add_argument('--engine', default='gpt-35-turbo-0301', type=str)
    parser.add_argument('--dataset', default='en_conll03', type=str)
    parser.add_argument('--setting', default='zero', type=str)
    parser.add_argument('--query_batch_size', default=20, type=int)
    
    return parser.parse_args()


def inference(args):
    """
    Query gpt to inference on test data.
    Args:
        - dataset
        - setting (random, knn, zero)
        - query_batch_size
    """
    # task type
    task_type = 're' if args.dataset == 'en_retacred' else 'ner'
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dst_dir = os.path.join(dir_path, f'results/{args.dataset}/')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_path = os.path.join(dir_path, f'data/{args.dataset}/test.jsonl')
    dst_path = os.path.join(dir_path, f'results/{args.dataset}/test-{args.setting}-{args.engine}.jsonl')
    # load data
    data = []
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    # load demo and index
    demo_file_path = os.path.join(dir_path, f'data/{args.dataset}/demo.jsonl')
    demo_file = dict()
    with open(demo_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            demo_file[sample['id']] = sample
    if args.setting == 'random' or args.setting == 'knn':
        demo_index_path = os.path.join(dir_path, f'data/{args.dataset}/test-{args.setting}-demo.json')
        demo_index = json.load(open(demo_index_path, 'r', encoding='utf-8'))
    elif args.setting == 'zero':
        pass
    else:
        raise ValueError(f'Unknown setting {args.setting}.')
    # annotate
    annotator = Annotator(engine=args.engine, config_name=f'{args.dataset}_base')
    if args.engine == 'gpt4':
        assert args.query_batch_size == 1
    results = []
    total_cost = 0
    inputs = []
    for sample in data:
        if args.setting == 'random' or args.setting == 'knn':
            # pointer: {'id': id, ('score': score)}
            demo = [demo_file[pointer['id']] for pointer in reversed(demo_index[sample['id']])]
        else:
            demo = None
        inputs.append({
            'sample': sample,
            'demo': demo
        })
    tick = 0
    for left in tqdm(range(0, len(data), args.query_batch_size)):
        right = min(len(data), left + args.query_batch_size)
        samples = inputs[left: right]

        outputs = [None] * (right - left)
        if args.engine == 'gpt4':
            # annotate by sample; query_batch_size == 1
            sample = samples[0]
            for j in range(RETRY):
                try:
                    output, cost = annotator.online_annotate(sample['sample'], sample['demo'], return_cost=True)
                    break
                except FunctionTimedOut:
                    print('Timeout. Retrying...')
                except RateLimitError:
                    print('Rate limit. Sleep for 60 seconds...')
                    time.sleep(60)
            outputs = [output]
        else:
            # annotate by batch
            for j in range(RETRY):
                try:
                    outputs, cost = annotator.batch_annotate(samples, return_cost=True)
                    break
                except FunctionTimedOut:
                    print('Timeout. Retrying...')
                except RateLimitError:
                    print('Rate limit. Sleep for 60 seconds...')
                    time.sleep(60)
        
        results.extend(outputs)
        total_cost += cost
        if task_type == 'ner':
            for result in outputs:
                result_string = json.dumps(result, ensure_ascii=False)
                with open(dst_path, 'a', encoding='utf-8') as fw:
                    fw.write(result_string + '\n')
        elif task_type == 're':
            for result, sample in zip(outputs, samples):
                if result not in [
                    "no_relation", 
                    "per:age",
                    "per:parents",
                    "per:spouse",
                    "per:siblings",
                    "per:children",
                    "per:nationality"
                    ]:
                    result = 'no_relation'
                result_string = json.dumps(result, ensure_ascii=False)
                with open(dst_path, 'a', encoding='utf-8') as fw:
                    fw.write(result_string + '\t' + 
                            sample['sample']['label'] + '\t' +
                            sample['sample']['id'] + '\n')
        tick += 1
        # if tick % 50 == 0:
        #     time.sleep(30)
    print(f'total cost: {total_cost}')
    print(f'average cost: {total_cost / len(data)}')

def evaluation(args):
    # path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    gold_path = os.path.join(dir_path, f'data/{args.dataset}/test.jsonl')
    pred_path = os.path.join(dir_path, f'results/{args.dataset}/test-{args.setting}.jsonl')
    # classify task type
    ner_flag =  args.dataset in ['en_conll03', 'zh_msra', 'zh_onto4']
    if ner_flag:
        language = args.dataset[:2]
        # convert predictions to tag sequence, and use evaluation function to compare.
        gold_data, pred_data = [], []               # raw data
        gold_tag_seqs, pred_tag_seqs = [], []       # sequence of tags
        with open(gold_path, 'r', encoding='utf-8') as f:
            for line in f:
                gold_data.append(json.loads(line.strip()))
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                labels = json.loads(line.strip())
                if labels is None:
                    labels = []
                labels = [entity for entity in labels if isinstance(entity['span'], str)]
                pred_data.append(labels)

        for sample, pred_labels in zip(gold_data, pred_data):
            pred_labels = sorted(pred_labels, key=lambda x: len(x['span']), reverse=True)
            span_to_type = {entity['span']: entity['type'] for entity in pred_labels}
            if language == 'zh':
                sent = sample['text']
                for entity in pred_labels:
                    span = entity['span'].strip()
                    sent = ('[sep]' + span + '[sep]').join(sent.split(span))
                words = sent.split('[sep]')
            else:
                # build a tokenizer first
                dictionary = dict()
                for token in sample['tokens']:
                    if token not in dictionary:
                        dictionary[token] = f'[{len(dictionary)}]'
                # print(dictionary)
                id_string = ' '.join([dictionary[token] for token in sample['tokens']])
                # print(id_string)
                for entity in pred_labels:
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
                        label = ([f'B-{span_to_type[word]}'] + [f'M-{span_to_type[word]}' for _ in range(word_length - 2)]
                                  + [f'E-{span_to_type[word]}'])
                else:
                    label = ['O' for _ in range(word_length)]  
                seq_label.extend(label)

            assert len(sample['tags']) == len(seq_label)

            gold_tag_seqs.append(sample['tags'])
            pred_tag_seqs.append(seq_label)
        metric = MetricForNer(None)
        metric.update(gold_tag_seqs, pred_tag_seqs)
        f1, overall, class_info = metric.stats()
        print(f1)
        print(overall)
        print(class_info)
    else:   # re task
        gold_data, pred_data = [], []
        with open(gold_path, 'r', encoding='utf-8') as f:
            for line in f:
                gold_data.append(json.loads(line.strip()))
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('\t')[0]
                labels = json.loads(line.strip())
                if labels is None:
                    labels = []
                pred_data.append(labels)
        tp, tn, fp, fn, cnt = 0, 0, 0, 0, 0
        for sample, pred_label in zip(gold_data, pred_data):
            cnt += 1
            if pred_label == sample['label']:
                if pred_label != 'no_relation':
                    tp += 1
                else:
                    tn += 1
            else:
                if sample['label'] != 'no_relation':
                    fn += 1
                if pred_label != 'no_relation' and pred_label is not None:
                    fp += 1
        acc = (tp + tn) / cnt
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        print(f'acc: {acc} | f1: {f1} | precision: {p} | recall: {r}')
        print(f'tp: {tp} | tn: {tn} | fp: {fp} | fn: {fn} | cnt: {cnt}')


if __name__ == '__main__':
    args = get_opt()
    if args.query:
        inference(args)
    if args.eval:
        evaluation(args)