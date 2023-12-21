# data processor
# load data as dataset
# handle cache update
import os
import ujson as json
from transformers import PreTrainedTokenizer, AutoTokenizer

from .ner_reader import ner_reader
from .re_reader import re_reader
from .synth_ner_reader import synth_ner_reader
from .synth_re_reader import synth_re_reader

class Processor(object):
    def __init__(self, dataset: str, tokenizer: PreTrainedTokenizer, cache_name: str='cache'):
        """
        Base class for data processor.
        Functionality:
            - Load data as dataset.
            - Handle cache update by reloading.
        """
        self.tokenizer = tokenizer
        if '-' in dataset:          # synth data, i.e. aumentation by data generation
                                    # should be as `{dataset}-{setting}`. e.g. `en_conll03-zero`      
            dataset, setting = dataset.split('-')
            # get dir_path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = os.path.dirname(os.path.dirname(dir_path))
            dir_path = os.path.join(dir_path, f'data/{dataset}')
            # determine task type & reader
            assert dataset in ['en_conll03', 'zh_msra', 'zh_onto4',     # ner task
                               'en_semeval10', 'en_retacred']           # re task
            self.task_type = 'ner' if dataset in ['en_conll03', 'zh_msra', 'zh_onto4'] else 're'
            if self.task_type == 'ner':
                self.reader = synth_ner_reader
            else:
                self.reader = synth_re_reader
            meta_path = os.path.join(dir_path, 'meta.json')
            self.tag2id = json.load(open(meta_path, 'r'))['tag2id']
            self.id2tag = {v: k for k, v in self.tag2id.items()}
            print(f'Loading dataset {dataset} with synthesized training data ...')
            self.features = self.reader(tokenizer, dataset, setting)
            print('Finish loading dataset.')
        else:                       # original dataloader
            self.dataset = dataset
            # get dir_path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = os.path.dirname(os.path.dirname(dir_path))
            dir_path = os.path.join(dir_path, f'data/{dataset}')
            # determine task type & reader
            assert dataset in ['en_conll03', 'zh_msra', 'zh_onto4',     # ner task
                               'en_semeval10', 'en_retacred']           # re task
            self.task_type = 'ner' if dataset in ['en_conll03', 'zh_msra', 'zh_onto4'] else 're'
            if self.task_type == 'ner':
                self.reader = ner_reader
            else:
                self.reader = re_reader
            meta_path = os.path.join(dir_path, 'meta.json')
            self.tag2id = json.load(open(meta_path, 'r'))['tag2id']
            self.id2tag = {v: k for k, v in self.tag2id.items()}
            # cache
            if cache_name is None:
                self.cache_file = None
                self.cache_name = None
                self.use_cache = False
            else:
                self.cache_file = os.path.join(dir_path, f'{cache_name}.json')
                self.cache_name = cache_name
                self.use_cache = True
                # create cache file if not exist
                if not os.path.exists(self.cache_file):
                    cache = dict()
                    json.dump(cache, open(self.cache_file, 'w'))
            print(f'Loading dataset {dataset} ...')
            self.features = self.reader(tokenizer, self.dataset, 
                                        self.cache_name, self.use_cache)
            print('Finish loading dataset.')

    def update_cache(self, records: dict):
        cache = json.load(open(self.cache_file, 'r', encoding='utf-8'))
        cache.update(records)
        json.dump(cache, open(self.cache_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    def reload(self):
        print('Reloading dataset...')
        self.features = self.reader(self.tokenizer, self.dataset, self.cache_name, self.use_cache)
        print('Finish reloading dataset.')

    def get_id2tag(self):
        return self.id2tag
    
    def get_tag2id(self):
        return self.tag2id
    
    def get_features(self, split: str):
        return self.features[split]
    

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    processor = Processor('en_conll03-zero', tokenizer, None)
    features = processor.features['train']
    print(features[0])