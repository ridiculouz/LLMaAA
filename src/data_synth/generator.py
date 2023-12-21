import ujson as json
import os
from ujson import JSONDecodeError
from func_timeout import func_set_timeout
from .connector import Connector

class Generator(Connector):
    def __init__(self, config_name: str=None, dataset: str=None, task: str=None,
                 description: str=None, k: int=None,                       # k: num of generated data
                 example_format: str=None, output_format: str=None,
                 default_example: str=None):
        """
        A wrapper for gpt query for data generation.
        """
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'configs/{}.json'.format(config_name))
        config = json.load(open(config_path, 'r', encoding='utf-8'))

        self.dataset = config['dataset'] if dataset is None else dataset
        self.task = config['task'] if task is None else task
        assert self.task in ['ner', 're']

        self.description = config['description']if description is None else description
        self.k = config['k'] if k is None else k
        self.example_format = config['example_format'] if example_format is None else example_format
        self.output_format = config['output_format'] if output_format is None else output_format
        self.default_example = config['default_example'] if default_example is None else default_example

    def __get_prompt(self, demo=None):
        """
        Return the prompt for connector.

        Demo format:
        NER:
        {"text": text, "entities": [{"name": name, "type": type}]}

        RE:
        {"text": text, "subject": subject, "object": object, "relation": relation}
        """
        if demo is not None:
            if self.task == 'ner':
                old_demo = demo
                demo = []
                for sample in old_demo:
                    entities = sample['labels']
                    entities = [{'name': e['span'], 'type': e['type']} for e in entities]
                    demo.append({'text': sample['text'],
                                 'entities': entities})
            elif self.task == 're':
                demo = [{'text': sample['text'],
                         'subject': sample['h']['name'],
                         'object': sample['t']['name'],
                         'relation': sample['label']} for sample in demo]
            else:
                raise ValueError('Need to implement demo construction first.')
            demo = [json.dumps(sample, ensure_ascii=False) for sample in demo]
            demo_string = '\n'.join(demo)
        else:
            demo_string = self.default_example

        prompt = '\n'.join([self.description.format(self.k), self.example_format, demo_string, self.output_format])
        
        instruction = '<|im_start|>system\nAssistant is a large language model trained by OpenAI.\n<|im_end|>" \
        "\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n'
        prompt = instruction.format(prompt)

        return prompt
    
    @func_set_timeout(60)
    def online_generate(self, demo=None, return_cost: bool=True):
        """
        Generate k data at once.
        """
        prompt = self.__get_prompt(demo)
        try:
            response = self.online_query(prompt)
        except Exception:
            print(f'Content violation in inputs {prompt}.')
            if return_cost:
                return None, 0
            return None
        result, cost = response['result'], response['cost']
        result = result.split('\n')
        # do not do regularization
        decodable_result, undecodable_result = [], []
        for string in result:
            try:
                sample = json.loads(string.strip())
                assert isinstance(sample, dict)
                decodable_result.append(sample)
            except (JSONDecodeError, AssertionError):
                undecodable_result.append(string)
        outputs = {'valid': decodable_result, 'invalid': undecodable_result}
        if return_cost:
            return outputs, cost
        return outputs
    
    @func_set_timeout(60)
    def batch_generate(self, demo=None, repeat: int=1, return_cost: bool=True):
        """
        Share the same samples in the same batch.
        Return the same format as `online_generate`.
        `torch.cat` style.
        """
        prompt = self.__get_prompt(demo)
        prompts = [prompt] * repeat
        try:
            response = self.batch_query(prompts)
        except Exception:
            print(f'Content violation in inputs {prompt}.')
            if return_cost:
                return None, 0
            return None
        results, cost = response['results'], response['cost']

        decodable_result, undecodable_result = [], []
        for result in results:
            result = result.split('\n')
            # do not do regularization.
            for string in result:
                try:
                    sample = json.loads(string.strip())
                    assert isinstance(sample, dict)
                    decodable_result.append(sample)
                except (JSONDecodeError, AssertionError):
                    undecodable_result.append(string)
        outputs = {'valid': decodable_result, 'invalid': undecodable_result}
        if return_cost:
            return outputs, cost
        return outputs
        
        
if __name__ == '__main__':
    desc = """<|im_start|> You are an intelligent text data generator. Generate {} high-quality and diverse sentences in news domain containing relational triplet for the following relation types:
- per:age : the age of SUBJ is OBJ
- per:parents : SUBJ's parent is OBJ
- per:spouse : SUBJ's spouse is OBJ
- per:siblings : SUBJ is the sibling of OBJ
- per:children : SUBJ's children is OBJ
- per:nationality: SUBJ's nationality is OBJ
- no_relation : SUBJ has no known relations to OBJ
Write one sample per line in json format. Subject and object must appear in the sentence. No other output.\n"""
    example_format = 'Example:'
    output_format = 'Output:\n'
    default_example = '{"text": text, "subject": subject, "object": object, "relation": relation}\n'

    # store config
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dst_path = os.path.join(dir_path, 'configs/en_retacred.json')
    config = {
        'description': desc,
        'k': 10,
        'example_format': example_format,
        'output_format': output_format,  
        'default_example': default_example,
        'dataset': 'en_reatacred',
        'task': 're',       
    }
    json.dump(config, open(dst_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    generator = Generator('en_retacred')
    result, cost = generator.batch_generate(repeat=1)
    for sample in result['valid']:
        print(sample)
    print('='*20)
    for i, sample in enumerate(result['invalid']):
        print(f'sample {i}:')
        print(sample)