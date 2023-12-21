import ujson as json
import os
from ujson import JSONDecodeError
from openai.error import RateLimitError
from func_timeout import func_set_timeout
from .connector import Connector

class Annotator(Connector):
    def __init__(self, engine: str='gpt-35-turbo-0301',
                 config_name: str='default', dataset: str=None, task: str=None,      # dataset aligned with dirname
                 description: str=None, guidance: str=None, 
                 input_format: str=None, output_format: str=None, struct_format: str=None):
        """
        `dataset` is used in postprocess.
        """
        super().__init__(engine)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'configs/{}.json'.format(config_name))
        config = json.load(open(config_path, 'r', encoding='utf-8'))

        self.dataset = config['dataset'] if dataset is None else dataset
        self.task = config['task'] if task is None else task
        assert self.task in ['ner', 're']

        self.description = config['description'] if description is None else description
        self.guidance = config['guidance'] if guidance is None else guidance
        self.input_format = config['input_format'] if input_format is None else input_format
        self.output_format = config['output_format'] if output_format is None else output_format
        self.struct_format = config['struct_format'] if struct_format is None else struct_format    # for RE

    def __get_prompt(self, sample, demo=None):
        """
        Return the prompt for single sample/demo pair.
        Demo format:
        NER:
        {
            'text': text,
            'labels': [
                {
                    'span': span,
                    'type': type
                }, ...
            ]
        }
        RE:
        {
            'text': text,
            'subject': subject,
            'object': object,
            'relation': relation
        }

        Sample format:
        Same as demo, without 'labels'/'relation' key.
        """
        if demo is not None:
            if self.task == 'ner':
                demo_string = ''
                for example in demo:
                    example_string = (self.input_format.format(example['text']) + '\n' + 
                                      self.output_format.format(json.dumps(example['labels'], ensure_ascii=False)) + '\n\n')
                    demo_string = demo_string + example_string
            elif self.task == 're':
                demo_string = ''
                for example in demo:
                    example_string = (self.input_format.format(example['text']) + '\n' + 
                                      self.struct_format.format(example['h']['name'], example['t']['name']) + '\n' +
                                      self.output_format.format(example['label']) + '\n\n')
                    demo_string = demo_string + example_string
                demo_string = demo_string[:-1]
            else:
                raise ValueError('Need to implement demo construction first.')
        else:
            demo_string = ''
    
        if self.task == 'ner':
            task_string = self.input_format.format(sample['text']) + '\n' + self.output_format.format('')
        elif self.task == 're':
            task_string = (self.input_format.format(sample['text']) + '\n' +
                           self.struct_format.format(sample['h']['name'], sample['t']['name']) + '\n' +
                           self.output_format.format(''))
        else:
            raise ValueError('Unknown task type.')
        prompt = '\n'.join([self.description, self.guidance, demo_string, task_string])
        return prompt
    
    def __postprocess(self, result):
        if self.task == 'ner':              # filter out invalid type
            # load from meta
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = os.path.dirname(os.path.dirname(dir_path))
            meta_path = os.path.join(dir_path, f'data/{self.dataset}/meta.json')
            meta = json.load(open(meta_path, 'r'))
            tagset = meta['tagset']
            outputs = []
            for entity in result:
                if not isinstance(entity, dict):
                    continue
                if 'type' not in entity or 'span' not in entity:
                    continue
                if entity['type'] in tagset:
                    outputs.append(entity)
            return outputs
        elif self.task == 're':
            return result

    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None, return_cost: bool=False):
        """
        Annotate single sample.
        """
        prompt = self.__get_prompt(sample, demo)
        try:
            response = self.online_query(prompt)
        except RateLimitError:
            raise RateLimitError()
        except Exception:                               # in the case of content violation
            if return_cost:
                return None, 0
            return None
        if self.task == 'ner':
            try:
                result = json.loads(response['result'])     # this step takes risk
            except JSONDecodeError:
                result = response['result']
                print('JSONDecodeError occurs with:\n{}'.format(result))

            if not isinstance(result, list):
                result = []                                 # [] means no entities; None means sth goes wrong!
                                                            # re: need to change
        elif self.task == 're':
            result = response['result'].strip()
        else:
            raise ValueError('Unknown task type.')
        result = self.__postprocess(result)

        if return_cost:
            return result, response['cost']
        return result
    
    @func_set_timeout(30)
    def batch_annotate(self, inputs, return_cost: bool=False):
        """
        Each input is a {'sample': sample, 'demo': demo} dictionary.
        """
        if self.engine == 'gpt4':
            raise ValueError('ChatCompletion API does not support batch inference.')
        prompts = [self.__get_prompt(x['sample'], x['demo']) for x in inputs]
        try:
            response = self.batch_query(prompts)
        except Exception:
            print(f'Exception occurs with inputs: {prompts}.')
            if return_cost:
                return [None for _ in range(len(inputs))], 0
            return [None for _ in range(len(inputs))]

        results = []
        for x in response['results']:
            if self.task == 'ner':
                try:        
                    result = json.loads(x.strip())
                except JSONDecodeError:
                    result = x
                    print('JSONDecodeError occurs with:\n{}'.format(result))
                if not isinstance(result, list):
                    result = []
            elif self.task == 're':
                result = x.strip()
            else:
                raise ValueError('Unknown task type.')
            result = self.__postprocess(result)
            results.append(result)
        if return_cost:
            return results, response['cost']
        return results
    
        
if __name__ == '__main__':
    annotator = Annotator(engine='gpt4', config_name='zh_onto4_base')
    sample = {"tokens":["因","盛","产","绿","竹","笋","，","被","誉","为","「","绿","竹","笋","的","故","乡","」","的","八","里","，","就","像","台","湾","许","多","大","大","小","小","遥","远","的","乡","镇","，","在","期","待","与","失","落","中","，","承","载","着","生","活","必","需","的","悲","苦","与","欢","乐","，","并","由","于","位","处","边","陲","，","担","负","着","众","人","不","愿","承","受","之","重","。"],"tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-GPE","E-GPE","O","O","O","B-GPE","E-GPE","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"],"text":"因盛产绿竹笋，被誉为「绿竹笋的故乡」的八里，就像台湾许多大大小小遥远的乡镇，在期待与失落中，承载着生活必需的悲苦与欢乐，并由于位处边陲，担负着众人不愿承受之重。","labels":[{"span":"台湾","type":"GPE"},{"span":"八里","type":"GPE"}],"id":"23549"}
    demo = [sample]
    print(annotator.online_annotate(sample))