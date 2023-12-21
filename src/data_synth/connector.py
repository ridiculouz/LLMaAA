import os
import openai
import ujson as json

class Connector:
    def __init__(self, engine='gpt-35-turbo-0301'):
        # load key and base from config file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        config_path = os.path.join(dir_path, 'openai_config.json')
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        # fill with config
        self.key = config['key']
        self.base = config['base']
        self.type = 'azure'
        self.version = '2022-12-01'
        self.engine = engine
        openai.api_key = self.key
        openai.api_base = self.base
        openai.api_type = self.type
        openai.api_version = self.version

        self.stop = ['<|im_end|>', '<|im_sep|>']

    def online_query(self, prompt):
        task_param = {
            'engine': self.engine,
            'prompt': prompt,
            'temperature': 1,
            'max_tokens': 2000,
            'n': 1,
            'stop': self.stop,
            # 'logprobs': 5,
        }
        response = openai.Completion.create(**task_param)
        result = response['choices'][0]['text']
        cost = response['usage']['total_tokens']
        print('The cost of current query is {} tokens.'.format(cost))
        return {'result': result, 'cost': cost}
    
    def batch_query(self, prompts):
        """
        prompts is a list of prompt.
        """
        task_param = {
            'engine': self.engine,
            'prompt': prompts,
            'temperature': 1,
            'max_tokens': 2000,
            'n': 1,
            'stop': self.stop,
        }
        response = openai.Completion.create(**task_param)
        choices = sorted(response['choices'], key=lambda x: x['index'])
        results = [x['text'] for x in choices]
        cost = response['usage']['total_tokens']
        print('The cost of current query is {} tokens.'.format(cost))
        return {'results': results, 'cost': cost}
    
if __name__ == '__main__':
    conn = Connector()
    prompt = """<|im_start|> You are an intelligent text data generator. Generate 5 high-quality and diverse sentences containing entities for the following types:
[person (PER), location (LOC), organization (ORG), miscellaneous entity (MISC)]
Write one sample per line. No other output.

Example:
{"text": text, "entities": [{"name": name, "type": type}]}

Output:
"""
    res = conn.online_query(prompt)
    print(res['result'])