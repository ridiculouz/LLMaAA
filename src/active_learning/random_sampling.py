import numpy as np
from torch import nn
import torch.nn as nn
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str='knn', engine: str='gpt-35-turbo-0301',):
        super().__init__(annotator_config_name, pool_size, setting, engine)

    def query(self, args, k: int, model: nn.Module, features):
        pool_indices = self._get_pool_indices()
        np.random.shuffle(pool_indices)
        lab_indices = pool_indices[:k]
        return lab_indices