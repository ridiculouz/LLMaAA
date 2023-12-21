import torch
from torch import nn
from kmeans_pytorch import kmeans
from .strategy import Strategy
from .utils import get_bert_embeddings

class KMeansSampling(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str = 'knn', engine: str='gpt-35-turbo-0301'):
        super().__init__(annotator_config_name, pool_size, setting, engine)

    def query(self, args, k: int, model: nn.Module, features):
        pool_indices = self._get_pool_indices()
        pool_features = [features[i] for i in pool_indices]
        if self.task_type == 'ner' or self.task_type == 're':
            # get bert embeddings
            embeddings = get_bert_embeddings(args, pool_features, model)
            # compute k means centers
            ids, centers = kmeans(X=embeddings, num_clusters=k, device=args.device)
            # since kmeans move data to cpu, we need to transfer back
            device = embeddings.device
            centers = centers.to(device)
            dist = torch.cdist(centers, embeddings)     # [n_clusters, n_samples]
            min_distances, lab_indices = torch.min(dist, dim=-1)
        else:
            raise ValueError('tbd.')
        lab_indices = [pool_indices[i] for i in lab_indices]
        return lab_indices