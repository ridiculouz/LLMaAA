from tqdm import tqdm

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def get_embeddings(args, text):
    """
    `text` is a set of sentence expected to be encoded together.
    """
    data_size = len(text)
    batch_size = 20
    model = SentenceTransformer(args.embedding_model_name_or_path)
    model.to(args.device)
    embeddings = []
    for i in tqdm(range(0, data_size, batch_size), desc='calculate embeddings'):
        embeddings += model.encode(text[i: i + batch_size]).tolist()
    embeddings = torch.tensor(embeddings)
    return embeddings

def get_cosine_similarity(key: torch.Tensor, value: torch.Tensor):
    """
    Key, value are embeddings with same dimension.
    """
    ## norm
    key = F.normalize(key, dim=-1)
    value = F.normalize(value, dim=-1)
    ## matmul
    cos_sim = torch.mm(key, value.transpose(0, 1))
    return cos_sim