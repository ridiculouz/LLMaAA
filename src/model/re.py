import torch
import torch.nn as nn
from transformers import AutoModel

class ReModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class BertMarkerForRe(ReModel):
    def __init__(self, config, reduction='none'):
        super(BertMarkerForRe, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.reduction = reduction
        
        self.bert = AutoModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.act_fn = nn.GELU()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def resize_token_embeddings(self, size: int):
        self.bert.resize_token_embeddings(size)

    def forward(self, input_ids, attention_mask=None, ht_pos=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(self.act_fn(sequence_output))

        hidden_dim = sequence_output.size(-1)
        ht_pos = ht_pos.unsqueeze(-1).repeat((1, 1, hidden_dim))    # [batch_sz, 2] -> [batch_sz, 2, hidden_dim]
        ht_embeddings = torch.gather(sequence_output, dim=1, index=ht_pos)
        ht_embeddings = ht_embeddings.view(-1, 2 * hidden_dim)

        logits = self.classifier(ht_embeddings)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # Only keep active parts of the loss
            if self.loss_fn is None:
                raise ValueError('Should specific a loss function in training mode.')
            loss = self.loss_fn(logits, labels)     # shape is decided by reduction
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)