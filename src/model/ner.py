import torch.nn as nn
from transformers import AutoModel

class NerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class BertSoftmaxForNer(NerModel):
    def __init__(self, config, reduction='mean'):
        super(BertSoftmaxForNer, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.reduction = reduction
        
        self.bert = AutoModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act_fn = nn.GELU()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(self.act_fn(sequence_output))
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # Only keep active parts of the loss
            if self.loss_fn is None:
                raise ValueError('Should specific a loss function in training mode.')
            if attention_mask is not None:
                if self.reduction == 'none':                # leave the problem later
                    batch_sz, seq_len, num_class = logits.size()
                    logits = logits.view(-1, num_class)
                    labels = labels.view(-1)
                    loss = self.loss_fn(logits, labels)
                    loss = loss.view(batch_sz, seq_len)     
                else:                                       # handle the reduction. do filter.
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = self.loss_fn(active_logits, active_labels)
            else:
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)