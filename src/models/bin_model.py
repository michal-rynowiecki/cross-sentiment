from transformers import AutoModel
from torch import nn
import torch.optim as optim
import torch

class BinModel(nn.Module):
    def __init__(self, model_path, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.pooler = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        
        self.tanh = nn.Tanh()

        self.classifier = nn.Linear(hidden_size, 1)

        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None)
        
        last_hidden_state = outputs.last_hidden_state

        cls_embedding = last_hidden_state[:, 0, :]
        
        pooled_output = self.tanh(self.pooler(cls_embedding))
        
        cls_embedding = self.dropout(pooled_output)

        score = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            loss = self.loss_fct(score.flatten(), labels.float())

        # Return loss if training otherwise return predictions
        return {'loss': loss, 'logits': score} if loss is not None else score