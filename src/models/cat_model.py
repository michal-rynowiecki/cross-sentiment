import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from torch.utils.data import Dataset, DataLoader, random_split

from src.utils.read_input import read_data, read_conll
from src.utils.transform_tokens import get_entities, get_entities_batch

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, droput=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.droput = nn.Dropout(droput)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
            mask = (mask == 0)
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=mask
        )
    
        output=self.norm(query + self.droput(attn_output))
        return output

class CatModule(nn.Module):
    def __init__(self, model1, model2, classes1, classes2, attn_layers=None, dropout_rate=0.1, class1_weights=None, class2_weights=None):
        super().__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)

        if class1_weights == None:
            self.class1_weights = torch.tensor([0]*len(classes1))
            self.class2_weights = torch.tensor([0]*len(classes2))
            
        else:
            self.class1_weights = class1_weights
            self.class2_weights = class2_weights

        hidden_size = self.bert1.config.hidden_size
        num_layers = self.bert1.config.num_hidden_layers

        self.attn_layers = attn_layers if attn_layers is not None else [num_layers - 1]

        self.attn1_layers = nn.ModuleDict()
        self.attn2_layers = nn.ModuleDict()

        for layer_idx in self.attn_layers:
            self.attn1_layers[f"Cross-Attention {layer_idx}"] = CrossAttentionLayer(hidden_size)
            self.attn2_layers[f"Cross-Attention {layer_idx}"] = CrossAttentionLayer(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.classes1 = classes1
        self.classes2 = classes2

        self.pooler1 = nn.Linear(self.bert1.config.hidden_size, self.bert1.config.hidden_size)
        self.pooler2 = nn.Linear(self.bert2.config.hidden_size, self.bert2.config.hidden_size)
        self.tanh = nn.Tanh()

        self.classifier1 = nn.Linear(hidden_size, len(classes1))
        self.classifier2 = nn.Linear(hidden_size, len(classes2))

        self.loss_fct1 = nn.CrossEntropyLoss(weight=self.class1_weights)
        self.loss_fct2 = nn.CrossEntropyLoss(weight=self.class2_weights)


    def get_mask(self, attention_mask):
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):
        # Get embeddings
        hidden1 = self.bert1.embeddings(input_ids)
        hidden2 = self.bert2.embeddings(input_ids)

        mask = self.get_mask(attention_mask)

        layers1 = self.bert1.encoder.layer
        layers2 = self.bert2.encoder.layer

        for i, (layer1, layer2) in enumerate(zip(layers1, layers2)):
            hidden1 = layer1(hidden1, mask)[0]
            hidden2 = layer2(hidden2, mask)[0]

            if i in self.attn_layers:
                attn_layer1 = self.attn1_layers[f"Cross-Attention {i}"]
                attn_layer2 = self.attn2_layers[f"Cross-Attention {i}"]
            
                prev_hidden1 = hidden1.clone()
                prev_hidden2 = hidden2.clone()

                hidden1 = attn_layer1(query=prev_hidden1, key=prev_hidden2, value=prev_hidden2, mask=attention_mask)
                hidden2 = attn_layer2(query=prev_hidden2, key=prev_hidden1, value=prev_hidden1, mask=attention_mask)

        # Extract CLS tokens
        cls_token1 = hidden1[:, 0, :]
        cls_token2 = hidden2[:, 0, :]

        # Apply Pooler Logic
        pooled_output1 = self.tanh(self.pooler1(cls_token1))
        pooled_output2 = self.tanh(self.pooler2(cls_token2))

        # Apply Dropout
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)

        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)

        if labels1 is not None and labels2 is not None:
            loss1 = self.loss_fct1(logits1, labels1)
            loss2 = self.loss_fct2(logits2, labels2)
            return loss1 + loss2
        
        else:
            logits1 = logits1
            logits2 = logits2
        
            return logits1, logits2