import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils.read_input import read_data, read_conll
from src.utils.transform_tokens import get_entities, get_entities_batch

import json

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

class DualModule(nn.Module):
    def __init__(self, model1, model2, num_tags, attn_layers=None):
        super().__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)

        hidden_size = self.bert1.config.hidden_size
        num_layers = self.bert1.config.num_hidden_layers

        self.attn_layers = attn_layers if attn_layers is not None else [num_layers - 1]

        self.attn1_layers = nn.ModuleDict()
        self.attn2_layers = nn.ModuleDict()

        for layer_idx in self.attn_layers:
            self.attn1_layers[f"Cross-Attention {layer_idx}"] = CrossAttentionLayer(hidden_size)
            self.attn2_layers[f"Cross-Attention {layer_idx}"] = CrossAttentionLayer(hidden_size)

        self.emission1 = nn.Linear(hidden_size, num_tags)
        self.emission2 = nn.Linear(hidden_size, num_tags)
        self.crf1 = CRF(num_tags, batch_first=True)
        self.crf2 = CRF(num_tags, batch_first=True)

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
            
                hidden1 = attn_layer1(query=prev_hidden1, key=hidden2, value=prev_hidden2, mask=attention_mask)
                hidden2 = attn_layer2(query=prev_hidden2, key=prev_hidden1, value=prev_hidden1, mask=attention_mask)

        emissions1 = self.emission1(hidden1)
        emissions2 = self.emission2(hidden2)

        if labels1 is not None and labels2 is not None:
            loss1 = -self.crf1(emissions1, labels1, mask=attention_mask.bool(), reduction='mean')
            loss2 = -self.crf2(emissions2, labels2, mask=attention_mask.bool(), reduction='mean')
            return loss1 + loss2
        
        else:
            tags1 = self.crf1.decode(emissions1, mask=attention_mask.bool())
            tags2 = self.crf2.decode(emissions2, mask=attention_mask.bool())
        
            return tags1, tags2