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

class DualModule(nn.Module):
    def __init__(self, model1, model2, attn_layers=None, dropout_rate=0.1):
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

        self.dropout = nn.Dropout(dropout_rate)

        self.regressor1 = nn.Linear(self.bert1.config.hidden_size, 1)
        self.regressor2 = nn.Linear(self.bert2.config.hidden_size, 1)

        self.loss_v = nn.MSELoss()
        self.loss_a = nn.MSELoss()


    def get_mask(self, attention_mask):
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def forward(self, input_ids, attention_mask=None, gold1=None, gold2=None):
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
            
                hidden1 = attn_layer1(query=hidden1, key=hidden2, value=hidden2, mask=attention_mask)
                hidden2 = attn_layer2(query=hidden2, key=hidden1, value=hidden1, mask=attention_mask)

        cls_embedding1 = hidden1[:, 0, :]
        cls_embedding1 = self.dropout(cls_embedding1)

        cls_embedding2 = hidden2[:, 0, :]
        cls_embedding2 = self.dropout(cls_embedding2)

        score1 = self.regressor1(cls_embedding1)
        score2 = self.regressor2(cls_embedding2)

        if gold1 is not None and gold2 is not None:
            loss1 = self.loss_v(score1.flatten(), gold1)
            loss2 = self.loss_a(score2.flatten(), gold2)
            return loss1 + loss2
        
        else:
            return score1, score2