from transformers import AutoTokenizer
from torch import nn
import torch.optim as optim
import torch

import json

from torch.utils.data import Dataset, DataLoader, random_split

class BinDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.inference = False
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for entry in data:
            ide = entry['ID']
            sentence = entry['Text']
            aspect = entry['Aspect']
            opinion = entry['Opinion']
            if 'exists' in entry:
                label = 1.0 if entry['exists'] else 0.0
                self.samples.append((ide, aspect, opinion, sentence, label))

            else:
                self.inference = True
                self.samples.append((ide, aspect, opinion, sentence))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):

        if self.inference:
            ide, aspect, opinion, sentence = self.samples[index]
        else:
            ide, aspect, opinion, sentence, label = self.samples[index]

        encoding = self.tokenizer(
            f"{aspect}[SEP]{opinion}",
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        if self.inference:
            return {
                'ID': ide,
                'sentence': sentence,
                'aspect': aspect,
                'opinion': opinion,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
            }
        else:
            return {
                'ID': ide,
                'sentence': sentence,
                'aspect': aspect,
                'opinion': opinion,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }