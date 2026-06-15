from transformers import AutoTokenizer
from torch import nn
import torch.optim as optim
import torch

import json

from torch.utils.data import Dataset, DataLoader, random_split

class VADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.inference = False

        for entry in data:
            id = entry['ID']
            text = entry['Text']
            aspect = entry['Aspect']
            opinion = entry['Opinion']
            cat1 = entry['Cat1']
            cat2 = entry['Cat2']
            if 'Valence' in entry:
                valence = entry['Valence']
                arousal = entry['Arousal']
                self.samples.append((id, text, aspect, opinion, cat1, cat2, valence, arousal))
            else:
                self.inference = True
                self.samples.append((id, text, aspect, opinion, cat1, cat2))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.inference:
            id, text, aspect, opinion, cat1, cat2 = self.samples[index]
        else:
            id, text, aspect, opinion, cat1, cat2, valence, arousal = self.samples[index]
        encoding = self.tokenizer(
            f"{aspect}[SEP]{opinion}[SEP]{cat1}, {cat2}",
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        if not self.inference:
            return {
                'ID': id,
                "text": text,
                'aspect': aspect,
                'opinion': opinion,
                'cat1': cat1,
                'cat2': cat2,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'valence': torch.tensor(float(valence), dtype=torch.float),
                'arousal': torch.tensor(float(arousal), dtype=torch.float),
            }

        else:
            return {
                'ID': id,
                "text": text,
                'aspect': aspect,
                'opinion': opinion,
                'cat1': cat1,
                'cat2': cat2,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
            }