import torch
from torch.utils.data import Dataset


class CatDataset(Dataset):
    def __init__(self, data, tokenizer, id2label_1=None, id2label_2=None, max_len=256):
        self.label2id_1 = {v: k for k, v in id2label_1.items()}
        self.label2id_2 = {v: k for k, v in id2label_2.items()}
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.inference  = 'Cat1' not in data[0]
        self.samples    = []

        for entry in data:
            sample = (entry['ID'], entry['Text'], entry['Aspect'], entry['Opinion'])

            if not self.inference:
                sample += (self.label2id_1[entry['Cat1']], self.label2id_2[entry['Cat2']])
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        id, text, aspect, opinion = sample[:4]

        encoding = self.tokenizer(
            f"{aspect}[SEP]{opinion}",
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt',
        )

        item = {
            'ID':               id,
            'text':             text,
            'aspect':           aspect,
            'opinion':          opinion,
            'input_ids':        encoding['input_ids'].flatten(),
            'attention_mask':   encoding['attention_mask'].flatten(),
            'token_type_ids':   encoding['token_type_ids'].flatten(),
        }

        if not self.inference:
            _, _, _, _, cat1, cat2 = sample
            item['cat1'] = torch.tensor(cat1, dtype=torch.long)
            item['cat2'] = torch.tensor(cat2, dtype=torch.long)

        return item