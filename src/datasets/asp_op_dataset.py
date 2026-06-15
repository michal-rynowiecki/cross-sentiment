import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BIODataset(Dataset):
    def __init__(self, data, tokenizer, tag_to_id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _tokenize(self, words):
        return self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def _align_labels(self, word_ids, label_ids):
        aligned = []
        current_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == current_word_idx:
                aligned.append(0)
            else:
                aligned.append(label_ids[word_idx])
            current_word_idx = word_idx
        return aligned

    def __getitem__(self, idx):
        entry = self.data[idx]
        words = [pair[0] for pair in entry['Tags']]
        tags  = [self.tag_to_id[pair[1]] for pair in entry['Tags']]

        encoding  = self._tokenize(words)
        labels    = self._align_labels(encoding.word_ids(), tags)

        return {
            'ID':             entry['ID'],
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(labels),
        }


class BIODatasetDouble(BIODataset):
    def __init__(self, data1, data2, tokenizer, tag_to_id, max_len=128):
        super().__init__(data1, tokenizer, tag_to_id, max_len)
        self.data2 = data2

    def __getitem__(self, idx):
        entry = self.data[idx]
        words = [pair[0] for pair in entry['Tags']]
        tags1 = [self.tag_to_id[pair[1]] for pair in entry['Tags']]
        tags2 = [self.tag_to_id[pair[1]] for pair in self.data2[idx]['Tags']]

        encoding = self._tokenize(words)
        word_ids = encoding.word_ids()

        return {
            'ID':             entry['ID'],
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels1':        torch.tensor(self._align_labels(word_ids, tags1)),
            'labels2':        torch.tensor(self._align_labels(word_ids, tags2)),
        }


class BIODatasetInference(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry    = self.data[idx]
        sentence = entry['Text']

        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt',
        )

        return {
            'ID':               entry,
            'sentence':         sentence,
            'input_ids':        encoding['input_ids'].flatten(),
            'attention_mask':   encoding['attention_mask'].flatten(),
            'token_type_ids':   encoding['token_type_ids'].flatten(),
        }