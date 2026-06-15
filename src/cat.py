import importlib
import json
import numpy as np
from random import shuffle

import torch
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.models.cat_model import CatModule
from src.datasets.cat_dataset import CatDataset
from src.process_data.process_cat import split_json_categories


LABEL_MAPS = {
    "laptop": {
        "id2label_1": {
            0: 'LAPTOP',    1: 'HARDWARE',          2: 'DISPLAY',       3: 'KEYBOARD',
            4: 'BATTERY',   5: 'SOFTWARE',           6: 'SUPPORT',       7: 'MULTIMEDIA_DEVICES',
            8: 'OS',        9: 'HARD_DISK',         10: 'FANS_COOLING', 11: 'COMPANY',
            12: 'CPU',      13: 'MEMORY',           14: 'OPTICAL_DRIVES',15: 'PORTS',
            16: 'GRAPHICS', 17: 'SHIPPING',         18: 'POWER_SUPPLY', 19: 'MOUSE',
            20: 'WARRANTY', 21: 'MOTHERBOARD',      22: 'OUT_OF_SCOPE'
        },
        "id2label_2": {
            0: 'DESIGN_FEATURES',   1: 'GENERAL',   2: 'OPERATION_PERFORMANCE', 3: 'QUALITY',
            4: 'USABILITY',         5: 'PRICE',     6: 'PORTABILITY',           7: 'MISCELLANEOUS',
            8: 'CONNECTIVITY'
        },
    },
    "restaurant": {
        "id2label_1": {
            0: 'RESTAURANT', 1: 'FOOD',    2: 'DRINKS', 3: 'AMBIENCE',
            4: 'SERVICE',    5: 'LOCATION', 6: 'OUT_OF_SCOPE'
        },
        "id2label_2": {
            0: 'GENERAL', 1: 'PRICES', 2: 'QUALITY', 3: 'STYLE_OPTIONS', 4: 'MISCELLANEOUS'
        },
    },
}


def combine(quadruplet, text):
    return f"{quadruplet['Aspect']}[SEP]{quadruplet['Opinion']}[SEP]{text}"


def get_damped_class_weights(counts, device):
    counts = np.array(counts, dtype=float)
    counts[counts == 0] = 1
    weights = np.sqrt(np.max(counts) / counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def load_module_class(attn_type: str):
    module_map = {
        "self":  "src.models.cat_self",
        "cross": "src.models.cat_model",
    }
    mod = importlib.import_module(module_map[attn_type])
    return mod.CatModule


def calculate_metrics(predictions1, predictions2, labels1, labels2):
    correct1 = correct2 = both = total = 0

    for p1, p2, l1, l2 in zip(predictions1, predictions2, labels1, labels2):
        total += 1
        c1 = p1.item() == l1.item()
        c2 = p2.item() == l2.item()
        if c1: correct1 += 1
        if c2: correct2 += 1
        if c1 and c2: both += 1

    if total == 0:
        return {"cat1_acc": 0.0, "cat2_acc": 0.0, "both_acc": 0.0, "n": 0}

    return {
        "cat1_acc": correct1 / total,
        "cat2_acc": correct2 / total,
        "both_acc": both / total,
        "n":        total,
    }

def train(
    train: str,
    out_path: str,
    model_path: str,
    attn_layers: list[int],
    attn_type: str,         # "cross" or "self"
    domain: str,            # "laptop" or "restaurant"
    test: str = None,
    pretrained_weights: str = None,
    epochs: int = 3,
):
    id2label_1 = LABEL_MAPS[domain]["id2label_1"]
    id2label_2 = LABEL_MAPS[domain]["id2label_2"]

    loss_file  = f"{out_path}/log_files/{attn_type}_loss_cat.jsonl"
    preds_file = f"{out_path}/predictions/{attn_type}_cat_predictions.jsonl"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    DualModule = load_module_class(attn_type)

    # Training data
    train_data   = []
    cat1_counts  = {a: 0 for a in id2label_1.values()}
    cat2_counts  = {a: 0 for a in id2label_2.values()}
    train_path = split_json_categories(train, f"{out_path}/processed/cat_processed.jsonl")
    
    with open(train_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            for elem in temp['Quadruplet']:
                cat1_counts[elem['Cat1']] += 1
                cat2_counts[elem['Cat2']] += 1
                train_data.append({
                    'ID':      temp['ID'],
                    'Text':    temp['Text'],
                    'Aspect':  elem['Aspect'],
                    'Opinion': elem['Opinion'],
                    'Cat1':    elem['Cat1'],
                    'Cat2':    elem['Cat2'],
                })

    shuffle(train_data)

    class_weights1 = get_damped_class_weights(list(cat1_counts.values()), device)
    class_weights2 = get_damped_class_weights(list(cat2_counts.values()), device)

    train_loader = DataLoader(CatDataset(train_data, tokenizer, id2label_1, id2label_2), batch_size=8, shuffle=False)

    # Test (optional)
    test_loader = None
    if test is not None:
        test_data = []
        test_path = split_json_categories(test, f"{out_path}/processed/cat_processed.jsonl")
        
        # Check if labels are available or not
        with open(test) as f:
            has_labels = 'Quadruplet' in json.loads(f.readline())

        with open(test_path, 'r') as f:
            for line in f:
                temp = json.loads(line)

                if has_labels:
                    for elem in temp['Quadruplet']:
                        cat1_counts[elem['Cat1']] += 1
                        cat2_counts[elem['Cat2']] += 1
                        test_data.append({
                            'ID':      temp['ID'],
                            'Text':    temp['Text'],
                            'Aspect':  elem['Aspect'],
                            'Opinion': elem['Opinion'],
                            'Cat1':    elem['Cat1'],
                            'Cat2':    elem['Cat2'],
                        })
                else:
                    test_data.append({
                        'ID':      temp['ID'],
                        'Text':    temp['Text'],
                        'Aspect':  temp['Aspect'],
                        'Opinion': temp['Opinion'],
                    })

    if pretrained_weights is not None:
        class_weights1 = get_damped_class_weights(list(cat1_counts.values()), device)
        class_weights2 = get_damped_class_weights(list(cat2_counts.values()), device)
    
    test_loader = DataLoader(CatDataset(test_data, tokenizer, id2label_1, id2label_2), batch_size=8, shuffle=False)

    model = DualModule(
        model_path, model_path,
        list(id2label_1.values()), list(id2label_2.values()),
        attn_layers=attn_layers,
        class1_weights=class_weights1,
        class2_weights=class_weights2,
    )
    model.to(device)

    # Load pretrained weights (optional)
    if pretrained_weights is not None:
        print(pretrained_weights)
        state_dict = torch.load(pretrained_weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {pretrained_weights}, skipping training.")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)

        with open(loss_file, 'w') as l:
            model.train()
            for epoch in range(epochs):
                total_loss = 0

                for batch in train_loader:
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels1        = batch['cat1'].to(device)
                    labels2        = batch['cat2'].to(device)

                    optimizer.zero_grad()

                    loss = model(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)
                    print(loss)
                    l.write(str(loss.item()) + '\n')

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{out_path}/trained_models/cat_model.pt")

    # Evaluation
    if test_loader is not None:
        model.eval()
        all_p1, all_p2, all_l1, all_l2 = [], [], [], []

        with torch.no_grad(), open(preds_file, 'w') as f:
            for batch in test_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ids            = batch['ID']
                if has_labels:
                    labels1        = batch['cat1']
                    labels2        = batch['cat2']

                predictions1, predictions2 = model(input_ids, attention_mask=attention_mask)
                predictions1 = predictions1.argmax(dim=1)
                predictions2 = predictions2.argmax(dim=1)

                batch_output = [
                    {
                        "ID":      id_,
                        "Text":    text,
                        "Aspect":  asp,
                        "Opinion": op,
                        "Cat1":    id2label_1[p1.item()],
                        "Cat2":    id2label_2[p2.item()],
                    }
                    for id_, text, asp, op, p1, p2 in zip(
                        ids, batch['text'], batch['aspect'], batch['opinion'],
                        predictions1, predictions2
                    )
                ]

                for d in batch_output:
                    json.dump(d, f)
                    f.write('\n')

                if has_labels:
                    all_p1.extend(predictions1); all_p2.extend(predictions2)
                    all_l1.extend(labels1);      all_l2.extend(labels2)

        if has_labels:
            metrics = calculate_metrics(all_p1, all_p2, all_l1, all_l2)
            print(f"Test | Cat1 Acc: {metrics['cat1_acc']:.4f}  Cat2 Acc: {metrics['cat2_acc']:.4f}  "
                f"Both: {metrics['both_acc']:.4f}  N: {metrics['n']}")