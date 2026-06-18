import json

import torch
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.process_data.process_bin import create_binary_data
from src.models.bin_model import BinModel
from src.datasets.bin_dataset import BinDataset
from src.utils.read_input import read_data


def calculate_metrics(pred_labels, labels):
    tp = fp = fn = 0

    for pred, gold in zip(pred_labels, labels):
        pred, gold = pred.item(), gold.item()
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1

    if tp + fp == 0 or tp + fn == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": tp + fn}

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1        = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1, "n": tp + fn}


def train(
    train: str,
    out_path: str,
    model_path: str,
    test: str = None,
    pretrained_weights: str = None,
    epochs: int = 3,
    domain: str = "",
):
    loss_file  = f"{out_path}/log_files/bin_loss.jsonl"
    preds_file = f"{out_path}/predictions/bin_predictions.jsonl"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = BinModel(model_path)
    model.to(device)

    # Training data
    train_path = create_binary_data(train, f"{out_path}/processed/bin_processed.jsonl")
    train_data = read_data(train_path)
    train_loader = DataLoader(BinDataset(train_data, tokenizer), batch_size=16, shuffle=False)

    # Test data (optional)
    test_loader = None
    if test is not None:
        test_path = create_binary_data(train, f"{out_path}/processed/bin_processed.jsonl")
        test_data = read_data(test_path)
        test_loader = DataLoader(BinDataset(test_data, tokenizer), batch_size=16, shuffle=False)

    # Load pretrained weights (optional)
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location=device)
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Loaded weights: skipping training")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)

        with open(loss_file, 'w') as l:
            model.train()
            for epoch in range(epochs):
                total_loss = 0

                for batch in train_loader:
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    labels         = batch['labels'].to(device)
                    ids            = batch['ID']

                    optimizer.zero_grad()

                    loss = model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )['loss']
                    print(loss)

                    l.write(str(loss.item()) + '\n')
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{out_path}/trained_models/bin_model_{domain}.pt")

    # Evaluation
    if test_loader is not None:
        # Check if labels are available or not
        with open(test) as f:
            has_labels = 'Quadruplet' in json.loads(f.readline())

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad(), open(preds_file, 'w') as f:
            for batch in test_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                ids            = batch['ID']

                if has_labels:
                    labels     = batch['labels'].to(device)

                preds = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pred_labels = (torch.sigmoid(preds) > 0.95).long()

                if has_labels:
                    batch_output = [
                        {
                            "ID":              ide,
                            "Aspect":          aspect,
                            "Opinion":         opinion,
                            "Text":            sent,
                            "predicted_label": pred.item(),
                            "gold_label":      gold.item(),
                        }
                        for ide, aspect, opinion, sent, pred, gold in zip(
                            ids, batch['aspect'], batch['opinion'], batch['sentence'], pred_labels, labels
                        )
                    ]
                else:
                    batch_output = [
                        {
                            "ID":              ide,
                            "Aspect":          aspect,
                            "Opinion":         opinion,
                            "Text":            sent,
                            "predicted_label": pred.item(),
                        }
                        for ide, aspect, opinion, sent, pred in zip(
                            ids, batch['aspect'], batch['opinion'], batch['sentence'], pred_labels
                        )
                        if pred.item() == 1
                    ]

                for d in batch_output:
                    json.dump(d, f)
                    f.write('\n')

                if has_labels:
                    all_preds.extend(pred_labels)
                    all_labels.extend(labels)

        if has_labels:
            metrics = calculate_metrics(all_preds, all_labels)
            print(f"Test | P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}"
                f"F1: {metrics['f1']:.4f}  N: {metrics['n']}")