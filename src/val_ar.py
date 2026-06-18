import importlib
import json

import torch
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.datasets.val_ar_dataset import VADataset
from src.process_data.process_val_ar import process_val_ar_data


def load_module_class(attn_type: str):
    module_map = {
        "self":  "src.models.va_self",
        "cross": "src.models.val_ar_model",
    }
    mod = importlib.import_module(module_map[attn_type])
    return mod.DualModule


def calculate_metrics(v_preds, a_preds, v_labels, a_labels):
    v_preds  = torch.stack(v_preds).float()
    a_preds  = torch.stack(a_preds).float()
    v_labels = torch.stack(v_labels).float()
    a_labels = torch.stack(a_labels).float()

    v_mae = (v_preds - v_labels).abs().mean().item()
    a_mae = (a_preds - a_labels).abs().mean().item()
    v_mse = ((v_preds - v_labels) ** 2).mean().item()
    a_mse = ((a_preds - a_labels) ** 2).mean().item()

    return {
        "valence_mae": v_mae,
        "arousal_mae": a_mae,
        "valence_mse": v_mse,
        "arousal_mse": a_mse,
        "n": len(v_preds),
    }


def train(
    train: str,
    out_path: str,
    model_path: str,
    attn_layers: list[int],
    attn_type: str,
    test: str = None,
    pretrained_weights: str = None,
    epochs: int = 3,
    domain: str = "",
):
    loss_file  = f"{out_path}/log_files/{attn_type}_loss_va.jsonl"
    preds_file = f"{out_path}/predictions/{attn_type}_va_predictions.jsonl"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    DualModule = load_module_class(attn_type)

    # Training data
    train_path = process_val_ar_data(train, f"{out_path}/processed/val_ar_processed.jsonl")
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            for elem in temp['Quadruplet']:
                train_data.append({
                    'ID':      temp['ID'],
                    'Text':    temp['Text'],
                    'Aspect':  elem['Aspect'],
                    'Opinion': elem['Opinion'],
                    'Cat1':    elem['Cat1'],
                    'Cat2':    elem['Cat2'],
                    'Valence': elem['Valence'],
                    'Arousal': elem['Arousal'],
                })

    train_loader = DataLoader(VADataset(train_data, tokenizer), batch_size=16, shuffle=False)

    # Test data
    test_loader = None
    if test is not None:
        test_data = []
        test_path = process_val_ar_data(test, f"{out_path}/processed/val_ar_processed.jsonl")
        # Check if labels are available or not
        with open(test) as f:
            has_labels = 'Quadruplet' in json.loads(f.readline())

        with open(test_path, 'r') as f:
            for line in f:
                temp = json.loads(line)
                if has_labels:
                    for elem in temp['Quadruplet']:
                        test_data.append({
                            'ID':      temp['ID'],
                            'Text':    temp['Text'],
                            'Aspect':  elem['Aspect'],
                            'Opinion': elem['Opinion'],
                            'Cat1':    elem['Cat1'],
                            'Cat2':    elem['Cat2'],
                            'Valence': elem['Valence'],
                            'Arousal': elem['Arousal'],
                        })
                else:
                    test_data.append({
                        'ID':      temp['ID'],
                        'Text':    temp['Text'],
                        'Aspect':  temp['Aspect'],
                        'Opinion': temp['Opinion'],
                        'Cat1':    temp['Cat1'],
                        'Cat2':    temp['Cat2'],
                    })
                    
        test_loader = DataLoader(VADataset(test_data, tokenizer), batch_size=8, shuffle=False)


    model = DualModule(model_path, model_path, attn_layers=attn_layers)
    model.to(device)

    # Load pretrained weights (optional)
    if pretrained_weights is not None:
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
                    valence        = batch['valence'].to(device)
                    arousal        = batch['arousal'].to(device)

                    optimizer.zero_grad()

                    loss = model(input_ids, attention_mask=attention_mask, gold1=valence, gold2=arousal)
                    print(loss)
                    l.write(str(loss.item()) + '\n')

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{out_path}/trained_models/val_ar_model_{domain}.pt")

    # Evaluation
    if test_loader is not None:
        model.eval()
        all_v_preds, all_a_preds, all_v_labels, all_a_labels = [], [], [], []

        with torch.no_grad(), open(preds_file, 'w') as f:
            for batch in test_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ids            = batch['ID']

                v_preds, a_preds = model(input_ids, attention_mask=attention_mask)

                batch_output = [
                    {
                        "ID":      id_,
                        "Text":    text,
                        "Aspect":  asp,
                        "Opinion": op,
                        "Cat1":    cat1,
                        "Cat2":    cat2,
                        "Valence": v.item(),
                        "Arousal": a.item(),
                    }
                    for id_, text, asp, op, cat1, cat2, v, a in zip(
                        ids, batch['text'], batch['aspect'], batch['opinion'],
                        batch['cat1'], batch['cat2'], v_preds, a_preds
                    )
                ]

                for d in batch_output:
                    json.dump(d, f)
                    f.write('\n')

                all_v_preds.extend(v_preds); all_a_preds.extend(a_preds)

                if 'valence' in batch:
                    all_v_labels.extend(batch['valence'])
                    all_a_labels.extend(batch['arousal'])

        if all_v_labels:
            metrics = calculate_metrics(all_v_preds, all_a_preds, all_v_labels, all_a_labels)
            print(f"Test | Valence MAE: {metrics['valence_mae']:.4f}  Arousal MAE: {metrics['arousal_mae']:.4f}  "
                  f"Valence MSE: {metrics['valence_mse']:.4f}  Arousal MSE: {metrics['arousal_mse']:.4f}  "
                  f"N: {metrics['n']}")