import importlib
import json

import torch
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.utils.read_input import read_conll
from src.utils.transform_tokens import get_entities_batch

from src.process_data.process_asp_op import create_BIO_tags

from src.datasets.asp_op_dataset import BIODatasetDouble, BIODatasetInference


def load_module_class(attn_type: str):
    module_map = {
        "self":  "src.models.AO_self",
        "cross": "src.models.asp_op_model",
    }
    mod = importlib.import_module(module_map[attn_type])
    return mod.DualModule


def calculate_metrics(predictions1, predictions2, labels1, labels2, attention_mask):
    tp = fp = fn = n = 0

    for p1, p2, l1, l2, mask in zip(predictions1, predictions2, labels1, labels2, attention_mask):
        print(p1)
        print(l1)
        for i in range(len(l1)):
            if mask[i] != 1:
                continue

            for pred, label in [(p1[i], l1[i]), (p2[i], l2[i])]:
                gold_is_entity = label != 0
                pred_is_entity = pred  != 0

                if gold_is_entity:
                    n += 1

                if pred_is_entity and gold_is_entity:
                    if pred == label:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_is_entity and not gold_is_entity:
                    fp += 1
                elif not pred_is_entity and gold_is_entity:
                    fn += 1

    if tp + fp == 0 or tp + fn == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": n}

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1        = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1, "n": n}


def train(
    train: str,
    out_path: str,
    model_path: str,
    attn_layers: list[int],
    attn_type: str,        # "self" or "cross"
    test: str = None,      # if provided, evaluate on this file after training
    pretrained_weights: str = None,
    epochs: int = 3,
    domain: str = "",
):
    tag_to_id = {"O": 0, "B-Asp": 1, "I-Asp": 2}

    loss_file = f"{out_path}/log_files/{attn_type}_loss_ao.jsonl"
    preds_file = f"{out_path}/predictions/{attn_type}_predictions.jsonl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DualModule = load_module_class(attn_type)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    module = DualModule(model_path, model_path, 3, attn_layers=attn_layers)
    module.to(device)

    # Training data
    path1 = create_BIO_tags(train, f"{out_path}/processed", "Aspect", True)
    path2 = create_BIO_tags(train, f"{out_path}/processed", "Opinion", True)
    train_dataset = BIODatasetDouble(read_conll(path1), read_conll(path2), tokenizer, tag_to_id)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=False)

    # Test data
    if test is not None:
        with open(test) as f:
            has_labels = 'Quadruplet' in json.loads(f.readline())

        if has_labels:
            path1 = create_BIO_tags(test, f"{out_path}/processed", "Aspect", True)
            path2 = create_BIO_tags(test, f"{out_path}/processed", "Opinion", True)
            test_dataset = BIODatasetDouble(read_conll(path1), read_conll(path2), tokenizer, tag_to_id)
        else:
            test_data = [json.loads(line) for line in open(test)]
            test_dataset = BIODatasetInference(test_data, tokenizer)

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # Load pretrained weights (optional)
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location=device)
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        module.load_state_dict(state_dict)
        print("Loaded weights: skipping training")
    else:
        # Training loop
        optimizer = optim.AdamW(module.parameters(), lr=5e-5)

        with open(loss_file, 'w') as l:
            module.train()
            for epoch in range(epochs):
                total_loss = 0

                for batch in train_loader:
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels1        = batch['labels1'].to(device)
                    labels2        = batch['labels2'].to(device)

                    optimizer.zero_grad()

                    loss = module(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)
                    print(loss)
                    l.write(str(loss.item()) + '\n')

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
        
        torch.save(module.state_dict(), f"{out_path}/trained_models/asp_op_model_{domain}.pt")


    # Evaluation
    if test is not None:
        module.eval()
        all_p1, all_p2, all_l1, all_l2, all_mask = [], [], [], [], []

        with torch.no_grad(), open(preds_file, 'w') as f:
            for batch in test_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ids            = batch['ID']['ID']

                predictions1, predictions2 = module(input_ids, attention_mask=attention_mask)
                sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                if has_labels:
                    labels1        = batch['labels1'].to(device)
                    labels2        = batch['labels2'].to(device)
                    aspect_pred, aspect_label = get_entities_batch(tokenizer, input_ids, predictions1, labels1, False)
                    opinion_pred, opinion_label = get_entities_batch(tokenizer, input_ids, predictions2, labels2, False)

                    batch_output = [
                        {
                            "ID":                    id,
                            "sentence":              sent,
                            "aspect_predicted_tags": asp_t,
                            "aspect_gold_labels":    asp_g,
                            "opinion_predicted_tags": op_t,
                            "opinion_gold_labels":   op_g,
                        }
                        for id, sent, asp_t, asp_g, op_t, op_g in zip(
                            ids, sentences, aspect_pred, aspect_label, opinion_pred, opinion_label
                        )
                    ]

                    all_p1.extend(predictions1); all_p2.extend(predictions2)
                    all_l1.extend(labels1);      all_l2.extend(labels2)
                    all_mask.extend(attention_mask)
                else:
                    aspect_pred  = get_entities_batch(tokenizer, input_ids, predictions1, None, True)
                    opinion_pred = get_entities_batch(tokenizer, input_ids, predictions2, None, True)
                    batch_output = [
                        {
                            "ID":                     id,
                            "sentence":               sent,
                            "aspect_predicted_tags":  asp_t,
                            "opinion_predicted_tags": op_t,
                        }
                        for id, sent, asp_t, op_t in zip(
                            ids, sentences, aspect_pred, opinion_pred
                        )
                    ]
    
                for d in batch_output:
                    json.dump(d, f)
                    f.write('\n')
  
        if has_labels:
            metrics = calculate_metrics(all_p1, all_p2, all_l1, all_l2, all_mask)
            print(f"Test | P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}  "
                f"F1: {metrics['f1']:.4f}  N: {metrics['n']}")