# Dimensional Aspect-Based Sentiment Analysis for SemEval 2026 Task 3

This repository contains our implementation for **SemEval 2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DABSA)**.

The task consists of:

1. Recognizing **aspect–opinion pairs**.
2. Categorizing each aspect–opinion pair.
3. Assigning **Valence** and **Arousal** scores on a scale from **1–9**.

## Method Overview

Our system uses a set of encoder architectures with **cross-attention** modules for each sub-task:

- **Aspect–Opinion Recognition**
- **Category Classification**
- **Valence–Arousal (VA) Prediction**

After aspect–opinion recognition, we introduce an additional **filter encoder** that removes irrelevant aspect–opinion pairs before passing them to subsequent modules.

A more detailed description of the architecture and experiments can be found in the paper.

---

## Citation

If you use this code or build upon our work, please cite the following paper:

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {Proceedings of SemEval 2026},
  year      = {2026}
}
```

---

## Reproducing Results

To reproduce the experimental results, run the scripts in the `scripts/` directory in numerical order, starting with the aspect–opinion recognition script.

Example:

```bash
python 1.asp_op.py \
  --train eng_laptop_train_alltasks.jsonl \
  --out-path cross-sentiment/data \
  --model-path google-bert/bert-base-cased \
  --attn-type cross \
  --attn-layers 4 7 \
  --test eng/eng_laptop_test_task3.jsonl \
  --epochs 3
```

Subsequent scripts should be executed in the same order indicated by their numeric prefixes. After training, the outputs of the previous steps should be used as the test data for the subsequent models.

---

## Repository Structure

```text
.
├── data/
├── scripts/
└── src/
    ├── datasets/
    │   ├── asp_op_dataset.py
    │   ├── bin_dataset.py
    │   └── ...
    │
    ├── models/
    │   ├── asp_op_model.py
    │   └── ...
    │
    ├── process_data/
    │   ├── process_asp_op_data.py
    │   └── ...
    │
    ├── utils/
    │
    ├── asp_op.py
    ├── bin.py
    ├── cat.py
    ├── val_ar.py
    └── utils/
```


## Data

The original data used in this work comes from **SemEval 2026 Task 3**.

However, the framework is designed to be data-agnostic: if your dataset follows the same input format as the SemEval data, the pipeline should work without major modifications.

---

## Requirements

The implementation relies on PyTorch and Hugging Face Transformers.

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

Please refer to the SemEval data license for restrictions regarding dataset redistribution. Code licensing information should be added here.

---