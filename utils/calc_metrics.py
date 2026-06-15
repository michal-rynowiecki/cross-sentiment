# This file takes the predictions and calculates the f1 score per each stage against the gold labels file
import json
import argparse
import math


def load_jsonl(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            data[entry['ID']] = entry
    return data


def normalize(s):
    return s.strip().lower()


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def parse_va(va_str):
    parts = va_str.split('#', 1)
    return float(parts[0]), float(parts[1])


def va_dist(vp, ap, vg, ag, d_max):
    return math.sqrt((vp - vg) ** 2 + (ap - ag) ** 2) / d_max


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold labels.")
    parser.add_argument("gold", help="Path to the gold .jsonl file")
    parser.add_argument("pred", help="Path to the predictions .jsonl file")
    parser.add_argument(
        "--va-min", type=float, default=1.0,
        help="Minimum value of the VA scale (default: 1.0)"
    )
    parser.add_argument(
        "--va-max", type=float, default=9.0,
        help="Maximum value of the VA scale (default: 9.0)"
    )
    args = parser.parse_args()

    va_range = args.va_max - args.va_min
    d_max = math.sqrt(2) * va_range

    gold_data = load_jsonl(args.gold)
    pred_data = load_jsonl(args.pred)

    ao_tp = ao_fp = ao_fn = 0
    cat1_correct = cat1_total = 0
    cat2_correct = cat2_total = 0

    va_distances = []
    v_abs_errors = []
    a_abs_errors = []

    missing_ids = []

    for gold_id, gold_entry in gold_data.items():
        gold_ao = {}
        for q in gold_entry['Quadruplet']:
            key = (normalize(q['Aspect']), normalize(q['Opinion']))
            gold_ao[key] = q

        if gold_id not in pred_data:
            missing_ids.append(gold_id)
            ao_fn += len(gold_ao)
            continue

        pred_ao = {}
        for q in pred_data[gold_id]['Quadruplet']:
            key = (normalize(q['Aspect']), normalize(q['Opinion']))
            pred_ao[key] = q

        gold_keys = set(gold_ao)
        pred_keys = set(pred_ao)
        matched   = gold_keys & pred_keys

        ao_tp += len(matched)
        ao_fp += len(pred_keys - gold_keys)
        ao_fn += len(gold_keys - pred_keys)

        for key in matched:
            # Category
            g_cat = gold_ao[key]['Category'].split('#', 1)
            p_cat = pred_ao[key]['Category'].split('#', 1)

            g_cat1 = normalize(g_cat[0]) if len(g_cat) > 0 else ''
            g_cat2 = normalize(g_cat[1]) if len(g_cat) > 1 else ''
            p_cat1 = normalize(p_cat[0]) if len(p_cat) > 0 else ''
            p_cat2 = normalize(p_cat[1]) if len(p_cat) > 1 else ''

            cat1_total   += 1
            cat1_correct += int(g_cat1 == p_cat1)
            cat2_total   += 1
            cat2_correct += int(g_cat2 == p_cat2)

            # VA
            try:
                vg, ag = parse_va(gold_ao[key]['VA'])
                vp, ap = parse_va(pred_ao[key]['VA'])

                dist = va_dist(vp, ap, vg, ag, d_max)
                va_distances.append(dist)
                v_abs_errors.append(abs(vp - vg))
                a_abs_errors.append(abs(ap - ag))
            except (ValueError, KeyError) as e:
                print(f"  Warning: could not parse VA for {gold_id} {key}: {e}")

    ao_p, ao_r, ao_f1 = compute_f1(ao_tp, ao_fp, ao_fn)

    # --- matched-only metrics ---
    n_va          = len(va_distances)
    cat1_acc      = cat1_correct / cat1_total if cat1_total > 0 else 0.0
    cat2_acc      = cat2_correct / cat2_total if cat2_total > 0 else 0.0
    mean_dist     = sum(va_distances) / n_va if n_va > 0 else 0.0
    mean_v_mae    = sum(v_abs_errors) / n_va if n_va > 0 else 0.0
    mean_a_mae    = sum(a_abs_errors) / n_va if n_va > 0 else 0.0

    # --- penalised metrics (missing pairs count against you) ---
    total_gold         = cat1_total + ao_fn          # all gold pairs
    pen_cat1_acc       = cat1_correct / total_gold if total_gold > 0 else 0.0
    pen_cat2_acc       = cat2_correct / total_gold if total_gold > 0 else 0.0

    pen_va_distances   = va_distances + [1.0] * ao_fn
    n_va_pen           = len(pen_va_distances)
    pen_mean_dist      = sum(pen_va_distances) / n_va_pen if n_va_pen > 0 else 0.0

    print("=== Aspect–Opinion Pair Detection ===")
    print(f"  TP — correctly detected:  {ao_tp}")
    print(f"  FP — detected, shouldn't: {ao_fp}")
    print(f"  FN — missed entirely:     {ao_fn}")
    print(f"  Precision: {ao_p:.4f}")
    print(f"  Recall:    {ao_r:.4f}")
    print(f"  F1:        {ao_f1:.4f}")

    print(f"\n=== Category Accuracy ===")
    print(f"  Matched pairs only ({cat1_total} pairs):")
    print(f"    Entity    (part 1): {cat1_acc:.4f}  ({cat1_correct}/{cat1_total} correct)")
    print(f"    Attribute (part 2): {cat2_acc:.4f}  ({cat2_correct}/{cat2_total} correct)")
    print(f"  Penalised — missing pairs scored 0 ({total_gold} pairs):")
    print(f"    Entity    (part 1): {pen_cat1_acc:.4f}  ({cat1_correct}/{total_gold} correct)")
    print(f"    Attribute (part 2): {pen_cat2_acc:.4f}  ({cat2_correct}/{total_gold} correct)")

    print(f"\n=== Valence–Arousal Distance ===")
    print(f"  Scale: [{args.va_min}, {args.va_max}]  →  Dmax = √2 × {va_range} = {d_max:.4f}")
    print(f"  Matched pairs only ({n_va} pairs):")
    print(f"    Mean normalised Euclidean dist: {1 - mean_dist:.4f}  (1 = perfect, 0 = worst)")
    print(f"    Valence MAE: {mean_v_mae:.4f}")
    print(f"    Arousal MAE: {mean_a_mae:.4f}")
    print(f"  Penalised — missing pairs scored 1.0 ({n_va_pen} pairs):")
    print(f"    Mean normalised Euclidean dist: {1- pen_mean_dist:.4f}  (1 = perfect, 0 = worst)")

    if missing_ids:
        print(f"\n=== {len(missing_ids)} gold ID(s) not found in predictions ===")
        for mid in missing_ids:
            print(f"  {mid}")


if __name__ == "__main__":
    main()