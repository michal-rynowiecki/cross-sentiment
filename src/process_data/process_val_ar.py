import json
from collections import Counter


def process_val_ar_data(input_file: str, output_file: str) -> str:
    cat1_counter = Counter()
    cat2_counter = Counter()

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line)

            for quad in data.get("Quadruplet", []):
                cat1, cat2       = quad.pop("Category").split("#", 1)
                valence, arousal = quad.pop("VA").split("#", 1)

                quad["Cat1"]    = cat1
                quad["Cat2"]    = cat2
                quad["Valence"] = valence
                quad["Arousal"] = arousal

                cat1_counter[cat1] += 1
                cat2_counter[cat2] += 1

            f_out.write(json.dumps(data) + '\n')

    return output_file