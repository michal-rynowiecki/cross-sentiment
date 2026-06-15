import json
from collections import Counter

def split_json_categories(input_file: str, output_file: str) -> str:
    cat1_counter = Counter()
    cat2_counter = Counter()

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line)

            for quad in data.get("Quadruplet", []):
                cat1, cat2 = quad.pop("Category").split("#", 1)
                quad["Cat1"] = cat1
                quad["Cat2"] = cat2
                cat1_counter[cat1] += 1
                cat2_counter[cat2] += 1

            f_out.write(json.dumps(data) + '\n')

    print("### Cat1 (Entity) Counts ###")
    for cat, count in cat1_counter.items():
        print(f"  {cat}: {count}")

    print("\n### Cat2 (Attribute) Counts ###")
    for cat, count in cat2_counter.items():
        print(f"  {cat}: {count}")

    return output_file