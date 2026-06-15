import json
import argparse

def combine_data(data_list):
    grouped_data = {}
    for item in data_list:
        item_id = item['ID']
        if item_id not in grouped_data:
            grouped_data[item_id] = {
                "ID": item_id,
                "Quadruplet": []
            }
        quad = {
            "Aspect": item['Aspect'],
            "Category": f"{item['Cat1']}#{item['Cat2']}",
            "Opinion": item['Opinion'],
            "VA": f"{item['Valence']:.2f}#{item['Arousal']:.2f}"
        }
        grouped_data[item_id]["Quadruplet"].append(quad)
    return list(grouped_data.values())

def main():
    parser = argparse.ArgumentParser(description="Format predictions back into the original data format.")
    parser.add_argument("input", help="Path to the input .jsonl predictions file")
    parser.add_argument("output", help="Path to the output .jsonl file")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data_raw = [json.loads(line) for line in f]

    result = combine_data(data_raw)

    with open(args.output, 'w') as f_out:
        for res in result:
            json.dump(res, f_out)
            f_out.write('\n')

if __name__ == "__main__":
    main()