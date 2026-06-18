import json

import json

# tag_type is either "Aspect" or "Opinion"
# train indicates whether the source file is the training file or the dev file
def create_BIO_tags(path: str, output_path: str, tag_type: str, train=True):

    out_path = f"{output_path}/bio_tagged_{tag_type}.jsonl"

    with open(path, "r", encoding="utf-8") as file, \
         open(out_path, "w", encoding="utf-8") as file_out:

        for line in file:
            read = json.loads(line)

            text = read["Text"].split()
            sample_id = read["ID"]

            if train:
                flattened = [i[tag_type].split() for i in read["Quadruplet"]]
            else:
                # TODO: Add reading in data for dev set
                flattened = []

            BIO2 = ["O"] * len(text)

            for sub_aspect in flattened:
                len_aspect = len(sub_aspect)

                # Slide over text to find matching spans
                for i in range(len(text) - len_aspect + 1):
                    if text[i:i + len_aspect] == sub_aspect:
                        if BIO2[i] == "O":
                            BIO2[i] = "B-Asp"
                            for j in range(1, len_aspect):
                                BIO2[i + j] = "I-Asp"

            file_out.write(f"{sample_id}\n")
            for w, label in zip(text, BIO2):
                file_out.write(f"{w}\t{label}\n")

            file_out.write("\n")

    return out_path