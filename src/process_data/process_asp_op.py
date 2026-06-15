import json

# type is either Aspect or Opinion
# train is if the source file is the training file or the dev file
def create_BIO_tags(path: str, output_path: str, type: str, train=True):

    file = open(path, 'r')
    out_path = output_path + f"/bio_tagged_{type}.jsonl"
    file_out = open(out_path, 'w')

    for line in file.readlines():
        
        read = json.loads(line)

        text = read['Text'].split()
        id = read['ID']
        
        if train:
            flattened = [i[type].split() for i in read['Quadruplet']]
        else:
            1
            # TODO Add reading in data for 
            
        BIO2 = ['O'] * len(text)

        for sub_aspect in flattened:
            len_aspect = len(sub_aspect)

            # Slide over text to find matching spans (subtract the length of the aspect because the aspect can't start if there isn't enough words in the text to match with the words in the aspect)
            for i in range(len(text) - len_aspect + 1):
                if text[i:i+len_aspect] == sub_aspect:
                    # If it isn't marked by a previous aspect
                    if BIO2[i]=='O':
                        # First word -> B, rest -> I
                        BIO2[i] = 'B-Asp'
                        for j in range(1, len_aspect):
                            BIO2[i+j] = 'I-Asp'
        
        # Print result
        file_out.write(f"{id}\n")
        for w, label in zip(text, BIO2):
            file_out.write(f"{w}\t{label}\n")

        file_out.write("\n")

    return out_path