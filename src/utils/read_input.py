import json

def read_data(path: str):
    file = open(path, 'r')

    data = []
    for item in file.readlines():
        my_dict = json.loads(item)
        data.append(my_dict)

    print(data)
    return data

def read_conll(path: str):
    sentences = []
    total_sentence = {}
    current_sentence = []

    f = open(path, 'r', encoding='utf-8')

    for line in f:
        line = line.strip()

        if not line:
            if current_sentence:
                total_sentence['Tags'] = current_sentence
                sentences.append(total_sentence)
                total_sentence={}
                current_sentence = []
            continue

        if len(line.split()) == 1:
            total_sentence['ID'] = line
        else:
            word, tag = line.split()
            current_sentence.append((word, tag))
    print(sentences)
    return sentences