import json
from random import random, shuffle

def create_binary_data(in_path: str, out_path: str, inference: bool = False) -> str:
    data = []

    with open(in_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            inference = not 'Quadruplet' in temp

            if not inference:
                valid_pairs = {
                    (quad['Aspect'], quad['Opinion'])
                    for quad in temp['Quadruplet']
                }
                aspects  = {quad['Aspect']  for quad in temp['Quadruplet']}
                opinions = {quad['Opinion'] for quad in temp['Quadruplet']}

                for aspect in aspects:
                    for opinion in opinions:
                        flag = int((aspect, opinion) in valid_pairs)
                        data.append({
                            'ID':      temp['ID'],
                            'Text':    temp['Text'],
                            'Aspect':  aspect,
                            'Opinion': opinion,
                            'exists':  int(flag),
                        })
            else:
                aspects  = set(temp['aspect_predicted_tags'])
                opinions = set(temp['opinion_predicted_tags'])

                for aspect in aspects:
                    for opinion in opinions:
                        data.append({
                            'ID':      temp['ID'],
                            'Text':    temp['sentence'],
                            'Aspect':  aspect.replace('\u2581', ''),
                            'Opinion': opinion.replace('\u2581', ''),
                        })

    if not inference:
        shuffle(data)

    with open(out_path, 'w') as f_out:
        for item in data:
            json.dump(item, f_out)
            f_out.write('\n')

    return out_path