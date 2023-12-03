import json
import gzip


def load_dataset(path):
    print('Loading dataset from {}...'.format(path))
    with gzip.open(path, 'rt', encoding='utf-8') as gzip_file:
        for line in gzip_file:
            data = json.loads(line)
    
    # if there is an image attribute in each element of the data, remove it
    for i in range(len(data)):
        if 'image' in data[i].keys():
            del data[i]['image']

    return data
