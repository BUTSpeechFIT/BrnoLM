import numpy as np
from typing import Dict, List


def emb_line_iterator(f):
    for line in f:
        fields = line.split()
        key = fields[0]
        embedding = np.asarray([float(e) for e in fields[1:]])

        yield key, embedding


def all_embs_from_file(f):
    embs = []
    keys = []

    for key, emb in emb_line_iterator(f):
        embs.append(emb)
        keys.append(key)

    return keys, np.stack(embs)


def str_from_embedding(emb):
    return " ".join(["{:.4f}".format(e) for e in emb])


def all_embs_by_key(f, shall_be_collected=lambda w: True, key_transform=lambda w: w):
    collection: Dict[str, List[np.ndarray]] = {}

    for word, emb in emb_line_iterator(f):
        word = key_transform(word)

        if not shall_be_collected(word):
            continue

        if word in collection:
            collection[word].append(emb)
        else:
            collection[word] = [emb]

    for w in collection:
        collection[w] = np.stack(collection[w])

    return collection
