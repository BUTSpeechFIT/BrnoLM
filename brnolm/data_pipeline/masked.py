from typing import List
import torch


def masked_tensor_from_sentences(sentences: List[List[int]], filler=0):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")

    batch_size = len(sentences)
    max_len = max(len(s) for s in sentences)

    shape = (batch_size, max_len-1)
    input = torch.zeros(shape, dtype=torch.int32)
    target = torch.zeros(shape, dtype=torch.int32)
    mask = torch.zeros(shape, dtype=torch.int32)

    for s in range(len(sentences)):
        for t in range(len(sentences[s]) - 1):
            input[s, t] = sentences[s][t]
            target[s, t] = sentences[s][t+1]
            mask[s, t] = 1

    return input, target, mask
