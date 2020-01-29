from typing import List
import torch


def masked_tensor_from_sentences(sentences: List[List[int]]):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")

    batch_size = len(sentences)

    input = torch.zeros((batch_size, len(sentences[0])-1), dtype=torch.int32)
    target = torch.zeros((batch_size, len(sentences[0])-1), dtype=torch.int32)
    mask = torch.zeros((batch_size, len(sentences[0])-1), dtype=torch.int32)

    for s in range(len(sentences)):
        for t in range(len(sentences[s]) - 1):
            input[s, t] = sentences[s][t]
            target[s, t] = sentences[s][t+1]
            mask[s, t] = 1

    return input, target, mask
