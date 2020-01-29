from typing import List
import torch


def masked_tensor_from_sentences(sentences: List[List[int]]):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")

    input = torch.zeros((1, len(sentences[0])-1), dtype=torch.int32)
    target = torch.zeros((1, len(sentences[0])-1), dtype=torch.int32)
    mask = torch.zeros((1, len(sentences[0])-1), dtype=torch.int32)

    for i in range(len(sentences[0]) - 1):
        input[0, i] = sentences[0][i]
        target[0, i] = sentences[0][i+1]
        mask[0, i] = 1

    return input, target, mask
