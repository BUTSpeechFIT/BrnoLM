from typing import List
import torch


def masked_tensor_from_sentences(sentences: List[List[int]]):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")

    input = torch.zeros((1, 1), dtype=torch.int32)
    target = torch.zeros((1, 1), dtype=torch.int32)
    mask = torch.zeros((1, 1), dtype=torch.int32)

    input[0, 0] = sentences[0][0]
    target[0, 0] = sentences[0][1]
    mask[0, 0] = 1

    return input, target, mask
