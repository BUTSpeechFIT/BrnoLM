from typing import List
import torch


def masked_tensor_from_sentences(sentences: List[List[int]], filler=0, device=torch.device('cpu'), target_all=False):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")

    batch_size = len(sentences)
    max_len = max(len(s) for s in sentences)

    shape = (batch_size, max_len-1)
    dtype = torch.int64
    input = torch.zeros(shape, dtype=dtype, device=device)
    target = torch.zeros(shape, dtype=dtype, device=device)
    mask = torch.zeros(shape, dtype=dtype, device=device)

    for s in range(len(sentences)):
        for t in range(len(sentences[s]) - 1):
            input[s, t] = sentences[s][t]
            target[s, t] = sentences[s][t+1]
            mask[s, t] = 1

    if target_all:
        first_inputs = torch.tensor([s[0] for s in sentences], dtype=dtype, device=device)
        target = torch.cat([first_inputs.view(-1, 1), target], dim=1)

        batch_of_ones = torch.ones((batch_size, 1), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([batch_of_ones, mask], dim=1)

    if input.shape[1] == 0:
        batch_of_zeros = torch.zeros((batch_size, 1), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, batch_of_zeros], dim=1)
        input = batch_of_zeros.clone().detach()
        target = torch.cat([target, batch_of_zeros], dim=1)

    return input, target, mask
