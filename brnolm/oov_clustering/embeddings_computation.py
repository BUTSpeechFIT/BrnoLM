import torch


def tensor_from_words(words, vocab):
    return torch.tensor([vocab[w] for w in words]).view(1, -1)
