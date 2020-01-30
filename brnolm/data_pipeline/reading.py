import torch


def word_splitter(line):
    return line.split()


def char_splitter(line, sentence_end_token=None):
    chars = list(line)

    if sentence_end_token is None:
        return chars
    else:
        return chars + [sentence_end_token]


def tokens_from_file(f, vocab, randomize, regime='words'):
    ids = []

    lines = f.read().split('\n')

    if regime == 'words':
        tokenizer = word_splitter
    elif regime == 'chars':
        tokenizer = lambda line: char_splitter(line, '<sb>')
    else:
        raise ValueError("unsupported regime {}".format(regime))

    if randomize:
        import random
        random.shuffle(lines)

    for line in lines:
        tokens = tokenizer(line)
        ids.extend([vocab[e] for e in tokens])

    return torch.LongTensor(ids)


def tokens_from_fn(fn, vocab, randomize, regime='words'):
    with open(fn, 'r') as f:
        return tokens_from_file(f, vocab, randomize, regime)
