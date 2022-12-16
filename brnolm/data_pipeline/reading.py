import torch


def word_splitter(line):
    return line.split()


def char_splitter(line, sentence_end_token=None):
    chars = list(line)

    if sentence_end_token is None:
        return chars
    else:
        return chars + [sentence_end_token]


class WordIdProvider:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return [self.vocab[w] for w in text.split()]


class WordIdLineEndProvider:
    def __init__(self, vocab, line_end='</s>'):
        self.vocab = vocab
        self.line_end = line_end

    def __call__(self, text):
        return [self.vocab[w] for w in text.split() + [self.line_end]]


class CharIdProvider:
    def __init__(self, vocab, sentence_end_token='</s>'):
        self.vocab = vocab
        self.sentence_end_token = sentence_end_token

    def __call__(self, text):
        chars = [self.sentence_end_token if c == '\n' else c for c in text]
        return [self.vocab[c] for c in chars]


class TokenizerFactory:
    tokenize_regimes = {
        'words': WordIdProvider,
        'words-lines': WordIdLineEndProvider,
        'chars': CharIdProvider,
    }

    def construct_tokenizer(self, regime, vocab):
        if regime in self.tokenize_regimes:
            return self.tokenize_regimes[regime](vocab)
        else:
            raise ValueError(f'Unsupported tokenization regime {regime}')

tokenizer_factory = TokenizerFactory()


def tokens_from_file(f, vocab, randomize, regime='words'):
    ids = []

    lines = f.read().split('\n')

    if regime == 'words':
        tokenizer = word_splitter
    elif regime == 'chars':
        tokenizer = lambda line: char_splitter(line, '</s>')
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


def get_independent_lines(f, vocab):
    lines = []
    for line in f:
        words = line.split()
        if words:
            lines.append(torch.tensor([vocab[w] for w in words]))

    return lines
