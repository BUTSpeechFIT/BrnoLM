from collections import Mapping
import re


class IndexGenerator():
    def __init__(self, assigned):
        self.next_ = 0
        self.assigned_ = assigned

    def next(self):
        while self.next_ in self.assigned_:
            self.next_ += 1

        retval = self.next_
        self.next_ += 1
        return retval


class Vocabulary(Mapping):
    def __init__(self, unk_word, unk_index):
        self.w2i_ = {unk_word: unk_index}
        self.i2w_ = {unk_index: unk_word}
        self.unk_index_ = unk_index
        self.unk_word_ = unk_word
        self.ind_gen_ = IndexGenerator([unk_index])

    def add_from_text(self, text):
        assert self.ind_gen_

        words = text.split()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.w2i_:
            index = self.ind_gen_.next()
            self.w2i_[word] = index
            self.i2w_[index] = word
        else:
            pass  # do not do anything for known words

    def is_continuous(self):
        return max(self.i2w_.keys()) == len(self) - 1

    def missing_indexes(self):
        return [i for i in range(len(self)) if i not in self.i2w_]

    def w2i(self, word):
        return self.w2i_.get(word, self.unk_index_)

    @property
    def unk_word(self):
        return self.unk_word_

    @property
    def unk_ind(self):
        return self.unk_index_

    def __getitem__(self, idx):
        return self.w2i(idx)

    def i2w(self, index):
        return self.i2w_[index]

    def __len__(self):
        return len(self.w2i_)

    def __iter__(self):
        return iter(self.w2i_)


def vocab_from_kaldi_wordlist_base(f, unk_word, word_re, remove_quotes):
    d = {}
    line_re = re.compile('\s*(?P<word>' + word_re + ')\s+(?P<ind>[0-9]+)\s*\n?')
    for i, line in enumerate(f):
        m = line_re.fullmatch(line)

        if m is None:
            raise ValueError("Weird line {}: '{}'".format(i, line))

        w = m.group('word')
        if remove_quotes:
            assert w[0] == w[-1] == "'"
            w = w[1:-1]
        i = int(m.group('ind'))
        assert i >= 0

        if w in d:
            raise ValueError(f'Attempt to redefince "{w}" to {i}, while it already has {d[w]} assigned')
        d[w] = i

    try:
        vocab = Vocabulary(unk_word, d[unk_word])
    except KeyError:
        raise ValueError("Unk word {} not present in the kaldi wordlist!".format(unk_word))
    vocab.ind_gen_ = None
    vocab.w2i_ = d
    for w in d:
        vocab.i2w_[d[w]] = w

    vocab.size_ = max(d.values()) + 1

    return vocab


def vocab_from_kaldi_wordlist(f, unk_word='<unk>'):
    return vocab_from_kaldi_wordlist_base(f, unk_word, word_re='\S+', remove_quotes=False)


def quoted_vocab_from_kaldi_wordlist(f, unk_word='<unk>'):
    return vocab_from_kaldi_wordlist_base(f, unk_word, word_re="'.+'", remove_quotes=True)
