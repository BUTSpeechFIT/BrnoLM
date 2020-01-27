import io
import tempfile
import pickle

import numpy as np
import torch

from smm import update_ws


class IvecExtractor():
    def __init__(self, model, nb_iters, lr, tokenizer):
        self._model = model
        self._nb_iters = nb_iters
        self._lr = lr
        self._tokenizer = tokenizer

    def __call__(self, sentence):
        """ Extract i-vectors given the model and stats """
        if isinstance(sentence, str):
            data = self._tokenizer.transform([sentence])
            data = torch.from_numpy(data.A.astype(np.float32))

        else:
            data = sentence

        if self._model.cuda:
            data = data.cuda()
        X = data.t()

        self._model.reset_w(X.size(-1))  # initialize i-vectors to zeros
        opt_w = torch.optim.Adagrad([self._model.W], lr=self._lr)

        loss = self._model.loss(X)

        # TODO this is a very nasty hack, needs to be
        # completely reworked, a separate class should be
        # prepared to implement this
        if self._nb_iters < 0:
            initrange = 10**(self._nb_iters)
            self._model.W.data.uniform_(-initrange, initrange)
        else:
            for i in range(self._nb_iters):
                loss = update_ws(self._model, opt_w, loss, X)

        return self._model.W.data.t().squeeze()

    def __str__(self):
        name = "IvecExtractor"
        ivec_size = self._model.W.size(0)

        fmt_str = "{} (\n\tiVectors size: {}\n\tLearning rate: {}\n\t # iterations: {}\n)\n"
        return fmt_str.format(name, ivec_size, self._lr, self._nb_iters)

    def save(self, f):
        tmp_f = tempfile.TemporaryFile()
        # self._model.cpu()
        torch.save(self._model, tmp_f)
        tmp_f.seek(0)
        model_bytes = io.BytesIO(tmp_f.read())

        nb_iters_bytes = io.BytesIO()
        pickle.dump(self._nb_iters, nb_iters_bytes)

        lr_bytes = io.BytesIO()
        pickle.dump(self._lr, lr_bytes)

        tokenizer_byters = io.BytesIO()
        pickle.dump(self._tokenizer, tokenizer_byters)

        complete_smm = {'model': model_bytes, 'tokenizer': tokenizer_byters,
                        'lr': lr_bytes, 'nb_iters': nb_iters_bytes}
        pickle.dump(complete_smm, f)

    def __eq__(self, other):
        return (torch.equal(self._model.T, other._model.T) and
                self._lr == other._lr and
                self._nb_iters == other._nb_iters and
                self._tokenizer == other._tokenizer)

    def zero_bows(self, nb_bows):
        empty_docs = ["" for _ in range(nb_bows)]
        bows = self._tokenizer.transform(empty_docs)
        bows = torch.from_numpy(bows.A.astype(np.float32))
        if self._model.T.is_cuda:
            bows = bows.cuda()
        return bows

    def build_translator(self, source_vocabulary):
        maxes = []
        argmaxes = []
        for w in source_vocabulary:
            bow = self._tokenizer.transform([w])
            prototype = torch.from_numpy(bow.A.astype(np.float32))
            p_max, p_argmax = prototype.max(dim=1)
            maxes.append(p_max)
            argmaxes.append(p_argmax)

        maxes = torch.cat(maxes, dim=0)
        argmaxes = torch.cat(argmaxes, dim=0)

        if self._model.T.is_cuda:
            maxes = maxes.cuda()
            argmaxes = argmaxes.cuda()

        return lambda W: translate(W, argmaxes, 1-maxes, prototype.size(1))


def translate(W, translation_table, translation_mask, dst_vocab_size):
    W_flat = W.view(-1)  # W was [B, T], W_flat is [BxT]
    translation = translation_table[W_flat]  # [BxT]
    invalid_translations = translation_mask[W_flat].nonzero().view(-1)  # [number of words without translation]

    one_hot = W.new(translation.size() + (dst_vocab_size,)).float()  # [BxT, SMM vocab_size]
    one_hot.zero_()
    one_hot.scatter_(1, translation.view(-1, 1), 1)
    if len(invalid_translations.size()) > 0:
        one_hot[invalid_translations] = 0  # zeroes the entries where no translation should ever happen

    one_hot_reshaped = one_hot.view(W.size() + (dst_vocab_size, ))  # [B, T, SMM vocab_size]
    return one_hot_reshaped.sum(dim=-2)  # [B, SMM vocab_size]


def load(f):
    complete_lm = pickle.load(f)

    model_bytes = complete_lm['model']
    tmp_f = tempfile.TemporaryFile()
    tmp_f.write(model_bytes.getvalue())
    tmp_f.seek(0)
    model = torch.load(tmp_f)

    tokenizer_bytes = complete_lm['tokenizer']
    tokenizer_bytes.seek(0)
    tokenizer = pickle.load(tokenizer_bytes)

    lr_bytes = complete_lm['lr']
    lr_bytes.seek(0)
    lr = pickle.load(lr_bytes)

    nb_iters_bytes = complete_lm['nb_iters']
    nb_iters_bytes.seek(0)
    nb_iters = pickle.load(nb_iters_bytes)

    return IvecExtractor(model, nb_iters, lr, tokenizer)
