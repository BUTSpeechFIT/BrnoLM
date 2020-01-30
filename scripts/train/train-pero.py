#!/usr/bin/env python

import argparse
import torch

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits

from brnolm.runtime.runtime_utils import TransposeWrapper, init_seeds, epoch_summary
from brnolm.runtime.runtime_multifile import evaluate_, train_

from brnolm.runtime.loggers import ProgressLogger
import math


def lstm_output_from_hidden(hidden):
    return hidden[0][0]


def train(lm, data, clip):
    lm.train()

    do_transpose = not lm.model.batch_first

    for inputs in data:
        X = inputs[0]
        targets = inputs[1]
        targets = torch.cat([X[:, 0].view(-1, 1), targets], dim=1)

        batch_size = X.size(0)
        if do_transpose:
            X = X.t()

        if do_transpose:
            targets = targets.t().contiguous()

        hidden = lm.model.init_hidden(batch_size)

        output, _ = lm.model(X, hidden)
        output = torch.cat(lstm_output_from_hidden(hidden).unsqueeze(0), output)

        loss, nb_words = lm.decoder.neg_log_prob(output, targets)
        loss /= nb_words

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(lm.parameters(), clip)

        optim.step()
        logger.log(loss.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--train', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--valid', type=str, required=True,
                        help='location of the valid corpus')
    parser.add_argument('--characters', action='store_true',
                        help='work on character level, whitespace is significant')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')

    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--beta', type=float, default=0,
                        help='L2 regularization penalty')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    lm = torch.load(args.load)
    if args.cuda:
        lm.cuda()
    print(lm.model)

    print("preparing data...")
    tokenize_regime = 'words'
    if args.characters:
        tokenize_regime = 'chars'

    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=tokenize_regime)
    train_batched = batchify(train_ids, args.batch_size, args.cuda)
    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    train_data = TransposeWrapper(train_data_tb)

    valid_ids = tokens_from_fn(args.valid, lm.vocab, randomize=False, regime=tokenize_regime)
    valid_batched = batchify(valid_ids, 10, args.cuda)
    valid_data_tb = TemporalSplits(
        valid_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    valid_data = TransposeWrapper(valid_data_tb)

    print('Initial perplexity {:.2f}'.format(math.exp(evaluate_(lm, valid_data, use_ivecs=False, custom_batches=False))))

    print("training...")
    lr = args.lr
    best_val_loss = None
    for epoch in range(1, args.epochs+1):
        logger = ProgressLogger(epoch, args.log_interval, lr, len(train_batched)//args.target_seq_len)
        optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)

        train_(
            lm, train_data, optim,
            logger, args.clip,
            use_ivecs=False,
            custom_batches=False,
        )

        val_loss = evaluate_(
            lm, valid_data,
            use_ivecs=False,
            custom_batches=False,
        )
        print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass
