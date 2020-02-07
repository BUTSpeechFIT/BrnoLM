#!/usr/bin/env python

import argparse
import math
import random
import torch
import time

from brnolm.data_pipeline.reading import get_independent_lines
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import Batcher

from brnolm.runtime.runtime_utils import init_seeds, epoch_summary
from brnolm.runtime.evaluation import IndependentLinesEvaluator

# from brnolm.runtime.loggers import ProgressLogger
from brnolm.runtime.reporting import ValidationWatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--valid', type=str, required=True,
                        help='location of the valid corpus')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--max-batch-size', type=int, default=20,
                        help='maxiamal batch size')
    parser.add_argument('--max-softmaxes', type=int, default=1000,
                        help='maximal number of softmaxes in a single batch')

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
    parser.add_argument('--val-interval', type=int, default=1000000, metavar='N',
                        help='validation interval in number of tokens')
    parser.add_argument('--workdir',
                        help='where to put models, logs etc.')
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

    print("preparing training data...")
    with open(args.train) as f:
        train_lines = get_independent_lines(f, lm.vocab)

    nb_train_tokens = sum(len(ids) for ids in train_lines)
    nb_oovs = sum(sum(ids == lm.vocab.unk_ind).detach().item() for ids in train_lines)
    print('Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, nb_train_tokens, 100.0 * nb_oovs/nb_train_tokens))

    evaluator = IndependentLinesEvaluator(lm, args.valid, args.max_batch_size, args.max_softmaxes)

    print("computing initial PPL...")
    initial_evaluation = evaluator.evaluate('')
    print('Initial perplexity {:.2f}'.format(math.exp(initial_evaluation.loss_per_token)))

    print("training...")
    lr = args.lr
    best_val_loss = None

    val_watcher = ValidationWatcher(lambda: evaluator.evaluate('').loss_per_token, initial_evaluation.loss_per_token, args.val_interval, args.workdir, lm)

    optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)
    for epoch in range(1, args.epochs + 1):
        # logger = ProgressLogger(epoch, args.log_interval, lr, len(train_batched) // args.target_seq_len)

        nb_batches = 0
        nb_tokens = 0
        running_loss = 0.0
        t0 = time.time()

        random.shuffle(train_lines)
        train_data_stream = OndemandDataProvider(Batcher(train_lines, args.max_batch_size, args.max_softmaxes), cuda=False)
        for batch in train_data_stream:
            nb_batches += 1
            lm.train()
            loss = lm.batch_nll_idxs(batch).sum()
            running_loss += loss.detach().item()
            nb_words = sum(len(s) for s in batch)
            nb_tokens += nb_words
            loss /= nb_words

            val_watcher.log_training_update(loss.data, nb_words)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(lm.parameters(), args.clip)

            optim.step()
            # logger.log(loss.data)

        val_loss = evaluator.evaluate('').loss_per_token
        print(f'epoch {epoch}: {nb_batches} batches, train loss {running_loss:.1f}, running PPL {math.exp(running_loss/nb_tokens):.2f}, val PPL {math.exp(val_loss):.2f}, {time.time() - t0:.1f} sec')
        # print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass


if __name__ == '__main__':
    main()
