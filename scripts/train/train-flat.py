#!/usr/bin/env python

import argparse
import math
import torch
import pathlib

from brnolm.data_pipeline.data import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.data_pipeline.threaded import OndemandDataProvider

from brnolm.runtime.runtime_utils import TransposeWrapper, init_seeds, epoch_summary
from brnolm.runtime.runtime_multifile import evaluate_, repackage_hidden

from brnolm.runtime.loggers import ProgressLogger


class ValidationWatcher:
    def __init__(self, val_fn, initial_val_loss, freq_in_tokens, workdir, lm):
        self.val_losses = [initial_val_loss]
        self.validation_fn = val_fn
        self.lm = lm

        pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

        assert(freq_in_tokens > 0)
        self.freq = freq_in_tokens

        self.report_fn = '{}/validation-report.txt'.format(workdir)
        self.best_model_fn = '{}/best.lm'.format(workdir)
        self.latest_model_fn = '{}/latest.lm'.format(workdir)

        self.running_loss = 0.0
        self.running_targets = 0
        self.running_updates = 0

        self.nb_total_updates = 0

    def log_training_update(self, loss, nb_targets):
        self.running_loss += loss
        self.running_targets += nb_targets
        self.running_updates += 1
        self.nb_total_updates += 1

        if self.running_targets > self.freq:
            val_loss = self.run_validation()

            running_ppl = math.exp(self.running_loss / self.running_updates)
            val_ppl = math.exp(val_loss)

            desc = '{} updates: {:.1f} {:.1f} {:.2f}\n'.format(
                self.nb_total_updates, running_ppl, val_ppl, val_ppl - running_ppl
            )
            with open(self.report_fn, 'a') as f:
                f.write(desc)

            torch.save(self.lm, self.latest_model_fn)
            if min(self.val_losses) == self.val_losses[-1]:
                torch.save(self.lm, self.best_model_fn)

            self.running_loss = 0.0
            self.running_targets = 0
            self.running_updates = 0

    def run_validation(self):
        val_loss = self.validation_fn()
        self.val_losses.append(val_loss)
        return val_loss


def main():
    parser = argparse.ArgumentParser()
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

    tokenize_regime = 'words'
    if args.characters:
        tokenize_regime = 'chars'

    print("preparing training data...")
    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=tokenize_regime)
    train_batched = batchify(train_ids, args.batch_size, cuda=False)
    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    train_data = TransposeWrapper(train_data_tb)
    train_data_stream = OndemandDataProvider(train_data, args.cuda)

    print("preparing validation data...")
    valid_ids = tokens_from_fn(args.valid, lm.vocab, randomize=False, regime=tokenize_regime)
    valid_batched = batchify(valid_ids, 10, args.cuda)
    valid_data_tb = TemporalSplits(
        valid_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    valid_data = TransposeWrapper(valid_data_tb)

    def val_loss_fn(lm):
        return evaluate_(lm, valid_data, use_ivecs=False, custom_batches=False)

    print("computing initial PPL...")
    initial_val_loss = val_loss_fn(lm)
    print('Initial perplexity {:.2f}'.format(math.exp(initial_val_loss)))

    print("training...")
    lr = args.lr
    best_val_loss = None

    val_watcher = ValidationWatcher(lambda: val_loss_fn(lm), initial_val_loss, args.val_interval, args.workdir, lm)

    optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)
    for epoch in range(1, args.epochs + 1):
        logger = ProgressLogger(epoch, args.log_interval, lr, len(train_batched) // args.target_seq_len)

        hidden = None
        for X, targets in train_data_stream:
            if hidden is None:
                hidden = lm.model.init_hidden(args.batch_size)

            hidden = repackage_hidden(hidden)

            lm.train()
            output, hidden = lm.model(X, hidden)
            loss, nb_words = lm.decoder.neg_log_prob(output, targets)
            loss /= nb_words

            val_watcher.log_training_update(loss.data, nb_words)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(lm.parameters(), args.clip)

            optim.step()
            logger.log(loss.data)

        val_loss = val_loss_fn(lm)
        print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass


if __name__ == '__main__':
    main()
