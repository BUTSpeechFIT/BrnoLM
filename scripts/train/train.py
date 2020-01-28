#!/usr/bin/env python

import argparse
import math
import torch
from logger import Logger

from brnolm.data_pipeline.data import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits

from brnolm.runtime.runtime_utils import TransposeWrapper, init_seeds, epoch_summary
from brnolm.runtime.runtime_multifile import evaluate_, train_

from brnolm.runtime.loggers import ProgressLogger


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
    parser.add_argument('--tb-logdir', type=str,
                        help='folder for logs for TensorBoard')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    tb_logger = None
    if args.tb_logdir:
        tb_logger = Logger(args.tb_logdir, update_freq=100)

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
            tb_logger=tb_logger,
        )

        val_loss = evaluate_(
            lm, valid_data,
            use_ivecs=False,
            custom_batches=False,
        )
        print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))
        if tb_logger is not None:
            info = {
                'loss/val': val_loss,
                'ppl/val': math.exp(val_loss),
            }
            for tag, value in info.items():
                tb_logger.scalar_summary(tag, value, enforce=True)
                tb_logger.hierarchical_scalar_summary(tag.split('/')[0], tag.split('/')[1], value, enforce=True)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass
