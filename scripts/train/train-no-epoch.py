#!/usr/bin/env python

import argparse
import logging
import math
import sys
import torch

from safe_gpu.safe_gpu import GPUOwner

from brnolm.data_pipeline.reading import tokenizer_factory
from brnolm.data_pipeline.pipeline_factories import plain_factory_noepoch, yaml_factory_noepoch

from brnolm.runtime.runtime_utils import init_seeds
from brnolm.runtime.runtime_multifile import repackage_hidden
from brnolm.runtime.evaluation import EnblockEvaluator

from brnolm.runtime.loggers import InfinityLogger
from brnolm.runtime.reporting import ValidationWatcher


def main(args):
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    lm = torch.load(args.load).to(device)
    print(lm.model)

    print("preparing training data...")
    tokenizer = tokenizer_factory.construct_tokenizer(args.tokenize_regime, lm.vocab)
    if args.train_yaml:
        train_data_stream, single_stream_len = yaml_factory_noepoch(args.train_yaml, lm, device)
    else:
        train_data_stream, single_stream_len = plain_factory_noepoch(
            data_fn=args.train,
            lm=lm,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
            target_seq_len=args.target_seq_len,
        )

    print("preparing validation data...")
    evaluator = EnblockEvaluator(lm, args.valid, 10, args.target_seq_len, tokenizer)

    def val_loss_fn():
        return evaluator.evaluate().loss_per_token

    print("computing initial PPL...")
    initial_val_loss = val_loss_fn()
    print('Initial perplexity {:.2f}'.format(math.exp(initial_val_loss)))

    print("training...")
    lr = args.lr
    best_val_loss = None

    val_watcher = ValidationWatcher(val_loss_fn, initial_val_loss, args.val_interval, args.workdir, lm)
    best_val_loss = initial_val_loss

    optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)
    patience_ticks = 0

    logger = InfinityLogger(0, args.log_interval, lr)

    hidden = None
    for X, targets in train_data_stream:
        if hidden is None:
            hidden = lm.model.init_hidden(X.shape[0])

        hidden = repackage_hidden(hidden)

        lm.train()
        output, hidden = lm.model(X, hidden)
        loss, nb_words = lm.decoder.neg_log_prob(output, targets)
        loss /= nb_words

        val_watcher.log_training_update(loss.data, nb_words)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), args.clip)

        optim.step()
        logger.log(loss.data)

    val_loss = val_loss_fn()

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save(lm, args.save)
        best_val_loss = val_loss
        patience_ticks = 0
    else:
        patience_ticks += 1
        if patience_ticks > args.patience:
            lr /= 2.0
            if lr < args.min_lr:
                print(f"Learning has reached {lr}, training was supposed to stop at {args.min_lr}, stopping.")
            for p in optim.param_groups:
                p['lr'] = lr
            patience_ticks = 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')
    parser = argparse.ArgumentParser()
    train_definition = parser.add_mutually_exclusive_group()
    train_definition.add_argument('--train-yaml', type=str,
                                  help='location of a yaml config describing the training dataset')
    train_definition.add_argument('--train', type=str,
                                  help='location of the train corpus')
    parser.add_argument('--valid', type=str, required=True,
                        help='location of the valid corpus')
    tokenizer_factory.register_parameter(parser, '--tokenize-regime')

    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')

    parser.add_argument('--lr', type=float, default=2.0,
                        help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-3,
                        help='minimal learning rate, once reached, training stops')
    parser.add_argument('--patience', type=int, default=0,
                        help='how many epochs since last improvement to wait till reducing LR')
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
    if not args.train and not args.train_yaml:
        sys.stderr.write('Either --train of --train-yaml have to be provided\n')
        sys.exit(2)

    if args.cuda:
        gpu_owner = GPUOwner()

    main(args)
