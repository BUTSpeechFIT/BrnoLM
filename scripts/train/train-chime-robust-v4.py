#!/usr/bin/env python

import argparse
import logging
import math
import torch

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.aug_paper_pipeline import CleanStreamsProvider, LazyBatcher, TemplSplitterClean
from brnolm.data_pipeline.aug_paper_pipeline import Corruptor, TargetCorruptor

from safe_gpu.safe_gpu import GPUOwner
from brnolm.runtime.runtime_utils import init_seeds, epoch_summary, TransposeWrapper
from brnolm.runtime.runtime_multifile import repackage_hidden
from brnolm.runtime.evaluation import SubstitutionalEnblockEvaluator_v2, EnblockEvaluator

from brnolm.runtime.loggers import ProgressLogger
from brnolm.runtime.reporting import ValidationWatcher


def main(args):
    print(args)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    lm = torch.load(args.load)
    if args.cuda:
        lm.cuda()

    print(lm.model)

    tokenize_regime = 'words'

    print("preparing training data...")
    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=tokenize_regime)
    train_streams = CleanStreamsProvider(train_ids)
    corrupted_provider = TargetCorruptor(train_streams, args.subs_rate, len(lm.vocab), args.del_rate, args.ins_rate, protected=[lm.vocab['</s>']])
    batch_former = LazyBatcher(args.batch_size, corrupted_provider)
    train_data = TemplSplitterClean(args.target_seq_len, batch_former)
    train_data_stream = OndemandDataProvider(TransposeWrapper(train_data), args.cuda)

    print("preparing validation data...")
    if args.augmented_eval:
        # Evaluation (de facto LR scheduling) with input corruption did not
        # help during the CHiMe-6 evaluation
        raise RuntimeError("Not supported")
        evaluator = SubstitutionalEnblockEvaluator_v2(
            lm, args.valid,
            batch_size=10,
            target_seq_len=args.target_seq_len,
            corruptor=lambda data: Corruptor(data, args.ins_rate, protected=['</s>']),
            nb_rounds=args.eval_rounds,
        )
    else:
        evaluator = EnblockEvaluator(lm, args.valid, 10, args.target_seq_len)

    def val_loss_fn():
        return evaluator.evaluate().loss_per_token

    print("computing initial PPL...")
    initial_val_loss = val_loss_fn()
    print('Initial perplexity {:.2f}'.format(math.exp(initial_val_loss)))

    print("training...")
    lr = args.lr
    best_val_loss = None

    val_watcher = ValidationWatcher(val_loss_fn, initial_val_loss, args.val_interval, args.workdir, lm)

    optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)
    patience_ticks = 0
    for epoch in range(1, args.epochs + 1):
        logger = ProgressLogger(epoch, args.log_interval, lr, 600)  # magic number just as a dirty hack, as getting the data is now costly

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

        val_loss = val_loss_fn()
        print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
            patience_ticks = 0
        else:
            patience_ticks += 1
            if patience_ticks > args.patience:
                lr /= 2.0
                patience_ticks = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--valid', type=str, required=True,
                        help='location of the valid corpus')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--subs-rate', type=float, required=True,
                        help='what ratio of tokens should be inserted')
    parser.add_argument('--del-rate', type=float, required=True,
                        help='what ratio of tokens should be removed')
    parser.add_argument('--ins-rate', type=float, required=True,
                        help='what ratio of tokens should be inserted')
    parser.add_argument('--augmented-eval', action='store_true',
                        help='evaluate with augmented data (as opposed to ground truth)')
    parser.add_argument('--eval-rounds', type=int, default=3,
                        help='How many times to go through eval with different augmentations')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')

    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
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

    if args.cuda:
        gpu_owner = GPUOwner()

    main(args)
