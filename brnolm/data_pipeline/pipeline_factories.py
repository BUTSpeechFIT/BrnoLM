import yaml

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.data_pipeline.threaded import OndemandDataProvider

from brnolm.runtime.runtime_utils import TransposeWrapper


def args_factory(args, lm):
    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=args.tokenize_regime)

    train_batched = batchify(train_ids, args.batch_size, cuda=False)
    single_stream_len = len(train_batched)

    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )

    train_data = TransposeWrapper(train_data_tb)
    return OndemandDataProvider(train_data, args.cuda), single_stream_len


def yaml_factory(yaml_fn, lm, place_on_cuda):
    with open(yaml_fn) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    train_ids = tokens_from_fn(config['file'], lm.vocab, randomize=False, regime=config['tokenize_regime'])

    train_batched = batchify(train_ids, config['batch_size'], cuda=False)
    single_stream_len = len(train_batched)

    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=config['target_seq_len']
    )

    train_data = TransposeWrapper(train_data_tb)
    return OndemandDataProvider(train_data, place_on_cuda), single_stream_len
