import yaml

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.data_pipeline.threaded import OndemandDataProvider

from brnolm.runtime.runtime_utils import TransposeWrapper


def yaml_factory(yaml_fn, lm, place_on_cuda):
    with open(yaml_fn) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    return plain_factory(
        data_fn=config['file'],
        lm=lm,
        tokenize_regime=config['tokenize_regime'],
        batch_size=config['batch_size'],
        place_on_cuda=place_on_cuda,
        target_seq_len=config['target_seq_len'],
    )


def plain_factory(data_fn, lm, tokenize_regime, batch_size, place_on_cuda, target_seq_len):
    train_ids = tokens_from_fn(data_fn, lm.vocab, randomize=False, regime=tokenize_regime)

    train_batched = batchify(train_ids, batch_size, cuda=False)
    single_stream_len = len(train_batched)

    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=target_seq_len
    )

    train_data = TransposeWrapper(train_data_tb)
    return OndemandDataProvider(train_data, place_on_cuda), single_stream_len
