import yaml

from brnolm.data_pipeline.reading import tokens_from_fn
# from brnolm.data_pipeline.multistream import batchify
# from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.data_pipeline.threaded import OndemandDataProvider

# from brnolm.data_pipeline.aug_paper_pipeline import Corruptor, form_input_targets, LazyBatcher, TemplSplitterClean
from brnolm.data_pipeline.aug_paper_pipeline import CleanStreamsProvider, LazyBatcher, TemplSplitterClean

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
    nb_batches = len(train_ids) // batch_size
    train_streams_provider = CleanStreamsProvider(train_ids)
    # corrupted_provider = Corruptor(train_streams, args.subs_rate, len(lm.vocab), args.del_rate, args.ins_rate, protected=[lm.vocab['</s>']])
    batch_former = LazyBatcher(batch_size, train_streams_provider)
    train_data = TemplSplitterClean(target_seq_len, batch_former)
    train_data = TransposeWrapper(train_data)
    return OndemandDataProvider(train_data, place_on_cuda), nb_batches
