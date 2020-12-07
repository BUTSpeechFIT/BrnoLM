import pickle
import yaml

from brnolm.data_pipeline.reading import tokens_from_fn
# from brnolm.data_pipeline.multistream import batchify
# from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.data_pipeline.threaded import OndemandDataProvider

from brnolm.data_pipeline.aug_paper_pipeline import CleanStreamsProvider, LazyBatcher, TemplSplitterClean
from brnolm.data_pipeline.aug_paper_pipeline import Corruptor
from brnolm.data_pipeline.aug_paper_pipeline import StatisticsCorruptor, Confuser
from brnolm.data_pipeline.aug_paper_pipeline import TargetCorruptor
from brnolm.data_pipeline.aug_paper_pipeline import InputTargetCorruptor

from brnolm.runtime.runtime_utils import TransposeWrapper


def yaml_factory(yaml_fn, lm, device):
    with open(yaml_fn) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    corruptor_config = config.get('corruptor', None)

    return plain_factory(
        data_fn=config['file'],
        lm=lm,
        tokenize_regime=config['tokenize_regime'],
        batch_size=config['batch_size'],
        device=device,
        target_seq_len=config['target_seq_len'],
        corruptor_config=corruptor_config,
    )


def plain_factory(data_fn, lm, tokenize_regime, batch_size, device, target_seq_len, corruptor_config=None):
    train_ids = tokens_from_fn(data_fn, lm.vocab, randomize=False, regime=tokenize_regime)
    nb_batches = len(train_ids) // batch_size
    train_streams_provider = CleanStreamsProvider(train_ids)

    if corruptor_config:
        train_streams_provider = corruptor_factory(corruptor_config, lm, train_streams_provider)

    batch_former = LazyBatcher(batch_size, train_streams_provider)
    if lm.model.in_len == 1:
        train_data = TemplSplitterClean(target_seq_len, batch_former)
    else:
        raise NotImplementedError("Current data pipeline only supports `in_len==1`.")
    train_data = TransposeWrapper(train_data)
    return OndemandDataProvider(train_data, device), nb_batches


def corruptor_factory(config, lm, input_streams_provider):
    if config['type'] == 'input-0gram':
        subs_rate = float(config['substitution-rate'])
        del_rate = float(config['deletion-rate'])
        ins_rate = float(config['insertion-rate'])

        corruptor = Corruptor(
            input_streams_provider,
            subs_rate,
            len(lm.vocab),
            del_rate,
            ins_rate,
            protected=[lm.vocab['</s>']]
        )
        return corruptor

    elif config['type'] == 'input-1gram':
        stats_filename = config['statistics']
        ins_rate = float(config['insertion-rate'])
        mincount = int(config['mincount'])

        with open(stats_filename, 'rb') as f:
            summary = pickle.load(f)
        confuser = Confuser(summary.confusions, lm.vocab, mincount=mincount)

        corrupted_provider = StatisticsCorruptor(
            input_streams_provider,
            confuser,
            ins_rate,
            protected=[lm.vocab['</s>']],
        )

        return corrupted_provider

    if config['type'] == 'target-0gram':
        subs_rate = float(config['substitution-rate'])
        del_rate = float(config['deletion-rate'])
        ins_rate = float(config['insertion-rate'])

        corrupted_provider = TargetCorruptor(
            input_streams_provider,
            subs_rate,
            len(lm.vocab),
            del_rate,
            ins_rate,
            protected=[lm.vocab['</s>']]
        )
        return corrupted_provider

    if config['type'] == 'input_target-0gram':
        in_subs_rate = float(config['input-substitution-rate'])
        target_subs_rate = float(config['target-substitution-rate'])
        del_rate = float(config['deletion-rate'])
        ins_rate = float(config['insertion-rate'])

        corrupted_provider = InputTargetCorruptor(
            input_streams_provider,
            in_subs_rate,
            target_subs_rate,
            len(lm.vocab),
            del_rate,
            ins_rate,
            protected=[lm.vocab['</s>']]
        )
        return corrupted_provider

    else:
        raise ValueError(f"Unsupported type of corruptor: {config['type']}")
