import torch

from .runtime_utils import repackage_hidden
from .tensor_reorganization import TensorReorganizer


def prepare_inputs(inputs, do_transpose, use_ivecs, custom_batches):
    X = inputs[0]
    batch_size = X.size(0)
    if do_transpose:
        X = X.t()

    targets = inputs[1]
    if do_transpose:
        targets = targets.t().contiguous()

    if use_ivecs:
        ivecs = inputs[2]
    else:
        ivecs = None

    if custom_batches:
        mask = inputs[-1]  # 3
    else:
        mask = None

    return X, targets, ivecs, mask, batch_size


def evaluate_(lm, data_source, use_ivecs, custom_batches):
    lm.eval()

    total_loss = 0.0
    total_timesteps = 0

    if custom_batches:
        hs_reorganizer = TensorReorganizer(lm.model.init_hidden)

    hidden = None
    do_transpose = not lm.model.batch_first

    for inputs in data_source:
        X, targets, ivecs, mask, batch_size = prepare_inputs(
            inputs,
            do_transpose, use_ivecs, custom_batches
        )

        if hidden is None:
            hidden = lm.model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, mask, batch_size)

        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = lm.model(X, hidden, ivecs)
        else:
            output, hidden = lm.model(X, hidden)

        loss, nb_words = lm.decoder.neg_log_prob(output, targets)
        total_loss += loss.data
        total_timesteps += nb_words

    return total_loss.item() / total_timesteps


def evaluate(model, data_source, use_ivecs):
    return evaluate_(
        model, data_source,
        use_ivecs, custom_batches=True
    )


def evaluate_no_transpose(model, data_source, use_ivecs):
    return evaluate_(
        model, data_source,
        use_ivecs, custom_batches=False
    )


def train_(lm, data, optim, logger, clip, use_ivecs, custom_batches, tb_logger=None):
    lm.train()

    if custom_batches:
        hs_reorganizer = TensorReorganizer(lm.model.init_hidden)

    hidden = None
    do_transpose = not lm.model.batch_first

    for inputs in data:
        X, targets, ivecs, mask, batch_size = prepare_inputs(
            inputs,
            do_transpose, use_ivecs, custom_batches
        )

        if hidden is None:
            hidden = lm.model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, mask, batch_size)
        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = lm.model(X, hidden, ivecs)
        else:
            output, hidden = lm.model(X, hidden)
        loss, nb_words = lm.decoder.neg_log_prob(output, targets)
        loss /= nb_words

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(lm.parameters(), clip)

        optim.step()
        logger.log(loss.data)

        if tb_logger is not None:
            tb_logger.next_step()
            info = {
                'loss/train': loss.item(),
                'ppl/train': loss.exp().item(),
            }
            for tag, value in info.items():
                tb_logger.scalar_summary(tag, value)
                tb_logger.hierarchical_scalar_summary(tag.split('/')[0], tag.split('/')[0], value, enforce=True)

            for tag, value in lm.named_parameters():
                tag = tag.replace('.', '/')
                tb_logger.histo_summary(tag, value.data.cpu().numpy())
                tb_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy())


def train(model, data, optim, logger, clip, use_ivecs):
    train_(
        model, data, optim, logger, clip,
        use_ivecs, custom_batches=True
    )


def train_no_transpose(model, data, optim, logger, clip, use_ivecs):
    train_(
        lm, data, optim, logger, clip,
        use_ivecs, custom_batches=False
    )
