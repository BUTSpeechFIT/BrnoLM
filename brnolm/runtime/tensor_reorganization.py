import torch

from typing import Dict, Any


class Singleton(type):
    _instances: Dict[Any, Any] = {}  # TODO what is the actual type?

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class InfiniNoneType(metaclass=Singleton):
    def __eq__(self, other):
        return other is None or isinstance(other, InfiniNone)

    def __iter__(self):
        while True:
            yield InfiniNoneType()


InfiniNone = InfiniNoneType()


def reorg_single(orig, mask, new=None):
    reorg = torch.index_select(orig, dim=-2, index=mask)
    if new is not InfiniNone:
        reorg = torch.cat([reorg, new], dim=-2)

    return reorg


class TensorReorganizer():
    def __init__(self, zeros_provider):
        self._zeros_provider = zeros_provider

    def __call__(self, orig, mask, batch_size):
        if len(mask) == 0:
            return self._zeros_provider(batch_size)

        if mask.size(0) > batch_size:
            raise ValueError("Cannot reorganize mask {} to batch size {}".format(mask, batch_size))

        if isinstance(orig, tuple):
            single_var = False
        elif isinstance(orig, torch.Tensor):
            single_var = True
        else:
            raise TypeError(
                "orig has unsupported type {}, "
                "only tuples and Tensors are accepted".format(
                    orig.__class__
                )
            )

        adding = mask.size(0) < batch_size
        if adding:
            nb_needed_new = batch_size - mask.size(0)
            new = self._zeros_provider(nb_needed_new)
        else:
            new = InfiniNone

        if single_var:
            reorg = reorg_single(orig, mask, new)
        else:
            reorg = tuple(reorg_single(o, mask, n) for o, n in zip(orig, new))

        return reorg
