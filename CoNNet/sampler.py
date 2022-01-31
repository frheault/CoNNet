#!/usr/bin/env python


import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        indices (sequence)   : a sequence of indices to the original dataset 
            slicing organisation
        weights (sequence)   : a sequence of weights, not necessary summing
        up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for
            that row.
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self,  indices: Sequence[int], weights: Sequence[float],
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = len(weights)
        self.indices = indices
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples,
                                        self.replacement,
                                        generator=self.generator)
        for i in rand_tensor.tolist():
            yield self.indices[i]

    def __len__(self) -> int:
        return self.num_samples
