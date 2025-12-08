"""Datasets."""

import dataclasses

import torch
from torch import nn
from torch.utils import data

from . import mappers, tsv


class Item(nn.Module):
    """Tensors representing a single example.

    Args:
        source: source tensor.
        tags: optional tags tensor.
    """

    source: torch.Tensor
    tags: torch.Tensor | None

    def __init__(self, source, tags=None):
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("tags", tags)

    @property
    def has_tags(self) -> bool:
        return self.tags is not None


@dataclasses.dataclass
class Dataset(data.Dataset):
    """Mappable data set.

    This class loads the entire file into memory and is therefore only suitable
    for in-core data sets.
    """

    samples: list[tsv.SampleType]
    mapper: mappers.Mapper
    has_tags: bool

    # Required API.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        sample = self.samples[idx]
        if self.has_tags:
            source, target = sample
            return Item(
                self.mapper.encode_source(source),
                self.mapper.encode_tags(target),
            )
        else:
            source = sample
            return Item(self.mapper.encode_source(source))
