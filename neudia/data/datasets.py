"""Datasets."""

import abc
import dataclasses
import mmap
from typing import BinaryIO, Iterator

import torch
from torch import nn
from torch.utils import data

from .. import defaults
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
class AbstractDataset(abc.ABC):
    """Base class for datasets.

    Args:
        path: path to input TSV file.
        mapper: mapper for encoding.
        parser: TSV parser object.
    """

    path: str
    mapper: mappers.Mapper
    parser: tsv.TsvParser

    @property
    def has_tags(self) -> bool:
        return self.parser.has_target

    def sample_to_item(self, sample: tsv.SampleType) -> Item:
        if self.has_tags:
            source, target = sample
            return Item(
                self.mapper.encode_source(source),
                self.mapper.encode_tags(source, target),
            )
        else:
            source = sample
            return Item(self.mapper.encode_source(source))


@dataclasses.dataclass
class IterableDataset(AbstractDataset, data.IterableDataset):
    """Iterable (non-random access) data set."""

    def __iter__(self) -> Iterator[Item]:
        for sample in self.parser.samples(self.path):
            yield self.sample_to_item(sample)


@dataclasses.dataclass
class MappableDataset(AbstractDataset, data.Dataset):
    """Mappable (random access) data set.

    This is implemented with a memory map after making a single pass through
    the file to compute offsets.

    Args:
        sequential (bool, optional): will this data set by used for repeated
            linear access, as is the case for validation data?
    """

    sequential: bool = False

    _offsets: list[int] = dataclasses.field(default_factory=list, init=False)
    _mmap: mmap.mmap | None = dataclasses.field(default=None, init=False)
    _fobj: BinaryIO | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        # Computes offsets.
        offset = 0
        with open(self.path, "rb") as source:
            for line in source:
                self._offsets.append(offset)
                offset += len(line)

    def _get_mmap(self) -> mmap.mmap:
        # Makes this safe for use with multiple workers.
        if self._mmap is None:
            self._fobj = open(self.path, "rb")
            if hasattr(mmap, "MAP_POPULATE"):  # Linux-specific.
                flags = mmap.MAP_SHARED
                if not self.sequential:
                    flags |= mmap.MAP_POPULATE
                self._mmap = mmap.mmap(
                    self._fobj.fileno(),
                    0,
                    flags=flags,
                    prot=mmap.PROT_READ,
                )
                if self.sequential:
                    self._mmap.madvise(mmap.MADV_WILLNEED)
                    self._mmap.madvise(mmap.MADV_SEQUENTIAL)
                else:
                    self._mmap.madvise(mmap.MADV_RANDOM)
            else:
                self._mmap = mmap.mmap(
                    self._fobj.fileno(), 0, access=mmap.ACCESS_READ
                )
        return self._mmap

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Item:
        mm = self._get_mmap()
        start = self._offsets[idx]
        if idx + 1 < len(self._offsets):
            end = self._offsets[idx + 1]
        else:
            end = mm.size()
        line = mm[start:end].decode(defaults.ENCODING).rstrip()
        sample = self.parser.parse_line(line)
        return self.sample_to_item(sample)

    def __del__(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fobj is not None:
            self._fobj.close()
