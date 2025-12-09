"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Iterator

import torch

from . import indexes
from .. import special


@dataclasses.dataclass
class Mapper:
    """Handles mapping between strings and tensors."""

    index: indexes.Index  # Usually copied from the DataModule.

    @classmethod
    def read(cls, model_dir: str) -> Mapper:
        """Loads mapper from an index.

        Args:
            model_dir (str).

        Returns:
            Mapper.
        """
        return cls(indexes.Index.read(model_dir))

    # Encoding.

    def encode_source(self, symbols: Iterable[str]) -> torch.Tensor:
        return torch.tensor(
            [self.index.source_vocabulary(symbol) for symbol in symbols]
        )

    def encode_tags(self, symbols: Iterable[str]) -> torch.Tensor:
        tags = []
        for symbol in symbols:
            idx = self.index.tag_vocabulary(symbol)
            # Skips over identity tags.
            if idx != special.UNK_IDX:
                tags.append(idx)
        return torch.tensor(tags)

    # Decoding.

    def decode_source(self, indices: torch.Tensor) -> Iterator[str]:
        for idx in indices:
            if idx == special.PAD_IDX:
                return
            yield self.index.vocabulary.get_symbol(idx)

    def decode_tagged(
        self, source_indices: torch.Tensor, tag_indices: torch.Tensor
    ) -> Iterator[str]:
        tag_it = iter(tag_indices)
        for source in source_indices:
            source_idx = source.item()
            if source_idx == special.PAD_IDX:
                return
            elif source_idx in self.index.encoder_keep:
                tag_idx = next(tag_it).item()
                yield self.index.tag_vocabulary.get_symbol(tag_idx)
            else:
                yield self.index.source_vocabulary.get_symbol(source_idx)
