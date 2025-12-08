"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses
from typing import Iterable

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
        # Skips over unkown symbols.
        tags = []
        for symbol in symbols:
            idx = self.index.tag_vocabulary(symbol)
            if idx != special.UNK_IDX:
                tags.append(idx)
        return torch.tensor(tags)

    # Decoding.

    @staticmethod
    def _decode(
        indices: torch.Tensor, vocabulary: indexes.Vocabulary
    ) -> list[str]:
        """Decodes a tensor.

        Padding symbols are omitted.
        """
        symbols = []
        for idx in indices:
            if idx == special.PAD:
                return symbols
            symbols.append(vocabulary.get_symbol(idx))
        return symbols

    def decode_source(self, symbols: Iterable[str]) -> torch.Tensor:
        return self._decode(symbols, self.index.source_vocabulary)

    # FIXME this should be more clever.

    def decode_tags(self, symbols: Iterable[str]) -> torch.Tensor:
        return self._decode(symbols, self.index.tag_vocabulary)
