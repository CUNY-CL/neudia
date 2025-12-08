"""Collators."""

import dataclasses

from . import batches, datasets


@dataclasses.dataclass
class Collator:
    """Pads data."""

    has_tags: bool

    def pad_source(
        self, itemlist: list[datasets.Item]
    ) -> batches.PaddedTensor:
        """Pads source."""
        return batches.PaddedTensor([item.source for item in itemlist])

    def pad_tags(
        self,
        itemlist: list[datasets.Item],
    ) -> batches.PaddedTensor:
        """Pads tags."""
        return batches.PaddedTensor([item.tags for item in itemlist])

    def __call__(self, itemlist: list[datasets.Item]) -> batches.Batch:
        """Pads all elements of an itemlist."""
        return batches.Batch(
            self.pad_source(itemlist),
            self.pad_tags(itemlist) if self.has_tags else None,
        )
