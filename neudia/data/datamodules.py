"""Data modules."""

import collections
import logging
from typing import Iterable

import lightning
from torch.utils import data

from .. import defaults
from . import collators, datasets, indexes, mappers, tsv


class DataModule(lightning.LightningDataModule):
    """Data module.

    This is responsible for indexing the data, collating/padding, and
    generating datasets.

    Args:
        model_dir: Path for checkpoints, indexes, and logs.
        train: Path for training data TSV.
        val: Path for validation data TSV.
        predict: Path for prediction data TSV.
        test: Path for test data TSV.
        source_col: 1-indexed column in TSV containing source strings.
        target_col: 1-indexed column in TSV containing target strings.
        batch_size: desired batch size.
    """

    train: str | None
    val: str | None
    predict: str | None
    test: str | None
    parser: tsv.TsvParser
    batch_size: int
    index: indexes.Index
    collator: collators.Collator

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        train=None,
        val=None,
        predict=None,
        test=None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        target_col: int = defaults.TARGET_COL,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.parser = tsv.TsvParser(
            source_col=source_col, target_col=target_col
        )
        self.batch_size = batch_size
        self.index = (
            self._make_index()
            if self.train
            else indexes.Index.read(self.model_dir)
        )
        self.log_vocabularies()
        self.collator = collators.Collator(self.has_target)

    def _make_index(self) -> indexes.Index:
        source_vocabulary = set()
        source2tags = collections.defaultdict(set)
        if self.has_target:
            for source_cell, target_cell in self.parser.samples(self.train):
                for source_char, target_char in zip(source_cell, target_cell):
                    source_vocabulary.add(source_char)
                    source2tags[source_char].add(target_char)
        index = indexes.Index(source_vocabulary, source2tags)
        # Writes it to the model directory.
        index.write(self.model_dir)
        return index

    # Logging.

    def log_vocabularies(self) -> None:
        """Logs this module's vocabularies."""
        logging.info(
            "Source vocabulary (%d): %s",
            len(self.index.source_vocabulary),
            self.pprint(self.index.source_vocabulary),
        )
        if self.has_target:
            logging.info(
                "Tag vocabulary (%d): %s",
                len(self.index.tag_vocabulary),
                self.pprint(self.index.tag_vocabulary),
            )

    @staticmethod
    def pprint(vocabulary: Iterable) -> str:
        """Prints the vocabulary for debugging dnd logging purposes."""
        return ", ".join(f"{symbol!r}" for symbol in vocabulary)

    # Properties.

    # has_source is always true.

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

    @property
    def source_vocab_size(self) -> int:
        return len(self.index.source_vocabulary)

    @property
    def tags_vocab_size(self) -> int:
        return len(self.index.tag_vocabulary)

    @property
    def source2tags(self) -> dict[int, list[int]]:
        return self.index.source2tags

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._dataset(self.train),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no val path"
        return data.DataLoader(
            self._dataset(self.val),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            self._dataset(self.predict),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._dataset(self.test),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def _dataset(self, path: str) -> datasets.Dataset:
        return datasets.Dataset(
            list(self.parser.samples(path)),
            mappers.Mapper(self.index),
            self.has_target,
        )
