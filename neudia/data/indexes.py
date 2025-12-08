"""Symbol indexes."""

from __future__ import annotations

import pickle
from typing import DefaultDict, Iterable

from yoyodyne import util

from .. import special


class Error(Exception):
    pass


class Vocabulary:
    """Maintains the index over the vocabulary."""

    _index2symbol: list[str]
    _symbol2index: dict[str, int]

    def __init__(self, vocabulary: Iterable[str]):
        self._index2symbol = special.SPECIAL + sorted(vocabulary)
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __call__(self, lookup: str) -> int:
        """Looks up index by symbol."""
        return self._symbol2index.get(lookup, special.UNK_IDX)

    def __iter__(self) -> Iterable[str]:
        return self._symbol2index.keys()

    def __len__(self) -> int:
        return len(self._index2symbol)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index."""
        return self._index2symbol[index]


class Index:
    """Maintains the index over the vocabulary."""

    source_vocabulary: Vocabulary
    tag_vocabulary: Vocabulary
    _source2tags: dict[str, list[int]]

    def __init__(
        self,
        source_vocabulary: Iterable[str],
        source2tags: DefaultDict[str, set[str]],
    ):
        self.source_vocabulary = Vocabulary(source_vocabulary)
        tag_vocabulary = set()
        self._source2tags = {}
        for source, tags in source2tags.items():
            if len(tags) > 1:
                tag_vocabulary.update(tags)
            elif len(tags) == 1:
                (tag,) = tags
                if source != tag:
                    raise Error(
                        "Cowardly refusing to create deterministic "
                        f"non-identity mapping: {source} -> {tag}"
                    )
        self.tag_vocabulary = Vocabulary(tag_vocabulary)
        for source, tags in source2tags.items():
            self._source2tags[self.source_vocabulary(source)] = sorted(
                self.tag_vocabulary(tag) for tag in tags
            )

    def __len__(self) -> int:
        return len(self._index2symbol)

    def get_tags(self, index: int) -> list[int] | None:
        return self._source2tags.get(index)

    # Serialization.

    @classmethod
    def read(cls, model_dir: str) -> Index:
        """Loads index.

        Args:
            model_dir (str).

        Returns:
            Index.
        """
        with open(cls.path(model_dir), "rb") as source:
            return pickle.load(source)

    def write(self, model_dir: str) -> None:
        """Writes index.

        Args:
            model_dir (str).
        """
        path = self.path(model_dir)
        util.mkpath(path)
        with open(path, "wb") as sink:
            pickle.dump(self, sink)

    @staticmethod
    def path(model_dir: str) -> str:
        return f"{model_dir}/index.pkl"
