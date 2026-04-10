"""Symbol indexes."""

from __future__ import annotations

import pickle
from typing import Any, DefaultDict, Dict, Iterable, Tuple

from .. import special

from torch import serialization
import yaml
from yoyodyne import util


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
        return iter(self._symbol2index)

    def __len__(self) -> int:
        return len(self._index2symbol)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index."""
        return self._index2symbol[index]


class Index:
    """Maintains the index over the vocabulary."""

    source_vocabulary: Vocabulary
    tag_vocabulary: Vocabulary
    # Keeps track of which source symbols we need the encouder output from.
    source2tags: dict[int, list[int]]

    def __init__(
        self,
        source_vocabulary: Iterable[str],
        source2tags: DefaultDict[str, set[str]],
    ):
        self.source_vocabulary = Vocabulary(source_vocabulary)
        tag_vocabulary = set()
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
        self.source2tags = {
            self.source_vocabulary(source): sorted(
                self.tag_vocabulary(tag) for tag in tags
            )
            for source, tags in source2tags.items()
            if len(tags) > 1
        }

    def __len__(self) -> int:
        return len(self._index2symbol)

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

    @staticmethod
    def _yaml_representer(dumper: yaml.Representer, data: Index):
        return dumper.represent_mapping(
            "!Index",
            {
                "source_vocabulary": data.source_vocabulary,
                "tag_vocabulary": data.tag_vocabulary,
                "source2tags": data.source2tags,
            },
        )

    @staticmethod
    def _yaml_constructor(loader: yaml.Constructor, node: yaml.Node):
        node_value = loader.construct_mapping(node, deep=True)
        obj = object.__new__(Index)
        obj.source_vocabulary = node_value["source_vocabulary"]
        obj.tag_vocabulary = node_value["tag_vocabulary"]
        obj.source2tags = node_value["source2tags"]
        return obj

    def __reduce__(self) -> Tuple[Any, Tuple[Dict[str, Any]]]:
        return (
            _reconstruct_index,
            (
                {
                    "source_vocabulary": self.source_vocabulary,
                    "tag_vocabulary": self.tag_vocabulary,
                    "source2tags": self.source2tags,
                },
            ),
        )


# This whitelists the Index for safe serialization.


def _reconstruct_index(state: Dict[str, Any]) -> Index:
    obj = object.__new__(Index)
    obj.source_vocabulary = state["source_vocabulary"]
    obj.tag_vocabulary = state["tag_vocabulary"]
    obj.source2tags = state["source2tags"]
    return obj


serialization.add_safe_globals([Index, Vocabulary, _reconstruct_index])
yaml.add_representer(Index, Index._yaml_representer, Dumper=yaml.SafeDumper)
yaml.add_constructor("!Index", Index._yaml_constructor, Loader=yaml.SafeLoader)
