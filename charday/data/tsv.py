"""TSV parsing.

The TsvParser yields data from TSV files using 1-based indexing.
"""

import csv
from typing import Iterator

from . import defaults


class Error(Exception):
    pass


SampleType = tuple[str] | str


@dataclasses.dataclass
class TsvParser:
    """Streams data from a TSV file.

    Args:
        source_col (int, optional): 1-indexed column in TSV containing
            source (defective) strings.
        target_col (int, optional): 1-indexed column in TSV containing
            target (plene) strings.
    """

    source_col: int = defaults.SOURCE_COL
    target_col: int = defaults.TARGET_COL

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Out of range source column: {self.source_col}")
        if self.target_col < 0:
            raise Error(f"Out of range target column: {self.target_col}")

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    def samples(self, path: str) -> Iterator[SampleType]:
        """Yields source, and target if available."""
        for row in self._tsv_reader(path):
            source = self._get_string(row, self.source_col)
            if self.has_target:
                target = self._get_string(row, self.target_col)
                yield source, target
            else:
                yield source

    @staticmethod
    def _get_string(row: list[str], col: int) -> str:
        """Returns a string from a row by index.

        Args:
           row (list[str]): the split row.
           col (int): the column index.

        Returns:
           str: symbol from that string.
        """
        return row[col - 1]  # -1 because we're using one-based indexing.

    @staticmethod
    def _tsv_reader(path: str) -> Iterator[list[str]]:
        with open(path, "r", encoding=defaults.ENCODING) as tsv:
            yield from csv.reader(tsv, delimiter="\t")
