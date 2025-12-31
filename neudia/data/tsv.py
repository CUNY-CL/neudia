"""TSV parsing.

The TsvParser yields data from TSV files using 1-based indexing.
"""

import csv
import dataclasses
from typing import Iterator

from .. import defaults


class Error(Exception):
    pass


SampleType = str | tuple[str]


@dataclasses.dataclass
class TsvParser:
    """Streams data from a TSV file.

    Args:
        source_col: 1-indexed column index in TSV containing source
            (defective) strings.
        target_col: 1-indexed column index in TSV containing target
            (plene) strings; 0 indicates no target column is present.
    """

    source_col: int = defaults.SOURCE_COL
    target_col: int = defaults.TARGET_COL

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Out of range source column: {self.source_col}")
        if self.target_col < 0:
            raise Error(f"Out of range target column: {self.target_col}")

    def parse_line(self, line: str) -> SampleType:
        """Parses a single TSV line string."""
        reader = csv.reader([line], delimiter="\t")
        try:
            row = next(reader)
        except StopIteration:
            raise Error("Empty line encountered")
        return self._row_to_sample(row)

    def samples(self, path: str) -> Iterator[SampleType]:
        """Yields source, and target if available."""
        with open(path, "r", encoding=defaults.ENCODING) as source:
            for row in csv.reader(source, delimiter="\t"):
                yield self._row_to_sample(row)

    def _row_to_sample(self, row: list[str]) -> SampleType:
        """Internal helper to convert a row to a SampleType."""
        source = self._get_string(row, self.source_col)
        if self.has_target:
            target = self._get_string(row, self.target_col)
            return source, target
        else:
            return source

    @staticmethod
    def _get_string(row: list[str], col: int) -> str:
        """Returns a string from a row by index.

        Args:
           row: the split row.
           col: the column index.

        Returns:
           The string from that cell.
        """
        return row[col - 1]  # -1 because we're using one-based indexing.

    @property
    def has_target(self) -> bool:
        return self.target_col != 0
