"""Parses the Uses a tagger spec to create all necessary tagger layers.

An empty key in the spec means that the diacritic:

- potentially applies to every segment, and
- is concatenative (i.e., non-spacing, a true diacritic)
"""

from torch import nn
import yaml

INPUT_SIZE = 256

SpecType = dict[str, dict[str, str]]


def _load(path: str) -> SpecType:
    with open(path, "r") as source:
        return yaml.safe_load(source)
