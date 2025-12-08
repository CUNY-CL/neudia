#!/usr/bin/env python
"""Creates parallel data by stripping diacritized data.

This can use Unicode decomposition ("unidecode") or the "uninames" method.

The defective text is in the first output column and the original plene text in
the second,
"""

import argparse
import csv
import functools
import logging
import re
import unicodedata


class Error(Exception):

    pass


@functools.lru_cache
def _uniname(char: str) -> str:
    """Single-char implementation of the uninames method."""
    assert len(char) == 1, "multi-character strings not supported"
    charname = unicodedata.name(char)
    if match := re.fullmatch(r"(.+)\s+WITH\s+(.+)", charname):
        stripname = match.group(1)
        try:
            return unicodedata.lookup(stripname)
        except KeyError:
            pass
    return char


def uninames(string: str, compatibility: bool = False) -> str:
    """Strips characters using the uninames method.

    This method is introduced by:

    Náplava, J., Straka, M., Straňák, P., and Hajič, J. 2018. Diacritic
    restoration using neural networks. In Proceedings of the Eleventh
    International Conference on Language Resources and Evaluation, pages
    1566-1573.

    Args:
        string: input string.
        compatibility: use compatibility rather than canonical composition?

    Returns:
        A stripped input string.

    >>> uninames("hold")
    'hold'
    >>> uninames("Straße")
    'Straße'
    >>> uninames("coöperation")
    'cooperation'
    >>> uninames("søster")
    'soster'
    >>> uninames("año")
    'ano'
    >>> uninames("māl")
    'mal'
    >>> uninames("Pająk")
    'Pajak'
    >>> uninames("açai")
    'acai'
    >>> uninames("ealneġ")
    'ealneg'
    """
    string = unicodedata.normalize("NFKC" if compatibility else "NFC", string)
    return "".join(_uniname(char) for char in string)


def unidecode(string: str, compatibility: bool = False) -> str:
    """Strips characters using Unicode decomposition.

    Args:
        string: input string.
        compatibility: use compatibility rather than canonical (de)composition?

    Returns:
        A stripped input string.

    >>> unidecode("hold")
    'hold'
    >>> unidecode("Straße")
    'Straße'
    >>> unidecode("coöperation")
    'cooperation'
    >>> unidecode("søster")  # Identity under this method.
    'søster'
    >>> unidecode("año")
    'ano'
    >>> unidecode("māl")
    'mal'
    >>> unidecode("Pająk")
    'Pajak'
    >>> unidecode("açai")
    'acai'
    >>> unidecode("ealneġ")
    'ealneg'
    """
    if compatibility:
        string = unicodedata.normalize("NFKC", string)
        string = unicodedata.normalize("NFKD", string)
    else:
        string = unicodedata.normalize("NFC", string)
        string = unicodedata.normalize("NFD", string)
    return "".join(char for char in string if not unicodedata.combining(char))


_method_name_to_method = {
    "unidecode": unidecode,
    "uninames": uninames,
}


def main(args: argparse.Namespace) -> None:
    try:
        method = _method_name_to_method[args.method]
    except KeyError:
        raise Error(f"Unknown method: {args.method}")
    with open(args.source, "r") as source, open(args.sink, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        for plene in source:
            plene = plene.rstrip()
            defec = method(plene, args.compatibility)
            tsv_writer.writerow([defec, plene])


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="WARNING")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("sink")
    parser.add_argument(
        "--compatibility",
        default=False,
        action="store_true",
        help="use compatibility rather than canonical (de)composition?",
    )
    parser.add_argument(
        "--method",
        default="unidecode",
        choices=["unidecode", "uninames"],
        help="decomposition method (default: %(default)s)",
    )
    main(parser.parse_args())
