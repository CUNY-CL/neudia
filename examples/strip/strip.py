#!/usr/bin/env python
"""Creates parallel data by stripping diacritized data.

This can use Unicode decomposition ("unidecode") or the "uninames" method.

The defective text is in the first output column and the original plene text in
the second.

By default, null characters are inserted to preserve alignment information.
"""

import argparse
import csv
import functools
import logging
import re
import unicodedata


class Error(Exception):

    pass


# Constants
ENCODING = "utf-8"
NUL = "\0"  # Null character used as separator for chunk alignment.


def _flatten(chunk: list[str], compatibility: bool = False) -> str:
    string = "".join(chunk)
    return unicodedata.normalize("NFKC" if compatibility else "NFC", string)


@functools.lru_cache
def _uniname(char: str) -> str:
    """Single-char implementation of the uninames method."""
    assert len(char) == 1, "multi-character strings not supported"
    charname = unicodedata.name(char, "")
    if not charname:
        return char
    if mtch := re.fullmatch(r"(.+)\s+WITH\s+(.+)", charname):
        stripname = mtch.group(1)
        try:
            return unicodedata.lookup(stripname)
        except KeyError:
            pass
    return char


def uninames(string: str, compatibility: bool = False) -> list[str]:
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
        A tuple of (defective chunks, plene chunks).

    >>> uninames("hold")
    (['h', 'o', 'l', 'd'], ['h', 'o', 'l', 'd'])

    >>> uninames("Straße")
    (['S', 't', 'r', 'a', 'ß', 'e'], ['S', 't', 'r', 'a', 'ß', 'e'])

    >>> uninames("coöp")
    (['c', 'o', 'o', 'p'], ['c', 'o', 'ö', 'p'])

    >>> uninames("søster")
    (['s', 'o', 's', 't', 'e', 'r'], ['s', 'ø', 's', 't', 'e', 'r'])

    >>> uninames("año")
    (['a', 'n', 'o'], ['a', 'ñ', 'o'])

    >>> uninames("māl")
    (['m', 'a', 'l'], ['m', 'ā', 'l'])

    >>> uninames("Pająk")
    (['P', 'a', 'j', 'a', 'k'], ['P', 'a', 'j', 'ą', 'k'])

    >>> uninames("açai")
    (['a', 'c', 'a', 'i'], ['a', 'ç', 'a', 'i'])

    >>> uninames("ealneǧ")
    (['e', 'a', 'l', 'n', 'e', 'g'], ['e', 'a', 'l', 'n', 'e', 'ǧ'])

    # Non-composable harakat.
    >>> uninames("مَرْحَبًا")
    (['م', 'ر', 'ح', 'ب', 'ا'], ['مَ', 'رْ', 'حَ', 'بً', 'ا'])

    # Non-composable nikkud.
    >>> uninames("שָׁלוֹם")
    (['ש', 'ל', 'ו', 'ם'], ['שָׁ', 'ל', 'וֹ', 'ם'])

    # Non-composable virama and matra.
    >>> uninames("नमस्ते")
    (['न', 'म', 'स', 'त'], ['न', 'म', 'स्', 'ते'])
    """
    string = unicodedata.normalize("NFKC" if compatibility else "NFC", string)
    plene = []
    defec = []
    chunk = []
    for plene_char in string:
        category = unicodedata.category(plene_char)
        if category.startswith("M") or category == "Cf":
            # Adds combining or zero-width characters to current chunk.
            chunk.append(plene_char)
        else:
            # Finalizes previous chunk if any.
            if chunk:
                plene.append(_flatten(chunk, compatibility))
                chunk.clear()
            # Starts new chunk.
            defec.append(_uniname(plene_char))
            chunk.append(plene_char)
    if chunk:
        # Finalizes last plene chunk if any.
        plene.append(_flatten(chunk, compatibility))
    assert len(defec) == len(plene), "length mismatch"
    return defec, plene


def unidecode(
    string: str, compatibility: bool = False
) -> tuple[list[str], list[str]]:
    """Strips characters using Unicode decomposition.

    Args:
        string: input string.
        compatibility: use compatibility rather than canonical (de)composition?

    Returns:
        A tuple of (defective chunks, plene chunks).

    >>> unidecode("hold")
    (['h', 'o', 'l', 'd'], ['h', 'o', 'l', 'd'])

    >>> unidecode("Straße")
    (['S', 't', 'r', 'a', 'ß', 'e'], ['S', 't', 'r', 'a', 'ß', 'e'])

    >>> unidecode("coöp")
    (['c', 'o', 'o', 'p'], ['c', 'o', 'ö', 'p'])

    # Identity under this method.
    >>> unidecode("søster")
    (['s', 'ø', 's', 't', 'e', 'r'], ['s', 'ø', 's', 't', 'e', 'r'])

    >>> unidecode("año")
    (['a', 'n', 'o'], ['a', 'ñ', 'o'])

    >>> unidecode("māl")
    (['m', 'a', 'l'], ['m', 'ā', 'l'])

    >>> unidecode("Pająk")
    (['P', 'a', 'j', 'a', 'k'], ['P', 'a', 'j', 'ą', 'k'])

    >>> unidecode("açai")
    (['a', 'c', 'a', 'i'], ['a', 'ç', 'a', 'i'])

    >>> unidecode("ealneǧ")
    (['e', 'a', 'l', 'n', 'e', 'g'], ['e', 'a', 'l', 'n', 'e', 'ǧ'])

    # Non-composable harakat.
    >>> unidecode("مَرْحَبًا")
    (['م', 'ر', 'ح', 'ب', 'ا'], ['مَ', 'رْ', 'حَ', 'بً', 'ا'])

    # Non-composable nikkud.
    >>> unidecode("שָׁלוֹם")
    (['ש', 'ל', 'ו', 'ם'], ['שָׁ', 'ל', 'וֹ', 'ם'])

    # Non-composable virama and matra.
    >>> unidecode("नमस्ते")
    (['न', 'म', 'स', 'त'], ['न', 'म', 'स्', 'ते'])
    """
    if compatibility:
        string = unicodedata.normalize("NFKC", string)
        string = unicodedata.normalize("NFKD", string)
    else:
        string = unicodedata.normalize("NFC", string)
        string = unicodedata.normalize("NFD", string)
    defec = []
    plene = []
    chunk = []
    for char in string:
        category = unicodedata.category(char)
        if category.startswith("M") or category == "Cf":
            # Adds combining or zero-width characters to current chunk.
            chunk.append(char)
        else:
            # Finalizes previous chunk if any.
            if chunk:
                plene.append(_flatten(chunk, compatibility))
                chunk.clear()
            # Starts new chunk.
            defec.append(char)
            chunk.append(char)
    if chunk:
        # Finalizes last plene chunk if any.
        plene.append(_flatten(chunk, compatibility))
    assert len(defec) == len(plene), "length mismatch"
    return defec, plene


_method_name_to_method = {
    "unidecode": unidecode,
    "uninames": uninames,
}


def main(args: argparse.Namespace) -> None:
    try:
        method = _method_name_to_method[args.method]
    except KeyError:
        raise Error(f"Unknown method: {args.method}")
    with (
        open(args.source, "r", encoding=ENCODING) as source,
        open(args.sink, "w", encoding=ENCODING, newline="") as sink,
    ):
        tsv_writer = csv.writer(sink, delimiter="\t")
        for plene_line in source:
            defec_chunks, plene_chunks = method(
                plene_line.rstrip(), args.compatibility
            )
            defec = (
                NUL.join(defec_chunks)
                if args.insert_nul
                else "".join(defec_chunks)
            )
            plene = (
                NUL.join(plene_chunks)
                if args.insert_nul
                else "".join(plene_chunks)
            )
            tsv_writer.writerow([defec, plene])


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="WARNING")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="input file path")
    parser.add_argument("sink", help="output file path")
    parser.add_argument(
        "--compatibility",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use compatibility (de)composition (default: %(default)s)",
    )
    parser.add_argument(
        "--method",
        default="unidecode",
        choices=["unidecode", "uninames"],
        help="decomposition method (default: %(default)s)",
    )
    parser.add_argument(
        "--insert-nul",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="inserts NUL bytes between chunks (default: %(default)s)",
    )
    main(parser.parse_args())
