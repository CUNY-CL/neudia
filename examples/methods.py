"""Decomposition methods."""

import functools
import re
import unicodedata


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
