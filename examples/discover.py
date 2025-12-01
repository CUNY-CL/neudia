#!/usr/bin/env python
"""Discovers a tagging spec from diacritized data.

This can use Unicode decomposition ("unidecode") or the "uninames" method.
"""

import argparse
import collections
import logging
import yaml

import methods


class Error(Exception):

    pass


_method_name_to_method = {
    "unidecode": methods.unidecode,
    "uninames": methods.uninames,
}


def main(args: argparse.Namespace) -> None:
    try:
        method = _method_name_to_method[args.method]
    except KeyError:
        raise Error(f"Unknown method: {args.method}")
    table = collections.defaultdict(set)
    with open(args.source, "r") as source:
        for plene in source:
            plene = plene.rstrip()
            defec = method(plene, args.compatibility)
            assert len(plene) == len(defec), "line lengths do not match"
            for pchar, dchar in zip(plene, defec):
                if pchar == dchar:
                    if dchar in table:
                        # Identity is possible for this defective character.
                        table[dchar].add(pchar)
                    else:
                        continue
                if pchar == dchar:
                    continue
                table[dchar].add(pchar)
    data = {dchar: sorted(pchars) for dchar, pchars in table.items()}
    with open(args.sink, "w") as sink:
        yaml.safe_dump(data, sink, allow_unicode=True)


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
