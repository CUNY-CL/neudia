#!/usr/bin/env python
"""Creates parallel data by stripping diacritized data.

This can use Unicode decomposition ("unidecode") or the "uninames" method.

The defective text is in the first output column and the original plene text in
the second,
"""

import argparse
import csv
import collections
import logging

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
