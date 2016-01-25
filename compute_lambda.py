#!/usr/bin/env python3


import os
import sys
import logging
import argparse

import pandas as pd

try:
    import scipy.stats
    expected_median = scipy.stats.chi2.ppf(0.5, 1)
except ImportError:
    expected_median = 0.4549364231195725


__copyright__ = "Copyright 2014, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "MIT"
__version__ = "0.1"


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lambda")


def main():
    """The main function."""
    args = parse_args()
    check_args(args)

    for fn in args.i_filenames:
        logger.info("Reading " + fn)
        data = pd.read_csv(fn, sep=args.delim, usecols=[args.field])
        data = data.dropna()

        stats = data[args.field]
        if not args.chi2:
            stats = stats ** 2

        inflation_factor = max(stats.median() / expected_median, 1)
        logger.info("  lambda = {:.6f}".format(round(inflation_factor, 6)))


def check_args(args):
    """Check arguments and options.

    Args:
        args (argparse.Namespace): the arguments and options.

    """
    for fn in args.i_filenames:
        # Checking the file exists
        if not os.path.isfile(fn):
            logger.critical("{}: no such file".format(fn))
            sys.exit(1)

        # Checking the column exists for each file
        with open(fn, "r") as i_file:
            header = set(i_file.readline().rstrip("\r\n").split(args.delim))
            if args.field not in header:
                logger.critical("{}: no field named {}".format(fn, args.field))
                sys.exit(1)


def parse_args():
    """Argument parser."""
    # Creating the parser
    parser = argparse.ArgumentParser(
        description="Compute inflation factor (lambda) in GWAS results.",
    )

    # The version of the script
    parser.add_argument("-v", "--version", action="version",
                        version="lambda-" + __version__)

    # Add the INPUT group
    group = parser.add_argument_group("INPUT FILES")
    group.add_argument("-i", "--input", required=True, nargs="+",
                       metavar="FILE", dest="i_filenames",
                       help="The list of files containing GWAS results.")

    group.add_argument("-d", "--delim", default="\t", metavar="DELIM",
                       help="The field delimiter (default is a tabulation).")

    group.add_argument("-f", "--field", required=True, metavar="NAME",
                       help="The name of the field containing the statistics.")

    group.add_argument("--chi2", action="store_true",
                       help="Statistics were computed using a chi-squared "
                            "distribution.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
