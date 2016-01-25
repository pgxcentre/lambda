#!/usr/bin/env python3


import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

import scipy.stats
EXPECTED_MEDIAN = scipy.stats.chi2.ppf(0.5, 1)


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

        if args.p_value:
            if not args.one_sided:
                stats = 0.5 * stats

            stats = scipy.stats.norm.ppf(1 - stats)

        if not args.chi2:
            stats = stats ** 2

        inflation_factor = max(np.median(stats) / EXPECTED_MEDIAN, 1)
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

    if args.one_sided and not args.p_value:
        raise ValueError("The --one-sided option is only valid if the tool "
                         "is used on p-values.")

    if args.chi2 and args.p_value:
        raise ValueError("Can't use the --p-value option when the statistics "
                         "follow a chi-square distribution (not implemented).")


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

    group.add_argument("--p-value", "-p", action="store_true",
                       help="Flag to use the p-value instead of the statistic."
                            " This assumes a standard normal distribution for "
                            "the test statistic.")

    group.add_argument("--one-sided", "-os", action="store_true",
                       help="Flag for one-sided tests (when using p-values "
                            "to compute the inflation factor")

    return parser.parse_args()


if __name__ == "__main__":
    main()
