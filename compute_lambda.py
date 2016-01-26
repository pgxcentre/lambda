#!/usr/bin/env python3


import os
import re
import sys
import logging
import argparse

import numpy as np
import pandas as pd

import scipy.stats
EXPECTED_MEDIAN = scipy.stats.chi2.ppf(0.5, 1)


__copyright__ = "Copyright 2014, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "MIT"
__version__ = "0.2"


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

    # The column to extract
    cols_to_extract = [args.field]

    # Reading the markers to extract (if required)
    snp_to_extract = set()
    if args.extract:
        with open(args.extract, "r") as i_file:
            snp_to_extract = set(i_file.read().splitlines())
        cols_to_extract.append(args.snp_field)

    for fn in args.i_filenames:
        logger.info("Reading '{}'".format(fn))

        # The 'read_csv' options
        read_csv_options = dict(usecols=cols_to_extract)
        if args.whitespace:
            read_csv_options["delim_whitespace"] = True
        else:
            read_csv_options["sep"] = args.delim

        # Reading the file
        data = pd.read_csv(fn, **read_csv_options)

        # Removing the NAs
        before = data.shape[0]
        data = data.dropna()
        after = data.shape[0]
        logger.info("  - {:,d} NA values removed".format(before - after))

        # If required, keeping only the required markers
        if args.extract:
            data = data[data[args.snp_field].isin(snp_to_extract)]
            logger.info("  - {:,d} markers extracted".format(data.shape[0]))

        # Reading the column containing the values
        stats = data[args.field]

        # If we have p-values, we need to change them back to z values.
        if args.p_value:
            if not args.one_sided:
                logger.info("  - computing two-sided statistics from p-values")
                stats = 0.5 * stats
            else:
                logger.info("  - computing one-sided statistics from p-values")
            stats = scipy.stats.norm.ppf(1 - stats)

        # If not a chi-squared distribution
        if not args.chi2:
            logger.info("  - using z/t statistics")
            stats = stats ** 2
        else:
            logger.info("  - using chi-squared statistics")

        # Computing the inflation factor using the statistics
        logger.info("  - computing inflation factor")
        inflation_factor = max(np.median(stats) / EXPECTED_MEDIAN, 1)
        logger.info("  - lambda = {:.6f}".format(round(inflation_factor, 6)))


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
            regex = re.compile(r"\s+" if args.whitespace else "\t")
            header = set(regex.split(i_file.readline().rstrip("\r\n")))
            if args.field not in header:
                logger.critical(
                    "{}: no field named '{}'".format(fn, args.field)
                )
                sys.exit(1)

            if args.extract:
                if args.snp_field not in header:
                    logger.critical(
                        "{}: no field named '{}'".format(fn, args.snp_field)
                    )
                    sys.exit(1)

    # Checking the file containing the markers to extract exists (if required)
    if args.extract is not None:
        if not os.path.isfile(args.extract):
            logger.critical("{}: no such file".format(args.extract))
            sys.exit(1)

    if args.one_sided and not args.p_value:
        logger.critical("The --one-sided option is only valid if the tool is "
                        "used on p-values.")
        sys.exit(1)

    if args.chi2 and args.p_value:
        logger.critical("Can't use the --p-value option when the statistics "
                        "follow a chi-square distribution (not implemented).")
        sys.exit(1)


def parse_args():
    """Argument parser."""
    # Creating the parser
    parser = argparse.ArgumentParser(
        description="Compute inflation factor (lambda) in GWAS results.",
    )

    # The version of the script
    parser.add_argument("-v", "--version", action="version",
                        version="lambda version " + __version__)

    # Add the INPUT group
    group = parser.add_argument_group("INPUT FILES")
    group.add_argument("-i", "--input", required=True, nargs="+",
                       metavar="FILE", dest="i_filenames",
                       help="The list of files containing GWAS results.")

    group.add_argument("-d", "--delim", default="\t", metavar="DELIM",
                       help="The field delimiter (default is a tabulation).")

    group.add_argument("-w", "--whitespace", action="store_true",
                       help="The file is delimited by white spaces "
                            "(e.g. Plink results).")

    group.add_argument("-f", "--field", required=True, metavar="NAME",
                       help="The name of the field containing the statistics.")

    group.add_argument("--snp-field", metavar="NAME", default="snp",
                       help="The name of the field containing the SNP name.")

    # Adding general options
    group = parser.add_argument_group("GENERAL OPTIONS")
    group.add_argument("--chi2", action="store_true",
                       help="Statistics were computed using a chi-squared "
                            "distribution.")

    group.add_argument("--p-value", "-p", action="store_true",
                       help="Flag to use the p-value instead of the statistic."
                            " This assumes a standard normal distribution for "
                            "the test statistic.")

    group.add_argument("--one-sided", action="store_true",
                       help="Flag for one-sided tests (when using p-values "
                            "to compute the inflation factor)")

    # Add subset options
    group = parser.add_argument_group("SUBSET OPTIONS")
    group.add_argument("-e", "--extract", metavar="FILE",
                       help="A file containing markers to extract for the "
                            "analysis (only one marker per line).")

    return parser.parse_args()


if __name__ == "__main__":
    main()
