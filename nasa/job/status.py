#!/usr/bin/env -S bin/conda/maap.sh python

"""
Get the status of a job.
"""

import argparse
import sys
from typing import Sequence

from maap.maap import MAAP  # type: ignore


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get the status of a MAAP DPS job")
    parser.add_argument("job_id", type=str, help="job ID", metavar="JOB_ID")

    return parser.parse_args(args)


if __name__ == "__main__":
    maap = MAAP()
    args = parse_args(sys.argv[1:])
    job_id = args.job_id
    status = maap.getJobStatus(job_id)

    print(status)
