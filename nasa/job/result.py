#!/usr/bin/env -S conda run -n sar2-d2 python

"""
Get the result of a job.
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
    result = maap.getJobResult(job_id)

    print("\n".join(result) if result else "No result found")
