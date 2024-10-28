#!/usr/bin/env -S bin/conda/run.sh python

"""
Submit a job using the NASA MAAP algorithm defined in the file nasa/algorithm.yml.

This ignores the value of `algorithm_version` within the YAML file, and uses one of the
following values as the version of the algorithm to run:

- `git tag --points-at HEAD`  # Git tag pointing to HEAD commit of current branch
- `git branch --show-current` # Name of current git branch

If there is no tag value, the branch name is used.
"""

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

import requests
from maap.maap import MAAP  # type: ignore


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    exit(1)


def get_username(maap: MAAP) -> str:
    account_info = maap.profile.account_info()

    if not account_info or not (username := account_info.get("username")):
        die("Unable to determine your username")

    # Wrap username in f-string simply to satisfy type checkers, because the
    # username variable is typed as Any, not str.
    return f"{username}"


def to_url(path: str, username: str) -> str:
    if path.startswith("http"):
        return path
    if not Path(path).is_file():
        die(f"File not found: {path}")

    if path.startswith("/projects/my-public-bucket/"):
        relpath = f"{username}/{path.removeprefix('/projects/my-public-bucket/')}"
    elif path.startswith("/projects/shared-buckets/"):
        relpath = path.removeprefix("/projects/shared-buckets/")
    else:
        die(f"Unable to convert to a URL: {path}")

    url = f"https://maap-ops-workspace.s3.amazonaws.com/shared/{relpath}"
    r = requests.head(url)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        die(f"Failed to convert '{path}' to a URL: {e}")

    return url


def parse_args(username: str, args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a MAAP DPS job")
    parser.add_argument(
        "calibration_file",
        type=lambda arg: to_url(arg, username),
        help="path to calibration file",
        metavar="calibration-file",
    )
    parser.add_argument(
        "left",
        type=float,
        help="left longitude of your desired bounding box",
    )
    parser.add_argument(
        "bottom",
        type=float,
        help="bottom latitude of your desired bounding box",
    )
    parser.add_argument(
        "right",
        type=float,
        help="right longitude of your desired bounding box",
    )
    parser.add_argument(
        "top",
        type=float,
        help="top latitude of your desired bounding box",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    import json
    import os
    import subprocess

    maap = MAAP()
    username = get_username(maap)
    args = parse_args(username, sys.argv[1:])
    calibration_file = args.calibration_file
    bbox = f"{args.left} {args.bottom} {args.right} {args.top}"

    name = os.environ["CONDA_DEFAULT_ENV"]
    version = (
        subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True)
        .stdout.decode()
        .strip()
    ) or (
        subprocess.run(["git", "branch", "--show-current", "HEAD"], capture_output=True)
        .stdout.decode()
        .strip()
    )

    result = maap.submitJob(
        username=username,
        identifier=name,
        algo_id=name,
        version=version,
        queue="maap-dps-worker-32vcpu-64gb",
        calibration_file=calibration_file,
        bbox=bbox,
    )
    job_id = result.id
    error_details = result.error_details

    if not job_id:
        die(f"{error_details}" if error_details else json.dumps(result, indent=2))

    print(f"Submitted job for algorithm {name}:{version} with job ID {job_id}")
