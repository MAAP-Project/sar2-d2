#!/usr/bin/env conda run -n sar2-d2 python

"""
Register this NASA MAAP algorithm using the file nasa/algorithm.yml.
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
        relpath = f"{username}/{path.lstrip('/projects/my-public-bucket/')}"
    elif path.startswith("/projects/shared-buckets/"):
        relpath = path.lstrip("/projects/shared-buckets/")
    else:
        die(f"Unable to convert to a URL: {path}")

    url = f"https://maap-ops-workspace.s3.amazonaws.com/shared/{relpath}"
    r = requests.head(url)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        die(f"Unable to reach '{url}': {e}")

    return url


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector"
    )
    parser.add_argument(
        "calibration_file",
        type=str,
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


args = parse_args(sys.argv[1:])
calibration_file = args.calibration_file
bbox = f"{args.left} {args.bottom} {args.right} {args.top}"

maap = MAAP()
job = maap.submitJob(
    identifier="sar2-d2",
    algo_id="sar2-d2",
    version="main",
    queue="maap-dps-worker-32vcpu-64gb",
    calibration_file=to_url(calibration_file, get_username(maap)),
    bbox=bbox,
)

print(job)
