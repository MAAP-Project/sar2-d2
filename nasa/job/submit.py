#!/usr/bin/env -S bin/conda/maap.sh python

"""
Submit a job using the NASA MAAP algorithm defined in the file nasa/algorithm.yml.

This ignores the value of `algorithm_version` within the YAML file, and uses one of the
following values as the version of the algorithm to run:

- `git tag --points-at HEAD`  # Git tag pointing to HEAD commit of current branch
- `git branch --show-current` # Name of current git branch

If there is no tag value, the branch name is used.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import requests
import yaml
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


def build_parser(
    parser: argparse.ArgumentParser, config: Mapping[str, Any]
) -> argparse.ArgumentParser:
    default_version = (
        subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True)
        .stdout.decode()
        .strip()
    ) or (
        subprocess.run(["git", "branch", "--show-current", "HEAD"], capture_output=True)
        .stdout.decode()
        .strip()
    )

    parser.add_argument(
        "--version",
        help="version of the algorithm to use",
        default=default_version,
    )
    parser.add_argument(
        "--queue",
        "-q",
        help="name of the job queue to use",
        default=config.get("queue"),
    )

    inputs = config.get("inputs", {})
    file_inputs = inputs.get("file", [])
    positional_inputs = inputs.get("positional", [])

    for file_input in file_inputs:
        add_file_argument(parser, file_input)

    for positional_input in positional_inputs:
        add_positional_argument(parser, positional_input)

    return parser


def add_file_argument(
    parser: argparse.ArgumentParser, input: Mapping[str, Any]
) -> argparse.ArgumentParser:
    # from urllib.parse import urlparse

    parser.add_argument(
        input["name"].upper(),
        # TODO use a function that checks for valid http(s) URLs
        type=str,
        help=input.get("description"),
        # Do NOT include 'required' option because all positional arguments are
        # required by argparse, so argparse throws an error when you include
        # this option.
        # required=input.get("required", False),
    )

    return parser


def add_positional_argument(
    parser: argparse.ArgumentParser, input: Mapping[str, Any]
) -> argparse.ArgumentParser:
    if default := input.get("default"):
        parser.add_argument(
            f"--{input['name'].replace('_', '-')}",
            help=input.get("description"),
            default=default,
            # Do NOT include 'required' option because all positional arguments are
            # required by argparse, so argparse throws an error when you include
            # this option.
            # required=input.get("required", False),
        )
    else:
        parser.add_argument(
            input["name"].upper(),
            help=input.get("description"),
            # Do NOT include 'required' option because all positional arguments are
            # required by argparse, so argparse throws an error when you include
            # this option.
            # required=input.get("required", False),
        )

    return parser


def parse_args(args: Sequence[str]) -> tuple[str, argparse.Namespace]:
    default_config_yaml = Path(__file__).parent.parent / "algorithm.yml"

    parser = argparse.ArgumentParser(
        description="Submit a MAAP DPS job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # TODO support alternative config yaml file (only nasa/algorithm.yml for now)
    # TODO see argparse.parse_known_args for help with this
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     help="path to algorithm configuration YAML file",
    #     metavar="YAML",
    #     default=default_config_yaml.relative_to(Path.cwd()),
    # )

    config = yaml.safe_load(default_config_yaml.read_text())
    name = config["algorithm_name"]

    build_parser(parser, config)

    return name, parser.parse_args(args)


if __name__ == "__main__":
    import json
    import os
    import subprocess

    name, args = parse_args(sys.argv[1:])
    version = args.version
    queue = args.queue
    inputs = {
        name: value
        for name, value in vars(args).items()
        if name not in {"version", "queue"}
    }

    maap = MAAP()
    username = get_username(maap)
    result = maap.submitJob(
        username=username,
        identifier=name,
        algo_id=name,
        version=args.version,
        queue=args.queue,
        **inputs,
    )
    job_id = result.id
    error_details = result.error_details

    if not job_id:
        die(f"{error_details}" if error_details else json.dumps(result, indent=2))

    print(f"Submitted job for algorithm {name}:{args.version} with job ID {job_id}")
