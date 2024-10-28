#!/usr/bin/env -S conda run -n sar2-d2 python

"""
Register this NASA MAAP algorithm using the file nasa/algorithm.yml.

This ignores the value of `algorithm_version` within the YAML file, and sets the
version of the registered algorithm to one of the following values:

- `git tag --points-at HEAD`  # Git tag pointing to HEAD commit of current branch
- `git branch --show-current` # Name of current git branch

If there is no tag value, the branch name is used.
"""

import sys
import subprocess
import yaml
from pathlib import Path

from maap.maap import MAAP  # type: ignore
import requests


maap = MAAP()
config = yaml.safe_load(Path("nasa/algorithm.yml").read_text())

# Set the algorithm version to the git tag that points to the HEAD commit, or
# to the current branch, if there is no such tag.
config["algorithm_version"] = (
    subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True)
    .stdout.decode()
    .strip()
) or (
    subprocess.run(["git", "branch", "--show-current", "HEAD"], capture_output=True)
    .stdout.decode()
    .strip()
)

print(
    "Registering algorithm",
    f"{config['algorithm_name']}:{config['algorithm_version']} ...",
)

response = maap.registerAlgorithm(config)

try:
    response.raise_for_status()
    print("Check registration progress at", response.json()["message"]["job_web_url"])
except requests.HTTPError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    exit(1)
