#!/usr/bin/env -S conda run -n sar2-d2 python

"""
Register this NASA MAAP algorithm using the file nasa/algorithm.yml.
"""

import sys

from maap.maap import MAAP  # type: ignore
import requests


maap = MAAP()
response = maap.register_algorithm_from_yaml_file("nasa/algorithm.yml")

try:
    response.raise_for_status()
    print("Check registration progress at", response.json()["message"]["job_web_url"])
except requests.HTTPError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    exit(1)
